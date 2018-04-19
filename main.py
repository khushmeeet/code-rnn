import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
import arg_parser
from mLSTM import Stacked_mLSTM, mLSTM
from settings import model_settings
import utils
import time
import os


options = arg_parser.parser.parse_args()

lr = model_settings['learning_rate']
layers = model_settings['layers']
batch_size = model_settings['batch_size']
rnn_size = model_settings['rnn_size']
embed_size = model_settings['embed_size']
seq_length = model_settings['seq_length']
dropout = model_settings['dropout']
data_size = 256 # ???

train_x = utils.tokenize(options.train_data)
train_x = utils.batchify(train_x, batch_size)
num_batches = train_x.size(0)//seq_length

if len(options.load_model) > 0:
    checkpoint = torch.load(options.load_model)
    embedding = checkpoint['embed']
    model = checkpoint['rnn']
else:
    embedding = nn.Embedding(256, embed_size)
    model = Stacked_mLSTM(mLSTM, layers, embed_size, rnn_size, data_size, dropout)

loss_fn = nn.CrossEntropyLoss()
embed_optimizer = optim.SGD(embedding.parameters(), lr=lr)
model_optimizer = optim.SGD(model.parameters(), lr=lr)

n_params = sum([p.nelement() for p in model.parameters()])
print('Total number of parameters:', n_params)
print('Total number of batches:', num_batches)
print()
print('Embedding Summary:')
print(embedding)
print()
print('RNN Summary:')
print(model)


def train_model(epoch):
    hidden_init = model.state0(batch_size)    		
    if options.cuda:
	    embedding.cuda()
	    model.cuda()
	    hidden_init = utils.make_cuda(hidden_init)

    loss_avg = 0

    for s in range(num_batches-1):
        embed_optimizer.zero_grad()
        model_optimizer.zero_grad()
        batch = Variable(train_x.narrow(0,s*seq_length,seq_length+1).long())
        start = time.time()
        hidden = hidden_init
        if options.cuda:
            batch = batch.cuda()
        loss = 0
        for t in range(seq_length):                  
            emb = embedding(batch[t])
            hidden, output = model(emb, hidden)
            loss += loss_fn(output, batch[t+1])
        
        loss.backward()

        hidden_init = utils.copy_state(hidden)
        gn = utils.calc_grad_norm(model)
        utils.clip_gradient(model, model_settings['clip_gradient'])
        utils.clip_gradient(embedding, model_settings['clip_gradient'])
        embed_optimizer.step()
        model_optimizer.step()
        loss_avg = .99*loss_avg + .01*loss.data[0]/seq_length
        
        if s % 10 == 0:
            print(f'epoch {epoch} | batch {s}/{num_batches} | loss {loss.data[0] / seq_length} | avg loss {loss_avg} | time {time.time() - start}')


if __name__ == '__main__':
    for e in options.epochs:
        try:
            train_model(e)
            lr *= 0.7
            utils.update_lr(model_optimizer, lr)
            utils.update_lr(embed_optimizer, lr)
        except KeyboardInterrupt:
            print('KeyboardInterrupt occured, saving the model')
            checkpoint = {
                'model': model,
                'embedding': embedding,
                'epoch': e
            }
            if options.paperspace:
                save_file = f'/artifacts/{options.save_model}_epoch{e}.pt'
            else:
                if os.path.exists('./saved_models/'):
                    save_file = f'./saved_models/{options.save_model}_epoch{e}.pt'
                else:
                    os.mkdir('./saved_models/')
                    save_file = f'./saved_models/{options.save_model}_epoch{e}.pt'
            torch.save(checkpoint, save_file)
            break
    

import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
from tensorboardX import SummaryWriter
import args
from mLSTM import Stacked_mLSTM, mLSTM
from settings import model_settings
import utils
import time
import os


options = args.parser.parse_args()
writer = SummaryWriter(log_dir='./logs')


def train_model(epoch):
    i = 0
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
            loss_step = loss_fn(output, batch[t + 1])
            loss += loss_step
            writer.add_scalar('loss per step', loss_step, i)
            i += 1
        
        writer.add_scalar('loss per batch ', loss, s)
        
        loss.backward()

        hidden_init = utils.copy_state(hidden)
        gn = utils.calc_grad_norm(model)
        utils.clip_gradient(model, model_settings['clip_gradient'])
        utils.clip_gradient(embedding, model_settings['clip_gradient'])
        embed_optimizer.step()
        model_optimizer.step()
        loss_avg = .99*loss_avg + .01*loss.data[0]/seq_length
        
        if s % 10 == 0:
            print(f'epoch: {epoch} | batch: {s}/{num_batches} | step loss: {loss.data[0] / seq_length} | batch loss: {loss.data[0]} | avg loss: {loss_avg} | time: {time.time() - start}s')


def generation(embedding, model, state, n, primer):
    sample = [c for c in primer]
    for c in primer:
        x = torch.ByteTensor([1])
        for l in c.encode():
            x[0] = l
        if options.cuda:
            x = Variable(x.long()).cuda()
        else:
            x = Variable(x.long())
        emb = embedding(x)
        hidden, output = model(emb, state)
    
    _, indices = output.data.topk(1)
    out_char = indices[0][0]
    sample.append(out_char)
    hidden = hidden
    next_input = Variable(indices[0], volatile=True)
    if options.cuda:
        next_input.cuda()


    for _ in range(int(n)):
        emb = embedding(next_input)
        hidden, output = model(emb, hidden)
        _, indices = output.data.topk(1)
        out_char = indices[0][0]
        sample.append(out_char)
        next_input = Variable(indices[0])
        if options.cuda:
            next_input.cuda()
    
    return ''.join(chr(i) for i in sample)


if __name__ == '__main__':
    if options.test:
        checkpoint = torch.load(options.load_model)
        embedding = checkpoint['embedding']
        model = checkpoint['model']

        state = model.state0(batch_size)
        if options.cuda:
            state = utils.make_cuda(state)
            embedding.cuda()
            model.cuda()

        gen_text = generation(embedding, model, state, options.n, options.primer)
        print(gen_text)
    else:
        lr = model_settings['learning_rate']
        layers = model_settings['layers']
        batch_size = model_settings['batch_size']
        rnn_size = model_settings['rnn_size']
        embed_size = model_settings['embed_size']
        seq_length = model_settings['seq_length']
        dropout = model_settings['dropout']
        data_size = 256  # ???

        train_x = utils.tokenize(options.train_data)
        train_x = utils.batchify(train_x, batch_size)
        num_batches = train_x.size(0) // seq_length

        if len(options.load_model) > 0:
            checkpoint = torch.load(options.load_model)
            embedding = checkpoint['embed']
            model = checkpoint['rnn']
        else:
            embedding = nn.Embedding(256, embed_size)
            model = Stacked_mLSTM(mLSTM, layers, embed_size,
                                rnn_size, data_size, dropout)

        loss_fn = nn.CrossEntropyLoss()
        embed_optimizer = optim.Adam(embedding.parameters(), lr=lr)
        model_optimizer = optim.Adam(model.parameters(), lr=lr)

        n_params = sum([p.nelement() for p in model.parameters()])
        print('Total number of parameters:', n_params)
        print('Total number of batches:', num_batches)
        print()
        print('Embedding Summary:')
        print(embedding)
        print()
        print('RNN Summary:')
        print(model)

        for e in range(int(options.epochs)):
            try:
                train_model(e)
                lr *= 0.7
                utils.update_lr(model_optimizer, lr)
                utils.update_lr(embed_optimizer, lr)
            except KeyboardInterrupt:
                print('KeyboardInterrupt occured, saving the model')
                utils.save_model(options, model, embedding, e)
                writer.export_scalars_to_json("./all_scalars.json")
                writer.close()
                break
        utils.save_model(options, model, embedding, e)

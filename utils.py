import os
from torch.autograd import Variable
import torch
import math
import numpy as np


def tokenize(path):
    assert os.path.exists(path)

    with open(path, 'r') as f:
        tokens = 0
        for line in f:
            tokens += len(line.encode())
    
    with open(path, 'r') as f:
        file_bytes = torch.ByteTensor(tokens)
        token = 0
        for line in f:
            for char in line.encode():
                file_bytes[token] = char
                token += 1
    
    return file_bytes


def batchify(data, batch_size):
    nbatch = data.size(0) // batch_size
    data = data.narrow(0, 0, nbatch * batch_size)
    data = data.view(batch_size, -1).t().contiguous()
    return data


def make_test_tensor(data, batch_size):
    tokens = len(data.encode())
    ids = torch.ByteTensor(tokens)
    token = 0
    for char in data.encode():
        ids[token] = char
        token += 1
    nbatch = ids.size(0) // batch_size
    ids = ids.narrow(0, 0, nbatch * batch_size)
    ids = ids.view(batch_size, -1).t().contiguous()
    return ids


def update_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr
    return


def clip_gradient_coeff(model, clip):
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, clip / (totalnorm + 1e-6))


def calc_grad_norm(model):
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    return math.sqrt(totalnorm)


def clip_gradient(model, clip):
    totalnorm = 0
    for p in model.parameters():
        p.grad.data = p.grad.data.clamp(-clip, clip)


def make_cuda(state):
    if isinstance(state, tuple):
    	return (state[0].cuda(), state[1].cuda())
    else:
    	return state.cuda()


def copy_state(state):
    if isinstance(state, tuple):
    	return (Variable(state[0].data), Variable(state[1].data))
    else:
    	return Variable(state.data)


# def pick_top_n(preds, vocab_size, top_n=5):
#     p = np.squeeze(preds)
#     p[np.argsort(p)[:-top_n]] = 0
#     p = p / np.sum(p)
#     c = np.random.choice(vocab_size, 1, p=p)[0]
#     return c

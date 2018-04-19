import os
from torch.autograd import Variable
import torch
import math


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


def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data


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

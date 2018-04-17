import os
import torch


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




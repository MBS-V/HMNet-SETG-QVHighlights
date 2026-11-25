import torch
def get_device(requested='auto'):
    if requested=='auto': return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d=torch.device(requested)
    if d.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    return d

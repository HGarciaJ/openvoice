import os
import torch
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, optimizer, epoch, path):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    torch.save(state, path)

def load_checkpoint(path, model, optimizer):
    state = torch.load(path)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    return state['epoch']

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def move_to_device(tensor, device):
    if isinstance(tensor, (list, tuple)):
        return [move_to_device(t, device) for t in tensor]
    return tensor.to(device)

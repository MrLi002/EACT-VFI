"""Small helpers: seeding, padding, checkpoints, logging."""
import os
import random

import numpy as np


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_int_list(s):
    if isinstance(s, (list, tuple)):
        return [int(v) for v in s]
    return [int(v) for v in str(s).replace(' ', '').split(',') if v != '']


def pad_to_multiple(tensor, multiple=8):
    """Zero-pad (B, C, H, W) on the bottom/right; returns padded tensor and (H, W)."""
    import torch.nn.functional as F
    h, w = tensor.shape[-2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return tensor, (h, w)
    return F.pad(tensor, [0, pw, 0, ph], mode='replicate' if tensor.dim() == 4 else 'constant'), (h, w)


def strip_module(state):
    return {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in state.items()}


def save_checkpoint(path, model, optimizer, scheduler, epoch, args):
    import torch
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
        'args': vars(args) if args is not None else None,
    }, path)


def write_log(text, path, echo=False):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a') as f:
        f.write(text + '\n')
    if echo:
        print(text)


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, v, n=1):
        self.sum += float(v) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)

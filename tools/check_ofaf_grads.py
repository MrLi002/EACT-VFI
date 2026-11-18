"""Which parameters actually receive a gradient?  (acceptance check for the BiW mask fix)

In OF-AF the forward flows F_{0->t}, F_{1->t} are used only in BiW, through
    hole = 1 - s / (s + 1e-7),  s = sum_splat(1, F)  -> derivative 1e-7 / (s + 1e-7)^2
    diff = [ |I_{t<-0} - avg_splat(I_0, F)| > 0.1 ]  -> hard threshold, zero derivative
so the forward flow head (the GRU + `flow` conv of each BidirFlow block) gets ~1e-7 of a
normal gradient from the reconstruction loss, except on rare pixels whose coverage s is tiny
(1e-6..1e-4), where the epsilon term spikes.  Neither is a useful learning signal.  With --mask_mode geo the coverage and the soft diff
mask are differentiable, and with --splat ebc also alpha and C get a gradient.

    # this repository, CPU is enough
    python tools/check_ofaf_grads.py --impl eact
    # the released OF-AF code (needs CUDA + cupy, like OF-AF itself)
    python tools/check_ofaf_grads.py --impl ofaf --ofaf_root /path/to/OF-AF

Prints the gradient L2 norm of every module group (first two name components).
"""
import argparse
import os
import sys
from collections import OrderedDict
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EACT_CONFIGS = OrderedDict(
    a0_ofaf=dict(query='ofaf', query_res='none', splat='avg', mask_mode='ofaf', time_cond=False),
    a0_geo=dict(query='linear', query_res='none', splat='avg', mask_mode='geo', time_cond=False),
    full_ebc_geo=dict(query='linear', query_res='bwd', splat='ebc', mask_mode='geo', time_cond=True),
    ebc_ofafmask=dict(query='linear', query_res='bwd', splat='ebc', mask_mode='ofaf', time_cond=True),
)


def group_norms(model):
    groups = OrderedDict()
    for name, p in model.named_parameters():
        g = '.'.join(name.split('.')[:2])
        sq = 0.0 if p.grad is None else float(p.grad.detach().double().pow(2).sum())
        groups[g] = groups.get(g, 0.0) + sq
    return OrderedDict((k, v ** 0.5) for k, v in groups.items())


def report(title, norms, keys=('flow', 'gru', 'log_alpha', 'threshold', 'diff_tau', 'residual', 'conv2')):
    print(f'\n== {title}')
    for k, v in norms.items():
        flag = '  <-- zero' if v == 0.0 else ''
        if any(s in k for s in keys) or v == 0.0:
            print(f'  {k:40s} {v:.3e}{flag}')


def run_eact(args):
    sys.path.insert(0, ROOT)
    from models import build_model
    torch.manual_seed(0)
    B, H, W, n, bps = 2, 64, 64, 4, 4
    im0, im1 = torch.rand(B, 3, H, W), torch.rand(B, 3, H, W)
    vox = torch.randn(B, n * bps, H, W)
    times = torch.tensor([[0.3, 0.6], [0.45, 0.8]])
    gt = torch.rand(B, 2, 3, H, W)
    qfeat = dict(q_before=torch.randn(B, 2, bps, H, W), q_s0t=torch.randn(B, 2, 1, H, W),
                 q_a0t=torch.rand(B, 2, 1, H, W) * 5, s_tot=torch.randn(B, 1, H, W), a_tot=torch.rand(B, 1, H, W) * 10)
    for name, cfg in EACT_CONFIGS.items():
        a = SimpleNamespace(voxel_bins=n * bps, nb_of_flow=n, hermite_on='fwd', res_window='bump',
                            query_res_ctx=False, diff_tau=0.1, diff_temp=0.02, ebc_c_init=0.2, ebc_learn_c=True,
                            ebc_pixel_c=True, ebc_fuse=True, ebc_alpha=5.0, **cfg)
        model = build_model(a)
        # a zero-initialised residual head would hide the gradient of everything before it
        if hasattr(model.query, 'residual'):
            torch.nn.init.normal_(model.query.residual.flow_out.weight, std=1e-2)
        out = model(im0, im1, vox, times, None, qfeat=qfeat)
        F.l1_loss(out, gt).backward()
        report(name, group_norms(model))


def run_ofaf(args):
    sys.path.insert(0, os.path.abspath(args.ofaf_root))
    from models.model import MyModel          # OF-AF
    torch.manual_seed(0)
    a = SimpleNamespace(nb_of_flow=args.nb_of_flow, voxel_bins=args.voxel_bins)
    model = MyModel(a).cuda().train()
    B, H, W = 1, 128, 128
    im0, im1 = torch.rand(B, 3, H, W).cuda(), torch.rand(B, 3, H, W).cuda()
    vox = torch.randn(B, args.voxel_bins, H, W).cuda()
    times = torch.tensor([[0.25, 0.5, 0.75]]).cuda()
    out = model(im0, im1, vox, times)
    F.l1_loss(out, torch.rand_like(out)).backward()
    report('OF-AF released model', group_norms(model))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--impl', default='eact', choices=['eact', 'ofaf'])
    ap.add_argument('--ofaf_root', default='')
    ap.add_argument('--nb_of_flow', default=16, type=int)
    ap.add_argument('--voxel_bins', default=128, type=int)
    args = ap.parse_args()
    run_eact(args) if args.impl == 'eact' else run_ofaf(args)

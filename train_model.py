import argparse
import os

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataset
from datasets.base import QFEAT_KEYS
from models import build_model
from models.loss import VFILoss
from options import add_model_args, add_partition_args, add_data_args, add_eval_args, str2bool
from util.misc import seed_everything, save_checkpoint, strip_module, write_log, AverageMeter, parse_int_list


def load_weights(args, model, optimizer, scheduler):
    epoch = 0
    if args.weights_path:
        state = torch.load(args.weights_path, map_location='cpu')
        sd = strip_module(state['model'])
        if args.only_weights:
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print('Missing:', missing)
            print('Unexpected:', unexpected)
        else:
            model.load_state_dict(sd)
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            epoch = state['epoch'] + 1
    return epoch


GRAD_GROUPS = {
    # forward flow head F_{0->t}, F_{1->t}: only reached through BiW (splat, coverage, diff mask)
    'fwd_flow_head': ('bidir_flow0.gru.', 'bidir_flow0.flow.', 'bidir_flow1.gru.', 'bidir_flow1.flow.'),
    'bwd_flow_head': ('bidir_flow0.conv2.', 'bidir_flow1.conv2.'),
    'ctq_residual': ('query.residual.',),
    'C': ('target.threshold.',),
    'beta': ('target.log_beta',),
    'alpha': ('biw.log_alpha',),
    'diff_tau': ('biw.fuse_warp.diff_tau',),
}


def grad_norms(model):
    """L2 norm of the gradient of each parameter group (nan if the group does not exist)."""
    m = model.module if hasattr(model, 'module') else model
    out = {}
    for name, prefixes in GRAD_GROUPS.items():
        sq, found = 0.0, False
        for pname, p in m.named_parameters():
            if pname.startswith(prefixes):
                found = True
                if p.grad is not None:
                    sq += float(p.grad.detach().pow(2).sum())
        out[name] = sq ** 0.5 if found else float('nan')
    return out


def check_batching(args):
    skips = parse_int_list(args.train_skips)
    variable_t = args.targets_per_sample <= 0 and len(set(skips)) > 1
    if args.batch_size > 1 and variable_t:
        raise SystemExit('Several train skips with all targets give a different number of frames per sample. '
                         'Use --batch_size 1 (as OF-AF) or set --targets_per_sample K <= min(skip).')
    if args.targets_per_sample > 0 and args.targets_per_sample > min(skips):
        raise SystemExit('--targets_per_sample must be <= the smallest train skip.')


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(args).to(device)
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {n_all}  Trainable params: {n_train}')

    check_batching(args)
    dataset = get_dataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=True,
                        drop_last=True, pin_memory=True, persistent_workers=args.num_worker > 0)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch * len(loader), eta_min=1e-7)
    epoch = load_weights(args, model, optimizer, scheduler)
    criterion = VFILoss(args.lpips_weight, 'vgg').to(device)

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, 'train_log.txt')
    write_log(str(vars(args)), log_path)

    if args.val_dataset != 'none':
        from evaluator import run_eval, make_lpips
        val_lpips = make_lpips(args, device)

    it = 0
    while epoch < args.epoch:
        model.train()
        meter = AverageMeter()
        bar = tqdm(loader, desc=f'Train Epoch({epoch}/{args.epoch})')
        for batch in bar:
            im0 = batch['im0'].to(device, non_blocking=True)
            im1 = batch['im1'].to(device, non_blocking=True)
            gts = batch['gts'].to(device, non_blocking=True)
            voxel = batch['voxel'].to(device, non_blocking=True)
            bounds = batch['bounds'].to(device, non_blocking=True)
            times = batch['times'].to(device, non_blocking=True)
            qfeat = None
            if all(k in batch for k in QFEAT_KEYS):
                qfeat = {k: batch[k].to(device, non_blocking=True) for k in QFEAT_KEYS}

            pred = model(im0, im1, voxel, times, bounds, qfeat=qfeat)
            loss, logs = criterion(pred, gts)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_report > 0 and it % args.grad_report == 0:
                g = grad_norms(model)
                write_log(f'epoch {epoch} iter {it} grad-norm ' + ' '.join(f'{k}={v:.3e}' for k, v in g.items()),
                          log_path, echo=False)
            it += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

            meter.update(logs['total'])
            bar.set_postfix({'loss': f'{meter.avg:.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'})

        save_checkpoint(os.path.join(args.save_dir, f'epoch{epoch}.pth'), model, optimizer, scheduler, epoch, args)
        write_log(f'epoch {epoch} loss {meter.avg:.5f}', log_path, echo=True)

        if args.val_dataset != 'none' and (epoch + 1) % args.val_every == 0:
            summ = run_eval(model, args, args.val_dataset, parse_int_list(args.val_skips),
                            os.path.join(args.save_dir, 'val', f'epoch{epoch}'), device, val_lpips, verbose=False)
            for skip, s in summ.items():
                write_log(f'epoch {epoch} val {args.val_dataset} skip{skip}: PSNR {s["psnr"]:.2f} '
                          f'(on-grid {s["psnr_on_grid"]:.2f}, off-grid {s["psnr_off_grid"]:.2f}) '
                          f'SSIM {s["ssim"]:.4f}', log_path, echo=True)
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--clip', default=0.1, type=float, help='grad-norm clip (OF-AF: 0.1)')
    parser.add_argument('--lpips_weight', default=0.1, type=float)
    parser.add_argument('--only_weights', action='store_true', help='only load model weights')
    parser.add_argument('--weights_path', default=None, type=str)
    parser.add_argument('--save_dir', default='train/eact', type=str)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--crop', default=256, type=int)
    parser.add_argument('--dataset', default='gopro', choices=['gopro', 'bsergb'])
    parser.add_argument('--train_skips', default='31', type=str,
                        help="comma list, e.g. '31' (OF-AF GoPro), '1,3' (OF-AF BS-ERGB), '3,4,5,6,7,9,11,15' (A2)")
    parser.add_argument('--skip_sampling', default='concat', choices=['concat', 'random'],
                        help='concat: one dataset per skip (OF-AF); random: draw a skip per sample')
    parser.add_argument('--targets_per_sample', default=-1, type=int, help='-1: all intermediate frames')
    parser.add_argument('--aug_reverse', default=False, type=str2bool, help='random time reversal')
    parser.add_argument('--val_dataset', default='none', choices=['none', 'gopro', 'bsergb'])
    parser.add_argument('--val_skips', default='7', type=str)
    parser.add_argument('--val_every', default=1, type=int)
    parser.add_argument('--grad_report', default=0, type=int,
                        help='every N iterations log the gradient norm of the forward flow head, CTQ residual, '
                             'C, beta, alpha, diff_tau (0: off)')
    add_model_args(parser)
    add_partition_args(parser)
    add_data_args(parser)
    add_eval_args(parser)
    args = parser.parse_args()
    if args.val_dataset != 'none' and args.max_windows < 0:
        args.max_windows = 20

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    train(args)

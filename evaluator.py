"""Evaluation loop shared by test_model.py and the per-epoch validation in train_model.py.

Writes for every (dataset, skip):
    <out_dir>/<dataset>/<skip>/frames.csv   one row per target frame, with t and its
                                            position in the uniform 1/n grid (grid_frac)
    <out_dir>/<dataset>/<skip>/summary.txt  overall / on-grid / off-grid means
With --splat_diag the rows also carry the forward-splat projection error |I_{src->t} - I_t|
(x255, mean over both sides) in collision pixels (coverage > --collision_thr, many-to-one)
and in single-hit pixels (0.5 < coverage <= collision_thr).  This is the quantity the splat
metric (avg / ebc / niklaus) is supposed to change; compare it with mask_mode fixed.
"""
import csv
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_eval_sets
from datasets.base import QFEAT_KEYS, qfeat_from_raw
from util.metrics import psnr, ssim, to_uint8, LPIPSMetric, consecutive_warp_error, grid_fraction
from util.misc import write_log


def _pad(x, multiple, mode):
    h, w = x.shape[-2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return x
    if mode == 'replicate':
        return F.pad(x, [0, pw, 0, ph], mode='replicate')
    return F.pad(x, [0, pw, 0, ph], mode='constant', value=0.0)


def _pad_qfeat(qfeat, multiple, device):
    if qfeat is None:
        return None
    return {k: _pad(v.unsqueeze(0).to(device), multiple, 'zero') for k, v in qfeat.items()}


def sample_qfeat(sample):
    if all(k in sample for k in QFEAT_KEYS):
        return {k: sample[k] for k in QFEAT_KEYS}
    return None


@torch.no_grad()
def predict(model, sample, times, device, multiple=8, qfeat=None, return_aux=False):
    """Run the model on one full-resolution sample; returns (T, 3, H, W) in [0, 1]
    (and, with return_aux, a list of per-time BiW diagnostics cropped to H x W)."""
    im0 = sample['im0'].unsqueeze(0).to(device)
    im1 = sample['im1'].unsqueeze(0).to(device)
    vox = sample['voxel'].unsqueeze(0).to(device)
    bounds = sample['bounds'].unsqueeze(0).to(device)
    h, w = im0.shape[-2:]
    r = model(_pad(im0, multiple, 'replicate'), _pad(im1, multiple, 'replicate'), _pad(vox, multiple, 'zero'),
              times.view(1, -1).to(device), bounds, qfeat=_pad_qfeat(qfeat, multiple, device),
              return_aux=return_aux)
    if not return_aux:
        return r[0, :, :, :h, :w].clamp(0, 1)
    out, auxs = r
    auxs = [{k: v[0, :, :h, :w] for k, v in a.items()} for a in auxs]
    return out[0, :, :, :h, :w].clamp(0, 1), auxs


def splat_errors(aux, gt, thr):
    """Mean |I_{src->t} - I_t| * 255 over collision / single-hit pixels, both sides pooled."""
    sums = dict(coll=0.0, single=0.0)
    cnts = dict(coll=0, single=0)
    for side in ('0', '1'):
        cov = aux['cov' + side][0]
        err = (aux['splat' + side] - gt).abs().mean(0) * 255.0
        for name, m in (('coll', cov > thr), ('single', (cov > 0.5) & (cov <= thr))):
            sums[name] += float(err[m].sum())
            cnts[name] += int(m.sum())
    total = 2 * gt.shape[-1] * gt.shape[-2]
    return dict(coll_frac=cnts['coll'] / total,
                err_coll=sums['coll'] / cnts['coll'] if cnts['coll'] else float('nan'),
                err_single=sums['single'] / cnts['single'] if cnts['single'] else float('nan'))


def _first(batch):
    return batch[0]


def _mean(xs):
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float('nan')


@torch.no_grad()
def run_eval(model, args, dataset, skips, out_dir, device='cuda', lpips_metric=None, verbose=True):
    model.eval()
    summaries = {}
    n = args.nb_of_flow
    for skip in skips:
        skip_dir = os.path.join(out_dir, dataset, str(skip))
        os.makedirs(skip_dir, exist_ok=True)
        csv_path = os.path.join(skip_dir, 'frames.csv')
        rows, dense_errs = [], []
        for ds in get_eval_sets(args, dataset, skip):
            if len(ds) == 0:
                continue
            loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_worker,
                                collate_fn=_first)
            for sample in tqdm(loader, desc=f'{dataset} skip{skip} {ds.name}', disable=not verbose, leave=False):
                qf = sample_qfeat(sample)
                if args.splat_diag:
                    preds, auxs = predict(model, sample, sample['times'], device, qfeat=qf, return_aux=True)
                else:
                    preds = predict(model, sample, sample['times'], device, qfeat=qf)
                gts = sample['gts']
                for i in range(preds.shape[0]):
                    t = float(sample['times'][i])
                    pu8, gu8 = to_uint8(preds[i]), to_uint8(gts[i])
                    lp = lpips_metric(preds[i:i + 1], gts[i:i + 1]) if lpips_metric is not None else float('nan')
                    row = dict(scene=ds.name, name=sample['names'][i], t=round(t, 6),
                               grid_frac=round(grid_fraction(t, n), 6),
                               psnr=psnr(gu8, pu8), ssim=ssim(gu8, pu8), lpips=lp)
                    if args.splat_diag:
                        row.update(splat_errors(auxs[i], gts[i].to(preds.device), args.collision_thr))
                    rows.append(row)
                    if args.save_pred:
                        p = os.path.join(skip_dir, 'pred', ds.name, sample['names'][i])
                        os.makedirs(os.path.dirname(p), exist_ok=True)
                        cv2.imwrite(p, cv2.cvtColor(pu8, cv2.COLOR_RGB2BGR))
                if args.dense > 0:
                    dt = torch.arange(1, args.dense + 1, dtype=torch.float32) / (args.dense + 1)
                    seq = [to_uint8(sample['im0'])]
                    H, W = sample['im0'].shape[-2:]
                    for chunk in dt.split(16):
                        qf = None
                        if args.query_events == 'exact' and 'raw' in sample:
                            qf = qfeat_from_raw(sample['raw'], sample['bounds'].numpy(), model_bps(model),
                                                H, W, chunk.tolist())
                        seq += [to_uint8(x) for x in predict(model, sample, chunk, device, qfeat=qf)]
                    seq.append(to_uint8(sample['im1']))
                    dense_errs.append(consecutive_warp_error(seq))

        with open(csv_path, 'w', newline='') as f:
            fields = ['scene', 'name', 't', 'grid_frac', 'psnr', 'ssim', 'lpips']
            if args.splat_diag:
                fields += ['coll_frac', 'err_coll', 'err_single']
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            wr.writerows(rows)

        on = [r for r in rows if r['grid_frac'] == 0.0]
        off = [r for r in rows if r['grid_frac'] != 0.0]
        summ = dict(
            n_frames=len(rows), psnr=_mean([r['psnr'] for r in rows]), ssim=_mean([r['ssim'] for r in rows]),
            lpips=_mean([r['lpips'] for r in rows]),
            n_on_grid=len(on), psnr_on_grid=_mean([r['psnr'] for r in on]),
            n_off_grid=len(off), psnr_off_grid=_mean([r['psnr'] for r in off]),
            dense_warp_err=_mean(dense_errs),
        )
        if args.splat_diag:
            summ.update(coll_frac=_mean([r['coll_frac'] for r in rows]),
                        err_coll=_mean([r['err_coll'] for r in rows]),
                        err_single=_mean([r['err_single'] for r in rows]),
                        err_coll_off_grid=_mean([r['err_coll'] for r in off]))
        summaries[skip] = summ
        line = (f'[{dataset} skip{skip}] frames={summ["n_frames"]} PSNR={summ["psnr"]:.2f} SSIM={summ["ssim"]:.4f} '
                f'LPIPS={summ["lpips"]:.4f} | on-grid({summ["n_on_grid"]}) PSNR={summ["psnr_on_grid"]:.2f} '
                f'| off-grid({summ["n_off_grid"]}) PSNR={summ["psnr_off_grid"]:.2f}'
                + (f' | dense warp err={summ["dense_warp_err"]:.3f}' if args.dense > 0 else '')
                + (f' | splat L1 collision={summ["err_coll"]:.2f} ({summ["coll_frac"]:.1%} px) '
                   f'single={summ["err_single"]:.2f}' if args.splat_diag else ''))
        write_log(line, os.path.join(skip_dir, 'summary.txt'), echo=verbose)
    return summaries


def model_bps(model):
    m = model.module if hasattr(model, 'module') else model
    return m.bins_per_seg


def make_lpips(args, device):
    return LPIPSMetric('alex', device) if args.lpips else None

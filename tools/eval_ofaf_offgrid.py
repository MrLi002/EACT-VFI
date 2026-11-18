"""Week-1 experiment: evaluate the *official* OF-AF code and weights at grid and off-grid times.

It imports OF-AF's own model, event loader and voxelization from --ofaf_root, so the
numbers are produced by the released implementation (nothing from this repository is
imported).  With n = 16 segments:
    skip 1, 3, 7, 15 -> every target time lies on the 1/16 grid (what OF-AF reports)
    skip 2, 4, 5, 6  -> most target times are off-grid (t = k/3, k/5, k/6, k/7)
If PSNR at off-grid skips drops clearly below the trend of the grid skips, the
piecewise-linear query is the bottleneck that CTQ addresses.

    python tools/eval_ofaf_offgrid.py --ofaf_root /path/to/OF-AF \
        --checkpoint /path/to/OF-AF/train/bsergb.pth --dataset bsergb \
        --data_root /DATASSD1/BSERGB --skips 1,2,3,4,5,6
"""
import argparse
import csv
import os
import sys

import numpy as np

BSERGB_TEST = ['acquarium_08', 'ball_05', 'ball_06', 'basket_08', 'basket_09', 'candies_03', 'eggs_04',
               'elastic_bands_01', 'fire_02', 'football_04', 'horse_11', 'horse_12', 'horse_13', 'horse_18',
               'horse_20', 'jacket_03', 'juggling_06', 'may28_axe_01', 'may29_redbull_01',
               'may29_water_tank_pouring_02', 'orange_juice_02', 'paprika_1000_gain_control_02', 'pen_03',
               'rope_jumping_01', 'street_crossing_08', 'watermelon_01']
BSERGB_SKIP_INDS = {'basket_09': [31, 32, 33, 34], 'may29_rooftop_handheld_02': [17, 70],
                    'may29_rooftop_handheld_03': [306], 'may29_rooftop_handheld_05': [21]}
GOPRO_TEST = ['GOPR0384_11_00', 'GOPR0384_11_05', 'GOPR0385_11_01', 'GOPR0396_11_00', 'GOPR0410_11_00',
              'GOPR0854_11_00', 'GOPR0862_11_00', 'GOPR0868_11_00', 'GOPR0869_11_00', 'GOPR0871_11_00',
              'GOPR0881_11_01']


def grid_frac(t, n):
    x = t * n
    f = x - np.floor(x)
    return 0.0 if (f < 1e-6 or 1 - f < 1e-6) else float(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ofaf_root', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--dataset', default='bsergb', choices=['bsergb', 'gopro'])
    ap.add_argument('--data_root', default='/DATASSD1/BSERGB')
    ap.add_argument('--skips', default='1,2,3,4,5,6')
    ap.add_argument('--nb_of_flow', default=16, type=int)
    ap.add_argument('--voxel_bins', default=128, type=int)
    ap.add_argument('--max_windows', default=-1, type=int)
    ap.add_argument('--out', default='eval/ofaf_offgrid')
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.ofaf_root))
    import cv2
    import torch
    import torch.nn.functional as F
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    from models.model import MyModel                     # OF-AF
    from util.event import EventSequence                 # OF-AF
    from util.voxelization import to_voxel_grid          # OF-AF
    try:
        from lpips import LPIPS
        lp = LPIPS(net='alex', verbose=False).cuda()
    except Exception:
        lp = None

    model = MyModel(args)
    sd = torch.load(args.checkpoint, map_location='cpu')['model']
    model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})
    model.cuda().eval()

    def rgb(path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def pad4(x):
        h, w = x.shape[-2:]
        return F.pad(x, [0, (4 - w % 4) % 4, 0, (4 - h % 4) % 4], mode='constant', value=0)

    os.makedirs(args.out, exist_ok=True)
    skips = [int(s) for s in args.skips.split(',') if s]
    for skip in skips:
        rows = []
        scenes = BSERGB_TEST if args.dataset == 'bsergb' else GOPRO_TEST
        for scene in scenes:
            if args.dataset == 'bsergb':
                sdir = os.path.join(args.data_root, '1_TEST', scene)
                img_dir, ev_dir = os.path.join(sdir, 'images'), os.path.join(sdir, 'events')
                names = sorted(f for f in os.listdir(img_dir) if f.endswith('.png'))
                ev_names = sorted(os.listdir(ev_dir))
                bad = set(BSERGB_SKIP_INDS.get(scene, []))
            else:
                sdir = os.path.join(args.data_root, 'test', scene)
                img_dir, ev_dir = os.path.join(sdir, 'imgs'), os.path.join(sdir, 'events_clean_8x_v2')
                names = sorted(os.listdir(img_dir))
                ev_names = sorted(os.listdir(ev_dir))
                with open(os.path.join(sdir, 'timestamps_clean_8x_v2.txt')) as f:
                    stamps = f.read().splitlines()
                bad = set()
            starts = [s for s in range(0, len(names) - skip - 1, skip)
                      if not any(i in bad for i in range(s, s + skip + 2))]
            if args.max_windows > 0:
                starts = starts[:args.max_windows]
            for s in starts:
                ind = list(range(s, s + skip + 2))
                frames = [rgb(os.path.join(img_dir, names[i])) for i in ind]
                h, w = frames[0].shape[:2]
                if args.dataset == 'bsergb':
                    paths = [os.path.join(ev_dir, f) for f in ev_names[ind[0]:ind[-1]]]
                    evs = EventSequence.from_npz_files(paths, h, w, bsergb=True)
                else:
                    a, b = stamps.index(str(ind[0] / 240)), stamps.index(str(ind[-1] / 240))
                    evs = EventSequence.from_npz_files([os.path.join(ev_dir, f) for f in ev_names[a:b]], h, w)
                vox = to_voxel_grid(evs, args.voxel_bins).unsqueeze(0) if evs is not None else \
                    torch.zeros(1, args.voxel_bins, h, w)
                imgs = torch.cat([torch.from_numpy(f) for f in frames], 2).permute(2, 0, 1).float().unsqueeze(0) / 255
                times = ((torch.tensor(ind[1:-1]) - ind[0]) / (skip + 1)).unsqueeze(0)
                with torch.no_grad():
                    imgs_p, vox_p = pad4(imgs).cuda(), pad4(vox).cuda()
                    res = model(imgs_p[:, :3], imgs_p[:, -3:], vox_p, times.cuda())
                preds = res.clamp(0, 1)[..., :h, :w]
                for i in range(skip):
                    gt = frames[i + 1]
                    pred = (preds[0, i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    t = float(times[0, i])
                    lpv = float('nan')
                    if lp is not None:
                        g = torch.from_numpy(gt).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255
                        lpv = float(lp(preds[:, i], g).item())   # OF-AF style, inputs in [0, 1]
                    rows.append(dict(scene=scene, name=names[ind[i + 1]], t=round(t, 6),
                                     grid_frac=round(grid_frac(t, args.nb_of_flow), 6),
                                     psnr=float(peak_signal_noise_ratio(gt, pred, data_range=255)),
                                     ssim=float(structural_similarity(gt, pred, data_range=255, channel_axis=2)),
                                     lpips=lpv))
            print(f'skip {skip} {scene}: {len(rows)} frames so far')

        path = os.path.join(args.out, f'{args.dataset}_skip{skip}.csv')
        with open(path, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        on = [r['psnr'] for r in rows if r['grid_frac'] == 0]
        off = [r['psnr'] for r in rows if r['grid_frac'] != 0]
        line = (f'[OF-AF {args.dataset} skip{skip}] all={np.mean([r["psnr"] for r in rows]):.2f} dB '
                f'on-grid({len(on)})={np.mean(on) if on else float("nan"):.2f} '
                f'off-grid({len(off)})={np.mean(off) if off else float("nan"):.2f}')
        print(line)
        with open(os.path.join(args.out, 'summary.txt'), 'a') as f:
            f.write(line + '\n')


if __name__ == '__main__':
    main()

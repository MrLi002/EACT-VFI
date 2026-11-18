"""Check that event timestamps and frame timestamps live on the same clock.

For every event file it prints the event time range next to the frame interval it is
supposed to cover, and the fraction of events inside that interval.  Use it before
training with --time_axis frame (the default).

    python tools/check_time_alignment.py --dataset bsergb --root /DATASSD1/BSERGB/1_TEST --scenes ball_05
    python tools/check_time_alignment.py --dataset gopro  --root /DATASSD1/GOPRO/test --scenes GOPR0384_11_00
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.event import load_events  # noqa: E402
from datasets.base import read_timestamp_file  # noqa: E402
from datasets.bsergb import TS_CANDIDATES  # noqa: E402


def report(name, t, a, b):
    if len(t) == 0:
        print(f'  {name}: empty')
        return np.nan
    inside = float(np.mean((t >= a) & (t <= b)))
    print(f'  {name}: events [{t.min():.6f}, {t.max():.6f}]  frame interval [{a:.6f}, {b:.6f}]  inside {inside:.0%}')
    return inside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['gopro', 'bsergb'])
    ap.add_argument('--root', required=True, help='split folder, e.g. .../BSERGB/1_TEST or .../GOPRO/test')
    ap.add_argument('--scenes', default='', help='comma list; empty = first 3 scenes')
    ap.add_argument('--files', default=5, type=int, help='event files to inspect per scene')
    ap.add_argument('--gopro_fps', default=240.0, type=float)
    ap.add_argument('--gopro_event_dir', default='events_clean_8x_v2')
    ap.add_argument('--gopro_ts_file', default='timestamps_clean_8x_v2.txt')
    ap.add_argument('--bsergb_ts_file', default='auto')
    ap.add_argument('--event_time_scale', default=1.0, type=float)
    ap.add_argument('--frame_time_scale', default=1.0, type=float)
    args = ap.parse_args()

    scenes = [s for s in args.scenes.split(',') if s] or sorted(os.listdir(args.root))[:3]
    fracs = []
    for scene in scenes:
        d = os.path.join(args.root, scene)
        print(f'== {scene}')
        if args.dataset == 'gopro':
            ts = read_timestamp_file(os.path.join(d, args.gopro_ts_file)) * args.frame_time_scale
            names = sorted(f for f in os.listdir(os.path.join(d, args.gopro_event_dir)) if f.endswith('.npz'))
            print(f'  {len(names)} event files, {len(ts)} timestamp entries, entry spacing '
                  f'{np.median(np.diff(ts)):.6f} (1/fps = {1 / args.gopro_fps:.6f})')
            for i, f in enumerate(names[:args.files]):
                ev = load_events(os.path.join(d, args.gopro_event_dir, f), 'default')
                ev[:, 2] *= args.event_time_scale
                fracs.append(report(f, ev[:, 2], ts[i], ts[i + 1]))
        else:
            imgs = sorted(f for f in os.listdir(os.path.join(d, 'images')) if f.endswith('.png'))
            cands = TS_CANDIDATES if args.bsergb_ts_file == 'auto' else (args.bsergb_ts_file,)
            ts, used = None, None
            for c in cands:
                ts = read_timestamp_file(os.path.join(d, c))
                if ts is not None:
                    used = c
                    break
            names = sorted(f for f in os.listdir(os.path.join(d, 'events')) if f.endswith('.npz'))
            print(f'  {len(imgs)} images, {len(names)} event files, timestamps: {used} '
                  f'({0 if ts is None else len(ts)} entries)')
            if ts is None:
                print('  no frame timestamps -> only --time_axis legacy is possible')
                continue
            ts = ts * args.frame_time_scale
            for i, f in enumerate(names[:args.files]):
                ev = load_events(os.path.join(d, 'events', f), 'bsergb')
                ev[:, 2] *= args.event_time_scale
                fracs.append(report(f'{f} -> frames {i},{i + 1}', ev[:, 2], ts[i], ts[i + 1]))
    fr = [f for f in fracs if f == f]
    if fr:
        print(f'\nmean inside fraction: {np.mean(fr):.1%}  (expect ~100%; ~0% means a unit/offset '
              f'mismatch, ~50% often means file i covers [t(i-1), t(i)] instead of [t(i), t(i+1)])')


if __name__ == '__main__':
    main()

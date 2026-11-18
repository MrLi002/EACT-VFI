"""Evaluate an EACT-Net checkpoint on GoPro / BS-ERGB at grid and off-grid times.

Examples
    # OF-AF protocol (grid times only)
    python test_model.py --checkpoint train/eact/epoch19.pth --datasets gopro --skips 7,15
    # off-grid protocol: t = k/5, k/7, k/11 are mostly not on the 1/16 grid
    python test_model.py --checkpoint train/eact/epoch19.pth --datasets gopro --skips 4,6,10 --eval_frames all
    # temporal consistency of 63 densely interpolated frames per window
    python test_model.py --checkpoint ... --datasets bsergb --skips 3 --dense 63 --max_windows 50
    # A2 direct effect: GoPro skip 31, odd j (t = j/32 lies mid-way between 1/16 anchors)
    python test_model.py --checkpoint ... --datasets gopro --skips 31 --eval_frames odd
    # A3: forward-splat projection error in many-to-one regions
    python test_model.py --checkpoint ... --datasets gopro --skips 7 --splat_diag true
"""
import argparse

import torch

from evaluator import run_eval, make_lpips
from models import build_model
from options import (add_model_args, add_partition_args, add_data_args, add_eval_args, copy_keys,
                     MODEL_KEYS, PARTITION_KEYS, DATA_KEYS)
from util.misc import strip_module, parse_int_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--datasets', default='gopro', type=str, help='comma list: gopro,bsergb')
    parser.add_argument('--skips', default='7,15', type=str)
    parser.add_argument('--out_dir', default='eval/eact', type=str)
    parser.add_argument('--use_ckpt_args', default=True, type=lambda v: str(v).lower() in ('1', 'true', 'yes'),
                        help='rebuild model / partition options from the checkpoint (recommended)')
    add_model_args(parser)
    add_partition_args(parser)
    add_data_args(parser)
    add_eval_args(parser)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location='cpu')
    if args.use_ckpt_args and state.get('args'):
        copy_keys(args, state['args'], MODEL_KEYS + PARTITION_KEYS + DATA_KEYS)
    print('model:', {k: getattr(args, k, None) for k in MODEL_KEYS})
    print('data:', {k: getattr(args, k) for k in DATA_KEYS})
    print('partition:', {k: getattr(args, k) for k in PARTITION_KEYS})

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(args)
    model.load_state_dict(strip_module(state['model']))
    model.to(device).eval()
    lpips_metric = make_lpips(args, device)

    for ds in [d for d in args.datasets.split(',') if d]:
        run_eval(model, args, ds, parse_int_list(args.skips), args.out_dir, device, lpips_metric)


if __name__ == '__main__':
    main()

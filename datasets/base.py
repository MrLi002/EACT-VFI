"""Shared sample construction for all datasets.

A sample is a dict of tensors:
    im0, im1 : (3, H, W) in [0, 1]
    gts      : (T, 3, H, W) target frames
    times    : (T,) target times on the window's [0, 1] axis
    voxel    : (n * bins_per_seg, H, W) segment voxel
    bounds   : (n + 1,) segment boundaries (uniform or EATP)
  with --query_events exact (default) also the exact per-query event quantities
    q_before (T, bins_per_seg, H, W), q_s0t (T, 1, H, W), q_a0t (T, 1, H, W),
    s_tot (1, H, W), a_tot (1, H, W)          (see util/voxelization.query_event_features)
  evaluation samples additionally carry 'raw' = (x, y, tau, p) numpy arrays, so that
  extra query times (dense evaluation) can be cut from the raw stream.
"""
import random

import cv2
import numpy as np
import torch

from util.event import X_COL, Y_COL, T_COL, P_COL, crop_events
from util.voxelization import (to_unit_time, inside_fraction, segment_voxel_from_tau,
                               query_event_features, warn_once)
from util.eatp import compute_boundaries

QFEAT_KEYS = ('q_before', 'q_s0t', 'q_a0t', 's_tot', 'a_tot')


def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def partition_cfg(args):
    return dict(partition=args.partition, lam=args.eatp_lambda, dmax=args.eatp_dmax,
                mode=args.eatp_mode, tile=args.eatp_tile, top=args.eatp_top)


def random_crop_box(h, w, crop):
    ch, cw = crop
    hs = random.randint(0, h - ch)
    ws = random.randint(0, w - cw)
    return [hs, ws, hs + ch, ws + cw]


def _to_tensor(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0


def check_frame_axis(events_full, t0, t1, tag, min_ok=0.9, min_fatal=0.5):
    """Events of the window must lie between the key-frame times on the frame axis."""
    if len(events_full) < 100:
        return
    frac = inside_fraction(events_full[:, T_COL], t0, t1)
    if frac < min_fatal:
        raise RuntimeError(
            f'[{tag}] only {frac:.0%} of the window events lie in the key-frame interval [{t0}, {t1}]. '
            f'Event and frame timestamps disagree (unit, offset or file-to-frame mapping). Run '
            f'tools/check_time_alignment.py, fix --event_time_scale / --frame_time_scale, '
            f'or use --time_axis legacy.')
    if frac < min_ok:
        warn_once(('axis', tag), f'[{tag}] {frac:.0%} of the window events lie in the key-frame interval; '
                                 f'the rest are dropped. Check tools/check_time_alignment.py.')


def resolve_axis(axis, events_full, t0_frame, t1_frame, trange, tag):
    """Key-frame times on the event clock.

    axis='frame'  : true frame times (required for the physical-time experiments A1-A3)
    axis='legacy' : min/max of the *uncropped* window events (OF-AF-like, crop independent)
    """
    if axis == 'frame':
        if t0_frame is not None and t1_frame is not None:
            check_frame_axis(events_full, t0_frame, t1_frame, tag)
            return float(t0_frame), float(t1_frame)
        warn_once(('noframe', tag), f'[{tag}] no frame timestamps found; using the legacy axis '
                                    f'(event min/max). A2/A3 physical-time results need real frame times.')
    if trange[0] is None or trange[1] <= trange[0]:
        return 0.0, 1.0
    return trange


def target_indices(skip, mode):
    """Target frame offsets j (1..skip) used for evaluation."""
    if mode == 'ofaf':                  # OF-AF GoPro protocol: prediction index i = 1, 3, ... -> j = i + 1
        return [i + 1 for i in range(1, skip, 2)] or [1]
    if mode == 'odd':                   # j odd: with skip 31, t = j/32 lies mid-way between 1/16 anchors
        return list(range(1, skip + 1, 2))
    return list(range(1, skip + 1))


def build_sample(key0, key1, targets, times, events, t0, t1, n, bins_per_seg, cfg,
                 flip_h=False, flip_v=False, reverse=False, query_events='exact', keep_raw=False):
    """
    key0, key1 : HxWx3 uint8 key frames
    targets    : list of HxWx3 uint8 target frames
    times      : target times in [0, 1] (same axis as the events)
    events     : (N, 4) array (x, y, t, p) already cropped to the frame size
    t0, t1     : key-frame times on the event clock (from the dataset's time axis)
    """
    H, W = key0.shape[:2]
    tau, keep = to_unit_time(events[:, T_COL], t0, t1)
    x = events[keep, X_COL].copy()
    y = events[keep, Y_COL].copy()
    p = events[keep, P_COL].copy()
    tau = tau[keep]
    times = [float(t) for t in times]

    frames = [key0] + list(targets) + [key1]
    if flip_h:
        x = (W - 1) - x
        frames = [f[:, ::-1] for f in frames]
    if flip_v:
        y = (H - 1) - y
        frames = [f[::-1] for f in frames]
    if reverse:                                   # play the clip backwards
        tau = 1.0 - tau
        p = -p
        frames = [frames[-1]] + frames[1:-1][::-1] + [frames[0]]
        times = [1.0 - t for t in times[::-1]]

    bounds = compute_boundaries((x, y, tau, p), n, cfg, H, W)
    vox = segment_voxel_from_tau(x, y, tau, p, bounds, bins_per_seg, H, W)

    gts = [_to_tensor(f) for f in frames[1:-1]]
    sample = dict(
        im0=_to_tensor(frames[0]),
        im1=_to_tensor(frames[-1]),
        gts=torch.stack(gts, 0) if gts else torch.zeros(0, 3, H, W),
        times=torch.tensor(times, dtype=torch.float32),
        voxel=torch.from_numpy(vox),
        bounds=torch.from_numpy(np.asarray(bounds, dtype=np.float32)),
    )
    if query_events == 'exact':
        q = query_event_features(x, y, tau, p, bounds, bins_per_seg, H, W, times)
        for k in QFEAT_KEYS:
            sample[k] = torch.from_numpy(q[k])
    if keep_raw:
        sample['raw'] = (x, y, tau, p)
    return sample


def qfeat_from_raw(raw, bounds, bins_per_seg, H, W, times):
    """Exact per-query features for arbitrary extra times (dense evaluation)."""
    x, y, tau, p = raw
    q = query_event_features(x, y, tau, p, np.asarray(bounds, dtype=np.float64), bins_per_seg, H, W,
                             [float(t) for t in times])
    return {k: torch.from_numpy(v) for k, v in q.items()}


def pick_targets(skip, k):
    """Indices (1..skip) of the target frames inside a window."""
    if k is None or k <= 0 or k >= skip:
        return list(range(1, skip + 1))
    return sorted(random.sample(range(1, skip + 1), k))


def read_timestamp_file(path):
    """Parse one timestamp per line (last numeric token); returns np.array or None."""
    vals = []
    try:
        with open(path) as f:
            for line in f:
                toks = line.replace(',', ' ').split()
                for tok in reversed(toks):
                    try:
                        vals.append(float(tok))
                        break
                    except ValueError:
                        continue
    except OSError:
        return None
    return np.asarray(vals, dtype=np.float64) if vals else None


__all__ = ['read_rgb', 'partition_cfg', 'random_crop_box', 'check_frame_axis', 'resolve_axis',
           'target_indices', 'build_sample',
           'qfeat_from_raw', 'pick_targets', 'read_timestamp_file', 'crop_events', 'QFEAT_KEYS']

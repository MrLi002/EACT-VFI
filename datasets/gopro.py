"""GoPro (240 fps) with simulated events, same folder layout as OF-AF:

    <root>/{train,test}/<scene>/imgs/*.png
    <root>/{train,test}/<scene>/events_clean_8x_v2/*.npz        (x, y, t, p)
    <root>/{train,test}/<scene>/timestamps_clean_8x_v2.txt       (one time per event file)

Frame i is at time i / fps.  The event files of a window are those between the
timestamp entries of its two key frames (OF-AF: timestamps.index(str(i / 240))).
A missing entry raises an error instead of silently using the nearest one.
The same class works for Adobe240 / SNU-FILM converted to this layout.
"""
import os
import random

import numpy as np
from torch.utils.data import Dataset

from util.event import load_events_multi, crop_events
from .base import (read_rgb, build_sample, partition_cfg, random_crop_box, pick_targets,
                   read_timestamp_file, resolve_axis, target_indices)


class GoProScene:
    def __init__(self, scene_dir, args):
        self.scene_dir = scene_dir
        self.fps = args.gopro_fps
        self.img_folder = os.path.join(scene_dir, 'imgs')
        self.img_names = sorted(f for f in os.listdir(self.img_folder) if f.lower().endswith(('.png', '.jpg')))
        self.event_folder = os.path.join(scene_dir, args.gopro_event_dir)
        self.event_names = sorted(f for f in os.listdir(self.event_folder) if f.endswith('.npz'))
        ts = read_timestamp_file(os.path.join(scene_dir, args.gopro_ts_file))
        if ts is None:
            raise FileNotFoundError(os.path.join(scene_dir, args.gopro_ts_file))
        self.ts = ts * args.frame_time_scale

    def img_path(self, i):
        return os.path.join(self.img_folder, self.img_names[i])

    def _entry(self, i):
        t = i / self.fps
        j = int(np.argmin(np.abs(self.ts - t)))
        if abs(self.ts[j] - t) > 1e-3 / self.fps:
            raise ValueError(f'[gopro] {self.scene_dir}: no timestamp entry for frame {i} '
                             f'(expected {t:.9f}, nearest {self.ts[j]:.9f}). Check --gopro_fps / --frame_time_scale.')
        return j

    def event_window(self, i0, i1):
        """Event files and key-frame times of the window [frame i0, frame i1]."""
        s, e = self._entry(i0), self._entry(i1)
        paths = [os.path.join(self.event_folder, f) for f in self.event_names[s:e]]
        return paths, i0 / self.fps, i1 / self.fps


class _GoProBase(Dataset):
    def _setup(self, scene_dir, args):
        self.scene = GoProScene(scene_dir, args)
        self.name = os.path.basename(scene_dir.rstrip('/'))
        self.n = args.nb_of_flow
        self.bps = args.voxel_bins // args.nb_of_flow
        self.cfg = partition_cfg(args)
        self.polarity = args.polarity
        self.axis = args.time_axis
        self.ev_scale = args.event_time_scale
        self.query_events = args.query_events

    def _events(self, i0, i1, crop):
        paths, t0f, t1f = self.scene.event_window(i0, i1)
        ev_full, trange = load_events_multi(paths, 'default', None, self.polarity, self.ev_scale)
        t0, t1 = resolve_axis(self.axis, ev_full, t0f, t1f, trange, f'gopro/{self.name}')
        return crop_events(ev_full, crop), t0, t1


class GOPRO_Train(_GoProBase):
    def __init__(self, args, scene_dir, skips, crop=256, targets_per_sample=-1):
        self._setup(scene_dir, args)
        self.skips = list(skips)
        self.max_skip = max(self.skips)
        self.crop = [crop, crop]
        self.k = targets_per_sample
        self.aug_reverse = args.aug_reverse

    def __len__(self):
        return max(0, len(self.scene.img_names) - self.max_skip - 1)

    def __getitem__(self, index):
        skip = random.choice(self.skips)
        i0, i1 = index, index + skip + 1
        key0 = read_rgb(self.scene.img_path(i0))
        box = random_crop_box(key0.shape[0], key0.shape[1], self.crop)
        hs, ws, he, we = box
        key0 = key0[hs:he, ws:we]
        key1 = read_rgb(self.scene.img_path(i1))[hs:he, ws:we]
        tids = pick_targets(skip, self.k)
        targets = [read_rgb(self.scene.img_path(i0 + j))[hs:he, ws:we] for j in tids]
        times = [j / (skip + 1) for j in tids]          # frames are equally spaced at fps
        events, t0, t1 = self._events(i0, i1, box)
        return build_sample(key0, key1, targets, times, events, t0, t1, self.n, self.bps, self.cfg,
                            flip_h=random.random() < 0.5, flip_v=random.random() < 0.5,
                            reverse=self.aug_reverse and random.random() < 0.5,
                            query_events=self.query_events)


class GOPRO_Eval(_GoProBase):
    """One (scene, skip) pair at full resolution.

    stride     : 'ofaf' -> windows start every `skip` frames (OF-AF protocol), 'full' -> every skip + 1
    eval_frames: 'ofaf' (j = 2, 4, ...), 'odd' (j = 1, 3, ...), 'all'
    """

    def __init__(self, args, scene_dir, skip, stride='ofaf', eval_frames='ofaf', max_windows=-1):
        self._setup(scene_dir, args)
        self.skip = skip
        step = skip if stride == 'ofaf' else skip + 1
        self.starts = list(range(0, len(self.scene.img_names) - skip - 1, step))
        if max_windows > 0:
            self.starts = self.starts[:max_windows]
        self.tids = target_indices(skip, eval_frames)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        i0 = self.starts[idx]
        i1 = i0 + self.skip + 1
        key0 = read_rgb(self.scene.img_path(i0))
        key1 = read_rgb(self.scene.img_path(i1))
        targets = [read_rgb(self.scene.img_path(i0 + j)) for j in self.tids]
        times = [j / (self.skip + 1) for j in self.tids]
        events, t0, t1 = self._events(i0, i1, None)
        s = build_sample(key0, key1, targets, times, events, t0, t1, self.n, self.bps, self.cfg,
                         query_events=self.query_events, keep_raw=True)
        s['names'] = [self.scene.img_names[i0 + j] for j in self.tids]
        return s

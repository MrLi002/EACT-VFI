"""BS-ERGB (TimeLens++), same layout as OF-AF:

    <root>/{3_TRAINING,1_TEST}/<scene>/images/*.png
    <root>/{3_TRAINING,1_TEST}/<scene>/events/*.npz      (one file per frame interval)
    frame timestamps: --bsergb_ts_file (path relative to the scene folder); 'auto' tries
                      images/timestamp.txt, images_timestamps.txt, images/timestamps.txt, timestamps.txt

--time_axis frame  (default): the window axis is [ts[i0], ts[i1]], target times are
                    (ts[i0 + j] - ts[i0]) / (ts[i1] - ts[i0]); events outside are dropped.
                    Requires frame timestamps on the event clock (check with
                    tools/check_time_alignment.py; scale with --frame_time_scale / --event_time_scale).
--time_axis legacy : OF-AF-like: target times j / (skip + 1), event axis = min/max of the
                    uncropped window events (OF-AF uses the cropped ones, which is crop dependent).
"""
import os
import random

from torch.utils.data import Dataset

from util.event import load_events_multi, crop_events
from util.voxelization import warn_once
from .base import (read_rgb, build_sample, partition_cfg, random_crop_box, pick_targets,
                   read_timestamp_file, resolve_axis, target_indices)

# Windows that OF-AF skips during BS-ERGB evaluation (bsergb_eval.py).
BSERGB_SKIP_INDS = {
    'basket_09': [31, 32, 33, 34],
    'may29_rooftop_handheld_02': [17, 70],
    'may29_rooftop_handheld_03': [306],
    'may29_rooftop_handheld_05': [21],
}
TS_CANDIDATES = ('images/timestamp.txt', 'images_timestamps.txt', 'images/timestamps.txt', 'timestamps.txt')


class BSERGBScene:
    def __init__(self, scene_dir, args):
        self.scene_dir = scene_dir
        self.img_folder = os.path.join(scene_dir, 'images')
        self.img_names = sorted(f for f in os.listdir(self.img_folder) if f.endswith('.png'))
        self.event_folder = os.path.join(scene_dir, 'events')
        self.event_names = sorted(f for f in os.listdir(self.event_folder) if f.endswith('.npz'))
        self.frame_ts = None
        cands = TS_CANDIDATES if args.bsergb_ts_file == 'auto' else (args.bsergb_ts_file,)
        for c in cands:
            ts = read_timestamp_file(os.path.join(scene_dir, c))
            if ts is None:
                continue
            if len(ts) < len(self.img_names):
                warn_once(('bsergb_ts_len', scene_dir), f'[bsergb] {c} has {len(ts)} entries for ' f'{len(self.img_names)} images; ignored.')
                continue
            if len(ts) != len(self.img_names):
                warn_once(('bsergb_ts_extra', scene_dir), f'[bsergb] {c} has {len(ts)} entries for ' f'{len(self.img_names)} images; using the first ones.')
            self.frame_ts = ts[:len(self.img_names)] * args.frame_time_scale
            break

    def img_path(self, i):
        return os.path.join(self.img_folder, self.img_names[i])

    def event_paths(self, i0, i1):
        return [os.path.join(self.event_folder, f) for f in self.event_names[i0:i1]]

    def window(self, i0, i1, tids, skip, axis):
        """(target times, t0 frame, t1 frame) for the chosen axis."""
        if axis == 'frame' and self.frame_ts is not None:
            t0, t1 = float(self.frame_ts[i0]), float(self.frame_ts[i1])
            return [float((self.frame_ts[i0 + j] - t0) / (t1 - t0)) for j in tids], t0, t1
        return [j / (skip + 1) for j in tids], None, None


class _BSERGBBase(Dataset):
    def _setup(self, scene_dir, args):
        self.scene = BSERGBScene(scene_dir, args)
        self.name = os.path.basename(scene_dir.rstrip('/'))
        self.n = args.nb_of_flow
        self.bps = args.voxel_bins // args.nb_of_flow
        self.cfg = partition_cfg(args)
        self.axis = args.time_axis
        self.ev_scale = args.event_time_scale
        self.query_events = args.query_events

    def _events(self, i0, i1, t0f, t1f, crop):
        ev_full, trange = load_events_multi(self.scene.event_paths(i0, i1), 'bsergb', None,
                                            time_scale=self.ev_scale)
        t0, t1 = resolve_axis(self.axis, ev_full, t0f, t1f, trange, f'bsergb/{self.name}')
        return crop_events(ev_full, crop), t0, t1


class BSERGB_Train(_BSERGBBase):
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
        times, t0f, t1f = self.scene.window(i0, i1, tids, skip, self.axis)
        events, t0, t1 = self._events(i0, i1, t0f, t1f, box)
        return build_sample(key0, key1, targets, times, events, t0, t1, self.n, self.bps, self.cfg,
                            flip_h=random.random() < 0.5, flip_v=random.random() < 0.5,
                            reverse=self.aug_reverse and random.random() < 0.5,
                            query_events=self.query_events)


class BSERGB_Eval(_BSERGBBase):
    """One (scene, skip) pair at full resolution; windows start every `skip` frames (OF-AF)."""

    def __init__(self, args, scene_dir, skip, stride='ofaf', eval_frames='all', max_windows=-1):
        self._setup(scene_dir, args)
        self.skip = skip
        bad = set(BSERGB_SKIP_INDS.get(self.name, []))
        step = skip if stride == 'ofaf' else skip + 1
        starts = range(0, len(self.scene.img_names) - skip - 1, step)
        self.starts = [s for s in starts if not any(i in bad for i in range(s, s + skip + 2))]
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
        times, t0f, t1f = self.scene.window(i0, i1, self.tids, self.skip, self.axis)
        events, t0, t1 = self._events(i0, i1, t0f, t1f, None)
        s = build_sample(key0, key1, targets, times, events, t0, t1, self.n, self.bps, self.cfg,
                         query_events=self.query_events, keep_raw=True)
        s['names'] = [self.scene.img_names[i0 + j] for j in self.tids]
        return s

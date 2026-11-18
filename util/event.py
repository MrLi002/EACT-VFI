"""Event loading utilities.

Events are returned as a float64 array of shape (N, 4) with columns
(x, y, t, p).  Polarity is always mapped to {-1, +1}.

Supported on-disk formats (same conventions as OF-AF):
    default : npz with keys x, y, t, p          (GoPro / Adobe240 / SNU-FILM via v2e)
    bsergb  : npz with keys x, y, timestamp, polarity, coordinates rescaled as in OF-AF
    hsergb  : npz with keys x, y, t, p, p stored as {0, 1}
"""
import warnings

import numpy as np

X_COL, Y_COL, T_COL, P_COL = 0, 1, 2, 3

_POLARITY_WARNED = set()


def _to_pm1(p, mode, tag):
    """Map polarity to {-1, +1}.

    mode = 'pm1'  : already {-1, +1}
           '01'   : stored as {0, 1}
           'auto' : {0, 1} storage is detected when no negative value exists
    OF-AF's default loader keeps p as stored; if your files store {0, 1},
    negative events silently become 0.  EBC-Splat needs signed polarity.
    """
    p = p.astype(np.float32)
    if mode == 'pm1':
        return p
    if mode == '01':
        return p * 2.0 - 1.0
    if p.size == 0:
        return p
    if p.min() >= 0.0:
        if tag not in _POLARITY_WARNED:
            warnings.warn(f"[event] polarity of '{tag}' looks like {{0,1}}; mapping to {{-1,+1}}. "
                          f"Pass --polarity pm1 if this is wrong.")
            _POLARITY_WARNED.add(tag)
        return p * 2.0 - 1.0
    return p


def load_events(path, fmt='default', crop=None, polarity='auto'):
    """Load one npz file.

    crop: [hs, ws, he, we] (OF-AF convention); events outside are dropped and
          coordinates are shifted to the crop origin.
    """
    d = np.load(path, allow_pickle=True)
    if fmt == 'bsergb':
        x = d['x'].astype(np.float64).reshape(-1)
        y = d['y'].astype(np.float64).reshape(-1)
        t = d['timestamp'].astype(np.float64).reshape(-1)
        p = d['polarity'].astype(np.float32).reshape(-1) * 2.0 - 1.0
        # Same rescaling as OF-AF (util/event.py, bsergb branch).
        x = np.floor(2.0 * 970.0 * x / 62000.0)
        y = np.floor(630.0 * y / 20160.0)
    elif fmt == 'hsergb':
        x = d['x'].astype(np.float64).reshape(-1)
        y = d['y'].astype(np.float64).reshape(-1)
        t = d['t'].astype(np.float64).reshape(-1)
        p = d['p'].astype(np.float32).reshape(-1) * 2.0 - 1.0
    else:
        x = d['x'].astype(np.float64).reshape(-1)
        y = d['y'].astype(np.float64).reshape(-1)
        t = d['t'].astype(np.float64).reshape(-1)
        p = _to_pm1(d['p'].reshape(-1), polarity, tag=fmt)
    ev = np.stack([x, y, t, p.astype(np.float64)], axis=1)
    if crop is not None and len(ev):
        hs, ws, he, we = crop
        keep = (ev[:, X_COL] >= ws) & (ev[:, X_COL] < we) & (ev[:, Y_COL] >= hs) & (ev[:, Y_COL] < he)
        ev = ev[keep]
        ev[:, X_COL] -= ws
        ev[:, Y_COL] -= hs
    return ev


def crop_events(ev, crop):
    """Keep events inside crop = [hs, ws, he, we] and shift them to the crop origin."""
    if crop is None or len(ev) == 0:
        return ev
    hs, ws, he, we = crop
    keep = (ev[:, X_COL] >= ws) & (ev[:, X_COL] < we) & (ev[:, Y_COL] >= hs) & (ev[:, Y_COL] < he)
    ev = ev[keep].copy()
    ev[:, X_COL] -= ws
    ev[:, Y_COL] -= hs
    return ev


def load_events_multi(paths, fmt='default', crop=None, polarity='auto', time_scale=1.0):
    """Concatenate several npz files (sorted by time).

    Returns (events, (tmin, tmax)) where the time range is measured on the full,
    *uncropped* stream, so anything derived from it does not depend on the crop.
    time_scale multiplies the event timestamps (e.g. 1e-6 for microseconds -> seconds).
    """
    chunks = [load_events(p, fmt, None, polarity) for p in paths]
    chunks = [c for c in chunks if len(c)]
    if not chunks:
        return np.zeros((0, 4), dtype=np.float64), (None, None)
    ev = np.concatenate(chunks, axis=0)
    if time_scale != 1.0:
        ev[:, T_COL] *= time_scale
    ev = ev[np.argsort(ev[:, T_COL], kind='stable')]
    trange = (float(ev[0, T_COL]), float(ev[-1, T_COL]))
    return crop_events(ev, crop), trange


def flip_events(events, H, W, horizontal=False, vertical=False):
    ev = events.copy()
    if horizontal:
        ev[:, X_COL] = (W - 1) - ev[:, X_COL]
    if vertical:
        ev[:, Y_COL] = (H - 1) - ev[:, Y_COL]
    return ev


def reverse_events_in_time(tau, polarity):
    """Time reversal of normalized timestamps: tau -> 1 - tau, p -> -p."""
    return 1.0 - tau, -polarity

"""Event voxelization and exact per-query event quantities.

Time axis
    Every window is mapped to [0, 1] with the *key-frame* times (t0, t1) chosen by
    the dataset (true frame timestamps, or the uncropped event min/max for the
    OF-AF-compatible 'legacy' axis).  Events outside [t0, t1] are dropped, not
    clipped, so they never pile up on the end bins or distort the EATP curve.

Representations
    segment_voxel_from_tau : n segments with arbitrary boundaries, bins_per_seg bins each,
                             linear in time between neighbouring bins (polarity sum preserved)
    query_event_features   : for every query time t, cut the raw event stream exactly at t:
                             signed sum S(0->t), event count A(0->t) and the events of t's
                             segment that happened before t (on that segment's bin grid).
                             Used by CTQ (A2) and EBC (A3) instead of re-weighting bins.
"""
import warnings

import numpy as np

from .event import X_COL, Y_COL, T_COL, P_COL

_WARNED = set()


def warn_once(key, msg):
    if key not in _WARNED:
        warnings.warn(msg)
        _WARNED.add(key)


def to_unit_time(t, t0, t1, tol=1e-6):
    """tau = (t - t0) / (t1 - t0) and a mask of events inside the window (with a tiny tolerance)."""
    span = float(t1) - float(t0)
    if span <= 0:
        raise ValueError(f'invalid time window [{t0}, {t1}]')
    tau = (np.asarray(t, dtype=np.float64) - t0) / span
    keep = (tau >= -tol) & (tau <= 1.0 + tol)
    return np.clip(tau, 0.0, 1.0), keep


def inside_fraction(t, t0, t1):
    if len(t) == 0:
        return 1.0
    return float(np.mean((t >= t0) & (t <= t1)))


def _splat(x, y, pos, weight, offset, nbins_local, nbins_total, H, W):
    """Linear-in-time splatting into a (nbins_total, H, W) grid; pos in [0, nbins_local - 1]."""
    xi = np.round(x).astype(np.int64)
    yi = np.round(y).astype(np.int64)
    valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    left = np.floor(pos).astype(np.int64)
    frac = pos - left
    size = nbins_total * H * W
    vox = np.zeros(size, dtype=np.float64)
    for b, wt in ((left, 1.0 - frac), (left + 1, frac)):
        m = valid & (b >= 0) & (b < nbins_local) & (wt > 0)
        if not np.any(m):
            continue
        idx = ((offset[m] + b[m]) * H + yi[m]) * W + xi[m]
        vox += np.bincount(idx, weights=weight[m] * wt[m], minlength=size)
    return vox.reshape(nbins_total, H, W).astype(np.float32)


def _pixel_sum(x, y, w, H, W):
    xi = np.round(x).astype(np.int64)
    yi = np.round(y).astype(np.int64)
    m = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    return np.bincount(yi[m] * W + xi[m], weights=w[m], minlength=H * W).reshape(H, W).astype(np.float32)


def global_voxel(events, bins, H, W, t0=None, t1=None):
    """OF-AF style voxel grid over [t0, t1] (event min/max if not given)."""
    if events is None or len(events) == 0:
        return np.zeros((bins, H, W), dtype=np.float32)
    t = events[:, T_COL]
    if t0 is None or t1 is None:
        t0, t1 = float(t.min()), float(t.max())
        if t1 <= t0:
            t1 = t0 + 1.0
    tau, keep = to_unit_time(t, t0, t1)
    ev = events[keep]
    tau = tau[keep]
    return _splat(ev[:, X_COL], ev[:, Y_COL], tau * (bins - 1), ev[:, P_COL],
                  np.zeros(len(tau), dtype=np.int64), bins, bins, H, W)


def segment_index(tau, boundaries):
    """Same convention as the model (torch.searchsorted(right=True) - 1, clipped)."""
    n = len(boundaries) - 1
    k = np.searchsorted(boundaries, tau, side='right') - 1
    return np.clip(k, 0, n - 1)


def segment_voxel_from_tau(x, y, tau, p, boundaries, bins_per_seg, H, W):
    """Voxelize events with normalized times `tau` into n * bins_per_seg channels."""
    boundaries = np.asarray(boundaries, dtype=np.float64)
    n = len(boundaries) - 1
    total = n * bins_per_seg
    if tau.size == 0:
        return np.zeros((total, H, W), dtype=np.float32)
    k = segment_index(tau, boundaries)
    lo = boundaries[k]
    hi = boundaries[k + 1]
    u = np.clip((tau - lo) / np.maximum(hi - lo, 1e-9), 0.0, 1.0)
    return _splat(x, y, u * (bins_per_seg - 1), p, k * bins_per_seg, bins_per_seg, total, H, W)


def query_event_features(x, y, tau, p, boundaries, bins_per_seg, H, W, times):
    """Exact event quantities at every query time (raw timestamps cut at t).

    Returns a dict of float32 arrays
        q_before : (T, bins_per_seg, H, W)  events of t's segment with tau <= t, on that
                   segment's bin grid (the part after t is seg_voxel[k] - q_before)
        q_s0t    : (T, 1, H, W)  signed polarity sum over tau <= t
        q_a0t    : (T, 1, H, W)  number of events with tau <= t (no cancellation)
        s_tot    : (1, H, W)     signed sum over the window
        a_tot    : (1, H, W)     event count over the window
    """
    boundaries = np.asarray(boundaries, dtype=np.float64)
    T = len(times)
    out = dict(q_before=np.zeros((T, bins_per_seg, H, W), np.float32),
               q_s0t=np.zeros((T, 1, H, W), np.float32),
               q_a0t=np.zeros((T, 1, H, W), np.float32))
    absp = np.abs(p)
    out['s_tot'] = _pixel_sum(x, y, p, H, W)[None]
    out['a_tot'] = _pixel_sum(x, y, absp, H, W)[None]
    if tau.size == 0:
        return out
    order = np.argsort(tau, kind='stable')
    x, y, tau, p, absp = x[order], y[order], tau[order], p[order], absp[order]
    seg = segment_index(tau, boundaries)
    for i, t in enumerate(times):
        t = float(t)
        cut = int(np.searchsorted(tau, t, side='right'))           # events with tau <= t
        out['q_s0t'][i, 0] = _pixel_sum(x[:cut], y[:cut], p[:cut], H, W)
        out['q_a0t'][i, 0] = _pixel_sum(x[:cut], y[:cut], absp[:cut], H, W)
        k = int(segment_index(np.array([t]), boundaries)[0])
        m = (seg[:cut] == k)
        if np.any(m):
            lo, hi = boundaries[k], boundaries[k + 1]
            u = np.clip((tau[:cut][m] - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
            out['q_before'][i] = _splat(x[:cut][m], y[:cut][m], u * (bins_per_seg - 1), p[:cut][m],
                                        np.zeros(int(m.sum()), np.int64), bins_per_seg, bins_per_seg, H, W)
    return out

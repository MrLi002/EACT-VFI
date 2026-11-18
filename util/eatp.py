"""EATP: Event-density Adaptive Temporal Partition.

Segment boundaries follow quantiles of the cumulative event curve:

    N_evt(t) = sum_{t_i <= t} w_i / sum_i w_i                  (w_i = |p_i| by default)
    N(t)     = (1 - gamma) * N_evt(t) + gamma * t               (uniform prior)
    t_k      = lam * k/n + (1 - lam) * N^{-1}(k/n),     k = 0..n

The segment-length constraint  dmin/n <= t_k - t_{k-1} <= dmax/n  holds by
construction:
    lower bound: the linear term gives every segment at least lam/n, so lam = dmin;
    upper bound: N'(t) >= gamma, so every quantile spacing of N is <= 1/(gamma n),
                 hence t_k - t_{k-1} <= (lam + (1 - lam)/gamma)/n = dmax/n
                 with gamma = (1 - lam) / (dmax - lam).
No post-hoc projection is needed, so boundaries stay where the events are.
"""
import numpy as np


def uniform_boundaries(n):
    return np.linspace(0.0, 1.0, n + 1).astype(np.float32)


def dense_tile_weights(x, y, H, W, tile=32, top_frac=0.1):
    """1 for events inside the densest `top_frac` tiles, 0 elsewhere."""
    ntx = int(np.ceil(W / tile))
    nty = int(np.ceil(H / tile))
    tx = np.clip((x // tile).astype(np.int64), 0, ntx - 1)
    ty = np.clip((y // tile).astype(np.int64), 0, nty - 1)
    tid = ty * ntx + tx
    counts = np.bincount(tid, minlength=ntx * nty)
    k = max(1, int(round(top_frac * ntx * nty)))
    thr = np.sort(counts)[::-1][k - 1]
    dense = counts >= max(thr, 1)
    return dense[tid].astype(np.float64)


def eatp_boundaries(tau, n, lam=0.5, dmin=None, dmax=4.0, weights=None, grid=1024, min_events=None):
    """Compute n+1 boundaries in [0, 1] from normalized event times `tau`.

    lam  : weight of the uniform time term (also the minimum segment length, in units of 1/n)
    dmin : optional, raises lam to at least dmin
    dmax : maximum segment length in units of 1/n
    """
    if min_events is None:
        min_events = 4 * n
    if tau is None or len(tau) < min_events:
        return uniform_boundaries(n)
    lam = float(lam if dmin is None else max(lam, dmin))
    lam = min(max(lam, 0.0), 1.0)
    if lam >= 1.0:
        return uniform_boundaries(n)
    gamma = (1.0 - lam) / max(dmax - lam, 1e-6)
    gamma = min(max(gamma, 1e-3), 1.0)

    hist, _ = np.histogram(tau, bins=grid, range=(0.0, 1.0), weights=weights)
    total = hist.sum()
    if total <= 0:
        return uniform_boundaries(n)
    g = np.linspace(0.0, 1.0, grid + 1)
    cdf = np.concatenate([[0.0], np.cumsum(hist)]) / total
    cdf = (1.0 - gamma) * cdf + gamma * g            # strictly increasing
    q = np.linspace(0.0, 1.0, n + 1)
    b_mix = np.interp(q, cdf, g)
    b = lam * q + (1.0 - lam) * b_mix
    b[0], b[-1] = 0.0, 1.0
    return b.astype(np.float32)


def compute_boundaries(events_xy_tau_p, n, cfg, H=None, W=None):
    """Front-end used by the datasets.

    events_xy_tau_p : tuple (x, y, tau, p) with tau already normalized to [0, 1]
    cfg             : dict with keys partition, lam, dmin, dmax, mode, tile, top
    """
    if cfg.get('partition', 'uniform') != 'eatp':
        return uniform_boundaries(n)
    x, y, tau, p = events_xy_tau_p
    if tau.size == 0:
        return uniform_boundaries(n)
    w = np.abs(p).astype(np.float64)
    if cfg.get('mode', 'global') == 'dense' and H is not None and W is not None:
        w = w * dense_tile_weights(x, y, H, W, cfg.get('tile', 32), cfg.get('top', 0.1))
        if w.sum() <= 0:
            w = np.abs(p).astype(np.float64)
    return eatp_boundaries(tau, n, cfg.get('lam', 0.5), cfg.get('dmin', None),
                           cfg.get('dmax', 4.0), weights=w)


def reverse_boundaries(b):
    """Boundaries after time reversal tau -> 1 - tau."""
    b = np.asarray(b, dtype=np.float32)
    return (1.0 - b[::-1]).astype(np.float32)

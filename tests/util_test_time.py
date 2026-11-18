"""numpy-only checks of the time axis and the exact per-query event features.
Run: python tests/util_test_time.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.event import crop_events  # noqa: E402
from util.voxelization import (to_unit_time, segment_voxel_from_tau, query_event_features,  # noqa: E402
                               segment_index)
from util.eatp import eatp_boundaries  # noqa: E402

rng = np.random.default_rng(0)
H, W = 20, 30

# 1) events outside the key-frame window are dropped, not clipped onto 0 / 1
t = np.concatenate([np.linspace(0.0, 1.0, 80), np.linspace(1.05, 1.3, 20)])
tau, keep = to_unit_time(t, 0.0, 1.0)
assert keep.sum() == 80 and np.all(tau[keep] <= 1.0)
print('[ok] out-of-window events dropped:', int((~keep).sum()))

# 2) the axis does not depend on the crop: same events -> same tau in every crop
N = 3000
ev = np.stack([rng.integers(0, W, N), rng.integers(0, H, N), np.sort(rng.uniform(10.2, 10.3, N)),
               rng.choice([-1.0, 1.0], N)], 1).astype(np.float64)
t0, t1 = 10.0, 11.0                                   # key-frame times
for box in ([0, 0, 10, 15], [5, 10, 20, 30]):
    c = crop_events(ev, box)
    tau_c, _ = to_unit_time(c[:, 2], t0, t1)
    assert tau_c.min() >= 0.2 - 1e-9 and tau_c.max() <= 0.3 + 1e-9
print('[ok] crop-independent axis: events at [10.2, 10.3] stay at tau in [0.2, 0.3]')

# 3) exact query features vs brute force, uniform and EATP boundaries
x, y = ev[:, 0], ev[:, 1]
tau, keep = to_unit_time(rng.uniform(0, 1, N) * 1.0, 0.0, 1.0)
p = ev[:, 3]
for bounds in (np.linspace(0, 1, 5), eatp_boundaries(tau, 4, lam=0.25, dmax=4.0)):
    bps = 8
    times = [0.0, 0.13, float(bounds[2]), 0.61, 1.0]
    q = query_event_features(x, y, tau, p, bounds, bps, H, W, times)
    seg = segment_voxel_from_tau(x, y, tau, p, bounds, bps, H, W)
    for i, tq in enumerate(times):
        m = tau <= tq
        S = np.zeros((H, W)); A = np.zeros((H, W))
        np.add.at(S, (y[m].astype(int), x[m].astype(int)), p[m])
        np.add.at(A, (y[m].astype(int), x[m].astype(int)), 1.0)
        assert np.allclose(q['q_s0t'][i, 0], S, atol=1e-4) and np.allclose(q['q_a0t'][i, 0], A, atol=1e-4)
        k = int(segment_index(np.array([tq]), bounds)[0])
        before = q['q_before'][i]
        full = seg[k * bps:(k + 1) * bps]
        mk = segment_index(tau, bounds) == k
        Sb = np.zeros((H, W)); np.add.at(Sb, (y[mk & m].astype(int), x[mk & m].astype(int)), p[mk & m])
        assert np.allclose(before.sum(0), Sb, atol=1e-4)
        after = full - before                       # what the model uses for [t, b_{k+1}]
        Sa = np.zeros((H, W)); np.add.at(Sa, (y[mk & ~m].astype(int), x[mk & ~m].astype(int)), p[mk & ~m])
        assert np.allclose(after.sum(0), Sa, atol=1e-4)
    assert np.allclose(q['s_tot'][0], q['q_s0t'][-1, 0]) and np.allclose(q['a_tot'][0], q['q_a0t'][-1, 0])
print('[ok] exact per-query features (signed sum, count, before/after split) match brute force')

# 4) counts are not cancelled by opposite polarities (unlike |signed bin|)
xx = np.array([3.0, 3.0]); yy = np.array([4.0, 4.0]); tt = np.array([0.5, 0.5]); pp = np.array([1.0, -1.0])
q = query_event_features(xx, yy, tt, pp, np.linspace(0, 1, 5), 8, H, W, [1.0])
assert q['q_a0t'][0, 0, 4, 3] == 2.0 and q['q_s0t'][0, 0, 4, 3] == 0.0
print('[ok] event count keeps cancelled +/- pairs')
print('ALL TIME-AXIS TESTS PASSED')

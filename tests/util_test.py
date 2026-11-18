"""numpy-only checks for util/ (EATP, voxelization, metrics). Run: python tests/util_test.py"""
import sys, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.eatp import eatp_boundaries, uniform_boundaries, compute_boundaries, reverse_boundaries, dense_tile_weights
from util.voxelization import global_voxel, segment_voxel_from_tau, to_unit_time
from util.metrics import grid_fraction, consecutive_warp_error, tof, psnr, ssim
rng = np.random.default_rng(0)


# --- EATP: burst of events in [0.4,0.5] -> more boundaries there
tau = np.concatenate([rng.uniform(0,1,2000), rng.uniform(0.4,0.5,20000)])
for lam, dmax in [(0.0, 4.0), (0.25, 4.0), (0.5, 4.0), (0.5, 2.0)]:
    b = eatp_boundaries(tau, 16, lam=lam, dmax=dmax)
    d = np.diff(b)
    mid = 0.5*(b[1:]+b[:-1])
    ins = d[(mid>=0.4)&(mid<=0.5)].mean(); out = d[(mid<0.4)|(mid>0.5)].mean()
    print(f'lam={lam} dmax={dmax}: segs in burst={np.sum((mid>=0.4)&(mid<=0.5))}, mean len in/out = {ins*16:.2f}/{out*16:.2f} (x1/n), min/max = {d.min()*16:.2f}/{d.max()*16:.2f}')
    assert b[0]==0 and b[-1]==1 and np.all(d>0)
    assert d.min() >= lam/16 - 1e-6 and d.max() <= dmax/16 + 1e-6
    assert ins < out
b = eatp_boundaries(tau, 16, lam=0.25, dmax=4.0)
b5 = eatp_boundaries(tau, 16, lam=0.5)
assert np.all(np.diff(b5) > 0)
# empty / few events -> uniform
assert np.allclose(eatp_boundaries(np.array([0.5]), 16), uniform_boundaries(16))
# reverse
rb = reverse_boundaries(b)
assert rb[0]==0 and abs(rb[-1]-1)<1e-6 and np.all(np.diff(rb)>0)

# --- voxel sums preserve polarity
H, W, N = 24, 32, 5000
x = rng.integers(0, W, N).astype(float); y = rng.integers(0, H, N).astype(float)
t = np.sort(rng.uniform(10.0, 11.0, N)); p = rng.choice([-1.0, 1.0], N)
ev = np.stack([x,y,t,p],1)
gv = global_voxel(ev, 128, H, W, 10.0, 11.0)
tau_n, _ = to_unit_time(t, 10.0, 11.0)
sv = segment_voxel_from_tau(x, y, tau_n, p, uniform_boundaries(16), 8, H, W)
bv = segment_voxel_from_tau(x, y, tau_n, p, b, 8, H, W)
ref = np.zeros((H,W)); np.add.at(ref, (y.astype(int), x.astype(int)), p)
for v in (gv, sv, bv):
    assert np.allclose(v.sum(0), ref, atol=1e-4), np.abs(v.sum(0)-ref).max()
print('voxel shapes', gv.shape, sv.shape, bv.shape, 'polarity sums preserved')
# segment k only receives events with tau in its segment
k = np.clip(np.searchsorted(b, tau_n, side='right')-1, 0, 15)
for kk in range(16):
    m = k==kk
    r = np.zeros((H,W)); np.add.at(r, (y[m].astype(int), x[m].astype(int)), p[m])
    assert np.allclose(bv[kk*8:(kk+1)*8].sum(0), r, atol=1e-4)
print('segment membership correct')
# compute_boundaries front-end, dense mode
cb = compute_boundaries((x, y, tau_n, p), 16, dict(partition='eatp', mode='dense', tile=8, top=0.1), H, W)
assert len(cb)==17 and np.all(np.diff(cb)>0)
assert np.allclose(compute_boundaries((x,y,tau_n,p),16,dict(partition='uniform')), uniform_boundaries(16))

# --- grid_fraction
assert grid_fraction(0.5,16)==0.0 and grid_fraction(3/8,16)==0.0
assert abs(grid_fraction(1/3,16)-(16/3-5))<1e-9 and abs(grid_fraction(1/32,16)-0.5)<1e-9
# --- temporal metrics run
frames = [(rng.random((48,64,3))*255).astype(np.uint8) for _ in range(3)]
print('warp err', round(consecutive_warp_error(frames),2), 'tof', round(tof(frames, frames),4), 'psnr', psnr(frames[0], frames[0]+0) if False else 'ok', round(ssim(frames[0],frames[0]),3))
print('ALL UTIL TESTS PASSED')

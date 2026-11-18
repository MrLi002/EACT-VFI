"""Run this first:  python tests/smoke_test.py      (CPU is enough, about a minute)

Checks
  1. softsplat (pure PyTorch) against a naive loop, plus gradcheck
  2. BackwardWarp identity / integer shift
  3. CTQ base interpolation: anchors reproduced, linear motion exact, 'ofaf' == OF-AF formula
  4. CTQ residual with *non-zero* weights: query == anchor at every boundary, no jump across
     boundaries (the unwindowed residual jumps), bump2 + Hermite keeps C1, only F_{t->0}, F_{t->1}
     change with --query_res bwd
  5. EventIntegral: approx (voxel) and exact (qfeat) statistics, conservation
  6. BiW masks: geometric coverage is differentiable in the flow, OF-AF's is not
  7. Full model forward/backward for every ablation config, uniform and EATP boundaries,
     with and without exact query features
  8. Gradient acceptance: forward flow head ~0 in the faithful OF-AF config and > 0 with
     mask_mode geo; alpha, C, diff_tau > 0 with ebc + geo; alpha == 0 with ebc + ofaf masks
  9. GoPro- and BS-ERGB-style datasets on a synthetic folder (frame time axis, exact
     query features), batching with K targets, evaluator incl. splat diagnostics
"""
import math
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
OK = []


def ok(msg):
    OK.append(msg)
    print('[ok]', msg)


def test_softsplat():
    from models.softsplat import softsplat
    B, C, H, W = 2, 3, 7, 9
    x = torch.randn(B, C, H, W, dtype=torch.float64)
    f = torch.randn(B, 2, H, W, dtype=torch.float64) * 2
    m = -torch.rand(B, 1, H, W, dtype=torch.float64) * 3

    def naive(inp, flow):
        out = torch.zeros_like(inp)
        for b in range(B):
            for yy in range(H):
                for xx in range(W):
                    fx = xx + flow[b, 0, yy, xx].item()
                    fy = yy + flow[b, 1, yy, xx].item()
                    x0, y0 = math.floor(fx), math.floor(fy)
                    for ox, oy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                        xi, yi = x0 + ox, y0 + oy
                        if 0 <= xi < W and 0 <= yi < H:
                            out[b, :, yi, xi] += (1 - abs(fx - xi)) * (1 - abs(fy - yi)) * inp[b, :, yy, xx]
        return out

    assert torch.allclose(softsplat(x, f, None, 'sum'), naive(x, f), atol=1e-10)
    e = m.exp()
    num, den = naive(x * e, f), naive(e, f)
    soft = softsplat(x, f, m, 'soft')
    sel = (den > 1e-7).expand_as(soft)
    assert torch.allclose(soft[sel], (num / den.clamp_min(1e-12))[sel], atol=1e-9)
    zero = torch.zeros(B, 2, H, W, dtype=torch.float64)
    assert torch.allclose(softsplat(x, zero, None, 'avg'), x, atol=1e-12)
    xs = torch.randn(1, 2, 5, 6, dtype=torch.float64, requires_grad=True)
    fs = (torch.rand(1, 2, 5, 6, dtype=torch.float64) * 0.6 + 0.2).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda a, b: softsplat(a, b, None, 'sum'), (xs, fs), eps=1e-6, atol=1e-5)
    ok('softsplat matches naive splatting and passes gradcheck')


def test_bwarp():
    from models.blocks import BackwardWarp
    bw = BackwardWarp()
    img = torch.arange(2 * 1 * 6 * 8, dtype=torch.float32).view(2, 1, 6, 8)
    assert torch.allclose(bw(img, torch.zeros(2, 2, 6, 8)), img, atol=1e-4)
    flow = torch.zeros(2, 2, 6, 8)
    flow[:, 0] = 1.0
    out = bw(img, flow)
    assert torch.allclose(out[..., :-1], img[..., 1:], atol=1e-4)
    ok('BackwardWarp identity and integer shift')


def _bounds(B, n, random_bounds):
    if not random_bounds:
        return torch.linspace(0, 1, n + 1).repeat(B, 1)
    inner = torch.sort(torch.rand(B, n - 1) * 0.9 + 0.05, dim=1).values
    return torch.cat([torch.zeros(B, 1), inner, torch.ones(B, 1)], 1)


def test_ctq_base():
    from models.CTQ import locate, interp_hermite, interp_linear, interp_ofaf, subsegment_weights
    B, n = 3, 8
    for rb in (False, True):
        bounds = _bounds(B, n, rb).double()
        stack = torch.randn(B, n + 1, 2, 4, 5, dtype=torch.float64)
        for k in range(n + 1):
            kk, lam = locate(bounds, bounds[:, k])
            assert torch.allclose(interp_hermite(stack, bounds, kk, lam), stack[:, k], atol=1e-9)
            assert torch.allclose(interp_linear(stack, kk, lam), stack[:, k], atol=1e-9)
        a = torch.randn(B, 1, 2, 4, 5, dtype=torch.float64)
        v = torch.randn(B, 1, 2, 4, 5, dtype=torch.float64)
        lin = a + v * bounds.view(B, n + 1, 1, 1, 1)
        t = torch.rand(B, dtype=torch.float64)
        kk, lam = locate(bounds, t)
        assert torch.allclose(interp_hermite(lin, bounds, kk, lam), (a + v * t.view(B, 1, 1, 1, 1))[:, 0], atol=1e-9)
    bounds = _bounds(B, n, False).double()
    stack = torch.randn(B, n + 1, 2, 3, 3, dtype=torch.float64)
    t = torch.tensor([0.3, 0.51, 0.97], dtype=torch.float64)
    kk, lam = locate(bounds, t)
    idx = (t * n).floor().long()
    pct = (t * n - idx).view(B, 1, 1, 1)
    a, b = stack[torch.arange(B), idx], stack[torch.arange(B), idx + 1]
    assert torch.equal(kk, idx)
    assert torch.allclose(interp_ofaf('f0t', stack, kk, lam), a * (1 - pct) + b * pct)
    assert torch.allclose(interp_ofaf('f1t', stack, kk, lam), b * (1 - pct) + a * pct)
    assert torch.allclose(interp_ofaf('ft0', stack, kk, lam), a * pct + b * (1 - pct))
    for bps in (1, 2, 8):
        w0 = subsegment_weights(torch.zeros(2), bps)
        w1 = subsegment_weights(torch.ones(2), bps)
        assert torch.allclose(w0, torch.zeros_like(w0)) and torch.allclose(w1, torch.ones_like(w1))
        prev = None
        for l in torch.linspace(0, 1, 21):
            w = subsegment_weights(l.view(1), bps)
            assert prev is None or bool((w >= prev - 1e-7).all())
            prev = w
    ok('CTQ base interpolation, OF-AF equivalence and sub-segment weights')


def _ctq_setup(base, residual, window, hermite_on='fwd', c=4, bps=4, n=4, H=16, W=16, seed=0):
    from models.CTQ import ContinuousTimeQuery
    torch.manual_seed(seed)
    q = ContinuousTimeQuery(c, bps, base=base, hermite_on=hermite_on, residual=residual, res_window=window).double()
    if residual != 'none':                          # make the (zero-initialised) output heads non-zero
        for head in (q.residual.flow_out, q.residual.ctx_out):
            if head is not None:
                torch.nn.init.normal_(head.weight, std=0.1)
                torch.nn.init.normal_(head.bias, std=0.1)
    h, w = H // 4, W // 4
    anchors = {k: torch.randn(1, n + 1, 2, h, w, dtype=torch.float64) for k in ('f0t', 'f1t', 'ft0', 'ft1')}
    anchors.update({k: torch.randn(1, n + 1, 4 * c, h, w, dtype=torch.float64) for k in ('c0', 'c1')})
    vox = torch.randn(1, n * bps, H, W, dtype=torch.float64)
    bounds = torch.tensor([[0.0, 0.15, 0.5, 0.62, 1.0]], dtype=torch.float64)
    return q, anchors, vox, bounds


def _ctq_at(q, anchors, vox, bounds, t):
    with torch.no_grad():
        return q(anchors, bounds, torch.tensor([t], dtype=torch.float64), vox)


def test_ctq_residual():
    import models.CTQ as ctq
    keys = ('f0t', 'f1t', 'ft0', 'ft1')
    eps = 1e-9
    q, anchors, vox, bounds = _ctq_setup('linear', 'all', 'bump')
    jump_win, jump_raw = 0.0, 0.0
    for k in range(1, 4):
        b = float(bounds[0, k])
        at = _ctq_at(q, anchors, vox, bounds, b)
        for key in keys:
            assert torch.allclose(at[key], anchors[key][:, k], atol=1e-12), (key, k)
        lo, hi = _ctq_at(q, anchors, vox, bounds, b - eps), _ctq_at(q, anchors, vox, bounds, b + eps)
        jump_win = max(jump_win, max(float((lo[key] - hi[key]).abs().max()) for key in keys))
    # without the window (old code): the residual jumps at the anchors
    orig = ctq.window
    ctq.window = lambda lam, kind: torch.ones_like(lam)
    try:
        for k in range(1, 4):
            b = float(bounds[0, k])
            lo, hi = _ctq_at(q, anchors, vox, bounds, b - eps), _ctq_at(q, anchors, vox, bounds, b + eps)
            jump_raw = max(jump_raw, max(float((lo[key] - hi[key]).abs().max()) for key in keys))
    finally:
        ctq.window = orig
    assert jump_raw > 1e-3 and jump_win < 1e-5 * jump_raw, (jump_win, jump_raw)

    # bump2 + Hermite on every stack + residual on every stack: one-sided slopes agree (C1)
    def slope_gap(window):
        q2, an, vx, bd = _ctq_setup('hermite', 'all', window, hermite_on='all', seed=1)
        gap = 0.0
        h = 1e-6
        for k in range(1, 4):
            b = float(bd[0, k])
            m, c0_, p = (_ctq_at(q2, an, vx, bd, b + d) for d in (-h, 0.0, h))
            for key in keys:
                gap = max(gap, float(((p[key] - c0_[key]) / h - (c0_[key] - m[key]) / h).abs().max()))
        return gap
    g2, g1 = slope_gap('bump2'), slope_gap('bump')
    assert g2 < 1e-2 * max(g1, 1e-12) or g2 < 1e-3, (g2, g1)

    # --query_res bwd: forward flows are untouched (pure base interpolation)
    q3, an, vx, bd = _ctq_setup('linear', 'bwd', 'bump', seed=2)
    out = _ctq_at(q3, an, vx, bd, 0.33)
    base = ctq.ContinuousTimeQuery(4, 4, base='linear', residual='none').double()
    ref = _ctq_at(base, an, vx, bd, 0.33)
    assert torch.equal(out['f0t'], ref['f0t']) and torch.equal(out['f1t'], ref['f1t'])
    assert not torch.allclose(out['ft0'], ref['ft0'])
    # exact q_before path gives the anchor at boundaries too
    qb = torch.randn(1, 4, 16, 16, dtype=torch.float64)
    with torch.no_grad():
        o = q3(an, bd, bd[:, 2], vx, qb)
    assert torch.allclose(o['ft0'], an['ft0'][:, 2], atol=1e-12)
    ok(f'CTQ residual: anchors exact, jump across anchors {jump_win:.1e} (unwindowed {jump_raw:.2f}), '
       f'slope gap bump2 {g2:.1e} vs bump {g1:.2f}, bwd leaves forward flows unchanged')


def test_event_integral():
    from models.ebc import EventIntegral
    from util.voxelization import segment_voxel_from_tau, query_event_features
    n, bps, H, W = 4, 8, 6, 7
    rng = np.random.default_rng(1)
    N = 4000
    x, y = rng.integers(0, W, N).astype(float), rng.integers(0, H, N).astype(float)
    tau, p = rng.random(N), rng.choice([-1.0, 1.0], N)
    bounds = np.array([0.0, 0.2, 0.5, 0.6, 1.0], dtype=np.float32)
    vox = torch.from_numpy(segment_voxel_from_tau(x, y, tau, p, bounds, bps, H, W)).unsqueeze(0)
    integ = EventIntegral(vox, n, bps)
    total = vox.sum(1, keepdim=True)
    for k in range(n + 1):
        kk = torch.tensor([min(k, n - 1)])
        lam = torch.tensor([0.0 if k < n else 1.0])
        st = integ.query(kk, lam)
        ref = np.zeros((H, W))
        m = tau < bounds[k] if k < n else np.ones(N, bool)
        np.add.at(ref, (y[m].astype(int), x[m].astype(int)), p[m])
        assert np.allclose(st['s0t'][0, 0].numpy(), ref, atol=1e-3), k
        assert torch.allclose(st['s0t'] + st['st1'], total, atol=1e-4)
    times = [0.1, 0.37, 0.55]
    qf = {k: torch.from_numpy(v).unsqueeze(0) for k, v in
          query_event_features(x, y, tau, p, bounds, bps, H, W, times).items()}
    ex = EventIntegral(vox, n, bps, qf)
    for i, t in enumerate(times):
        st = ex.query(None, None, i)
        cnt = np.zeros((H, W))
        m = tau <= t
        np.add.at(cnt, (y[m].astype(int), x[m].astype(int)), 1.0)
        assert np.allclose(st['a0t'][0, 0].numpy(), cnt)
        assert torch.allclose(st['s0t'] + st['st1'], st['s_tot']) and torch.allclose(st['a0t'] + st['at1'], st['a_tot'])
    ok('EventIntegral: approx boundary values, exact counts, conservation')


def test_masks():
    from models.BiW import coverage
    from models.softsplat import softsplat
    torch.manual_seed(0)
    # sub-pixel motion with noise: coverage stays >= ~0.3, so the OF-AF epsilon term cannot spike
    flow = (0.3 + 0.1 * torch.randn(1, 2, 8, 8, dtype=torch.float64)).requires_grad_(True)
    geo = 1 - coverage(flow).clamp(max=1.0)
    geo.sum().backward()
    g_geo = float(flow.grad.abs().sum())
    flow.grad = None
    ones = torch.ones(1, 1, 8, 8, dtype=torch.float64)
    ofaf = 1 - softsplat(ones, flow, None, 'avg', norm_eps='add')       # OF-AF: s / (s + 1e-7)
    ofaf.sum().backward()
    g_ofaf = float(flow.grad.abs().sum())
    assert g_geo > 1e-3 and g_ofaf < 1e-4 * g_geo, (g_geo, g_ofaf)
    ok(f'hole masks: geometric coverage grad {g_geo:.2f}, OF-AF avg-splat hole grad {g_ofaf:.1e}')


def make_args(**kw):
    from models import MODEL_DEFAULTS
    a = SimpleNamespace(**dict(MODEL_DEFAULTS, voxel_bins=16, nb_of_flow=4))
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _qfeat(B, T, bps, H, W):
    return dict(q_before=torch.randn(B, T, bps, H, W), q_s0t=torch.randn(B, T, 1, H, W),
                q_a0t=torch.rand(B, T, 1, H, W) * 4, s_tot=torch.randn(B, 1, H, W), a_tot=torch.rand(B, 1, H, W) * 8)


A0 = dict(query='ofaf', query_res='none', splat='avg', mask_mode='ofaf', time_cond=False)


def test_model():
    import warnings
    from models import build_model
    B, H, W = 2, 32, 32
    im0, im1 = torch.rand(B, 3, H, W), torch.rand(B, 3, H, W)
    vox = torch.randn(B, 16, H, W)
    times = torch.tensor([[0.1, 0.37, 0.8], [0.25, 0.5, 0.99]])
    configs = [
        A0,
        dict(A0, mask_mode='geo'),
        dict(query='linear', query_res='none', splat='avg'),
        dict(query='hermite', query_res='none', splat='avg'),
        dict(query='linear', query_res='bwd', splat='avg'),
        dict(query='linear', query_res='bwd', res_window='bump2', query_res_ctx=True, splat='avg'),
        dict(query='linear', query_res='none', splat='niklaus', time_cond=False),
        dict(query='linear', query_res='bwd', splat='ebc', ebc_learn_c=False, ebc_pixel_c=False),
        dict(query='linear', query_res='bwd', splat='ebc'),
        dict(query='hermite', hermite_on='all', query_res='all', splat='ebc', ebc_fuse=False),
        dict(query='hermite_res', splat='ebc'),                 # legacy name
    ]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for cfg in configs:
            for rb in (False, True):
                for use_q in (False, True):
                    model = build_model(make_args(**cfg))
                    qf = _qfeat(B, 3, 4, H, W) if use_q else None
                    out = model(im0, im1, vox, times, _bounds(B, 4, rb), qfeat=qf)
                    assert out.shape == (B, 3, 3, H, W), out.shape
                    assert torch.isfinite(out).all()
                    out.mean().backward()
                    bad = [nme for nme, p in model.named_parameters()
                           if p.grad is not None and not torch.isfinite(p.grad).all()]
                    assert not bad, bad
        model = build_model(make_args(splat='ebc'))
        out, aux = model(im0, im1, vox, times, None, qfeat=_qfeat(B, 3, 4, H, W), return_aux=True)
        assert len(aux) == 3 and aux[0]['cov0'].shape == (B, 1, H, W) and aux[0]['splat1'].shape == (B, 3, H, W)
    ok(f'model forward/backward for {len(configs)} configs x (uniform, EATP) x (approx, exact events)')

    torch.manual_seed(3)
    m_n = build_model(make_args(query='linear', query_res='none', splat='avg')).eval()
    m_r = build_model(make_args(query='linear', query_res='bwd', splat='avg')).eval()
    missing, unexpected = m_r.load_state_dict(m_n.state_dict(), strict=False)
    assert not unexpected and all(k.startswith('query.residual') for k in missing)
    with torch.no_grad():
        bnd = _bounds(B, 4, True)
        assert torch.allclose(m_n(im0, im1, vox, times, bnd), m_r(im0, im1, vox, times, bnd), atol=1e-5)
    ok('query_res bwd == none at initialisation (zero-initialised residual)')


def test_gradients():
    import warnings
    from models import build_model
    from train_model import grad_norms
    torch.manual_seed(0)
    B, H, W = 1, 48, 48
    im0, im1 = torch.rand(B, 3, H, W), torch.rand(B, 3, H, W)
    vox = torch.randn(B, 16, H, W)
    times = torch.tensor([[0.3, 0.7]])
    gt = torch.rand(B, 2, 3, H, W)

    qf = {k: v.double() for k, v in _qfeat(B, 2, 4, H, W).items()}

    def norms(cfg):                                  # float64: separates exact zeros from rounding noise
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = build_model(make_args(**cfg)).double()
        out = model(im0.double(), im1.double(), vox.double(), times.double(), None, qfeat=qf)
        torch.nn.functional.l1_loss(out, gt.double()).backward()
        return grad_norms(model)

    a0 = norms(A0)
    geo = norms(dict(A0, mask_mode='geo'))
    ebc = norms(dict(query='linear', query_res='none', splat='ebc', mask_mode='geo'))
    ebc_ofaf = norms(dict(query='linear', query_res='none', splat='ebc', mask_mode='ofaf'))
    # OF-AF masks: only the 1e-7 normalisation epsilon reaches the forward head
    assert a0['fwd_flow_head'] < 1e-4 * max(a0['bwd_flow_head'], 1e-12), a0
    assert geo['fwd_flow_head'] > 1e-3 * geo['bwd_flow_head'] and geo['fwd_flow_head'] > 1e2 * a0['fwd_flow_head'], (geo, a0)
    assert ebc['alpha'] > 0 and ebc['C'] > 0 and ebc['diff_tau'] > 0, ebc
    assert ebc_ofaf['alpha'] == 0 and ebc_ofaf['C'] == 0, ebc_ofaf
    ok(f"grad norms - forward flow head: OF-AF masks {a0['fwd_flow_head']:.1e}, geo {geo['fwd_flow_head']:.1e}; "
       f"ebc+geo alpha {ebc['alpha']:.1e} C {ebc['C']:.1e}; ebc+ofaf masks alpha {ebc_ofaf['alpha']:.0e}")


def _write_events(path, n, H, W, t0, t1, rng, fmt):
    x = rng.integers(0, W, n)
    y = rng.integers(0, H, n)
    t = np.sort(rng.uniform(t0, t1, n))
    p = rng.integers(0, 2, n)
    if fmt == 'bsergb':
        np.savez(path, x=(x * 62000 / 1940 + 1).astype(np.int64), y=(y * 20160 / 630 + 1).astype(np.int64),
                 timestamp=t, polarity=p.astype(bool))
    else:
        np.savez(path, x=x.astype(np.uint16), y=y.astype(np.uint16), t=t.astype(np.float64), p=(p * 2 - 1).astype(np.float32))


def test_datasets():
    from torch.utils.data import DataLoader
    from datasets import get_dataset, get_eval_sets
    from datasets.base import QFEAT_KEYS
    tmp = tempfile.mkdtemp()
    rng = np.random.default_rng(0)
    H, W, F = 64, 96, 24
    try:
        for split, scene in (('train', 'sceneA'), ('test', 'GOPR0384_11_00')):
            d = os.path.join(tmp, 'GOPRO', split, scene)
            os.makedirs(os.path.join(d, 'imgs'))
            os.makedirs(os.path.join(d, 'events_clean_8x_v2'))
            for i in range(F):
                cv2.imwrite(os.path.join(d, 'imgs', f'{i:06d}.png'), rng.integers(0, 255, (H, W, 3), dtype=np.uint8))
            ts = [i / 240 for i in range(F)]
            for i in range(F - 1):
                _write_events(os.path.join(d, 'events_clean_8x_v2', f'{i:06d}.npz'), 300, H, W, ts[i], ts[i + 1], rng, 'default')
            with open(os.path.join(d, 'timestamps_clean_8x_v2.txt'), 'w') as f:
                f.write('\n'.join(str(v) for v in ts))
        for split, scene in (('3_TRAINING', 'sceneB'), ('1_TEST', 'ball_05')):
            d = os.path.join(tmp, 'BSERGB', split, scene)
            os.makedirs(os.path.join(d, 'images'))
            os.makedirs(os.path.join(d, 'events'))
            for i in range(F):
                cv2.imwrite(os.path.join(d, 'images', f'{i:06d}.png'), rng.integers(0, 255, (H, W, 3), dtype=np.uint8))
            for i in range(F - 1):
                _write_events(os.path.join(d, 'events', f'{i:06d}.npz'), 300, H, W, i * 1e4, (i + 1) * 1e4, rng, 'bsergb')
            with open(os.path.join(d, 'images_timestamps.txt'), 'w') as f:
                f.write('\n'.join(str(i * 1e4) for i in range(F)))

        args = SimpleNamespace(
            dataset='gopro', gopro_root=os.path.join(tmp, 'GOPRO'), bsergb_root=os.path.join(tmp, 'BSERGB'),
            gopro_event_dir='events_clean_8x_v2', gopro_ts_file='timestamps_clean_8x_v2.txt', gopro_fps=240.0,
            bsergb_ts_file='auto', time_axis='frame', event_time_scale=1.0, frame_time_scale=1.0,
            query_events='exact', polarity='auto', num_worker=0, voxel_bins=16, nb_of_flow=4,
            partition='eatp', eatp_lambda=0.25, eatp_dmax=4.0, eatp_mode='global', eatp_tile=16, eatp_top=0.1,
            train_skips='3,4,5', skip_sampling='random', targets_per_sample=2, crop=32, aug_reverse=True,
            eval_stride='ofaf', eval_frames='all', scenes='all', max_windows=2, splat_diag=True, collision_thr=1.5)
        for name in ('gopro', 'bsergb'):
            args.dataset = name
            ds = get_dataset(args)
            batch = next(iter(DataLoader(ds, batch_size=2, shuffle=True)))
            assert batch['im0'].shape == (2, 3, 32, 32) and batch['gts'].shape == (2, 2, 3, 32, 32)
            assert batch['voxel'].shape == (2, 16, 32, 32) and batch['bounds'].shape == (2, 5)
            assert torch.all(batch['bounds'][:, 1:] > batch['bounds'][:, :-1])
            assert all(k in batch for k in QFEAT_KEYS) and batch['q_before'].shape == (2, 2, 4, 32, 32)
            for ev in get_eval_sets(args, name, 4):
                s = ev[0]
                assert s['gts'].shape == (4, 3, H, W) and len(s['names']) == 4
                assert torch.allclose(s['times'], torch.tensor([0.2, 0.4, 0.6, 0.8]), atol=1e-6)
                assert torch.allclose(s['a_tot'].sum(), torch.tensor(5 * 300.0)), float(s['a_tot'].sum())
        ok('GoPro / BS-ERGB synthetic datasets on the frame axis, exact query features, batching')

        from models import build_model
        from evaluator import run_eval
        args.dense = 3
        args.save_pred = False
        model = build_model(make_args())
        summ = run_eval(model, args, 'bsergb', [2, 3], os.path.join(tmp, 'eval'), 'cpu', None, verbose=False)
        assert summ[2]['n_off_grid'] > 0 and summ[3]['n_frames'] > 0 and summ[3]['err_single'] == summ[3]['err_single']
        ok('evaluator end-to-end (grid / off-grid split, dense warping error, splat diagnostics)')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    test_softsplat()
    test_bwarp()
    test_ctq_base()
    test_ctq_residual()
    test_event_integral()
    test_masks()
    test_model()
    test_gradients()
    test_datasets()
    print(f'\nALL {len(OK)} CHECKS PASSED')

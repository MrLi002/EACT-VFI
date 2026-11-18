"""CTQ: Continuous-Time Query of the one-shot trajectory (A2).

BiFEB produces n+1 anchors (flows and contexts) at the segment boundaries
b_0 = 0 < b_1 < ... < b_n = 1.  For a query time t in segment k, with
lambda = (t - b_k) / (b_{k+1} - b_k):

    query(t) = base(t) + w(lambda) * g_theta(events in [b_k, t] and [t, b_{k+1}], ...)

base
    ofaf    : exactly the weighting of OF-AF's released code (ablation A0)
    linear  : (1 - lambda) A_k + lambda A_{k+1}, same direction for every stack
    hermite : non-uniform Catmull-Rom (C1) on the stacks selected by `hermite_on`,
              linear on the others.  Default hermite_on='fwd': forward flows F_{0->t},
              F_{1->t} follow fixed key-frame pixels and are smooth in t; backward flows
              F_{t->0}, F_{t->1} live on the grid of the target time, where a pixel can
              switch objects between neighbouring times, so they stay linear.
residual ('none' | 'bwd' | 'all')
    'bwd' (default) corrects only F_{t->0}, F_{t->1}: these are the flows whose pixel
    changes object when an edge passes, which the events inside the segment reveal.
window w(lambda)
    'bump'  : 4 lambda (1 - lambda)            zero at both anchors -> continuous across anchors
    'bump2' : 16 lambda^2 (1 - lambda)^2       also zero slope -> keeps the C1 continuity of hermite
    Without a window the residual at lambda -> 1 (sees the whole previous segment) and at
    lambda -> 0 of the next segment (sees the whole next segment) can differ, so the query
    would jump at every anchor after training; the window makes anchors exact by construction.

Note: with uniform n = 16 the targets of BS-ERGB skip 1/3 and GoPro skip 7/15 (OF-AF protocol)
sit on anchors, where w = 0.  A2 acts directly only off the grid (e.g. GoPro skip 31, odd j).
"""
import math

import torch
from torch import nn

from .blocks import ResBlock, zero_init

FLOW_KEYS = ('f0t', 'f1t', 'ft0', 'ft1')
FWD_KEYS = ('f0t', 'f1t')
BWD_KEYS = ('ft0', 'ft1')
CTX_KEYS = ('c0', 'c1')
BASES = ('ofaf', 'linear', 'hermite')
RESIDUALS = ('none', 'bwd', 'all')
WINDOWS = ('bump', 'bump2')


def take(stack, idx):
    """stack (B, N, ...), idx (B,) long -> (B, ...)."""
    return stack[torch.arange(stack.shape[0], device=stack.device), idx]


def _bc(v, ref):
    return v.to(ref.dtype).view(-1, *([1] * (ref.dim() - 1)))


def locate(bounds, t):
    """Segment index k and local position lambda of t; bounds (B, n+1), t (B,)."""
    n = bounds.shape[1] - 1
    tt = t.to(bounds.dtype).reshape(-1, 1).contiguous()
    k = torch.searchsorted(bounds.contiguous(), tt, right=True).reshape(-1) - 1
    k = k.clamp(0, n - 1)
    tk = take(bounds, k)
    tk1 = take(bounds, k + 1)
    lam = ((tt.reshape(-1) - tk) / (tk1 - tk).clamp_min(1e-6)).clamp(0.0, 1.0)
    return k, lam


def interp_linear(stack, k, lam):
    a, b = take(stack, k), take(stack, k + 1)
    return a + _bc(lam, a) * (b - a)


def interp_ofaf(key, stack, k, lam):
    """Replicates OF-AF model.py / BiFEB.Query weighting."""
    a, b = take(stack, k), take(stack, k + 1)
    l = _bc(lam, a)
    if key == 'f0t':
        return a * (1 - l) + b * l
    if key == 'f1t':
        return b * (1 - l) + a * l
    return a * l + b * (1 - l)          # ft0, ft1, c0, c1 (Query)


def interp_hermite(stack, bounds, k, lam):
    n = stack.shape[1] - 1
    km1 = (k - 1).clamp(min=0)
    kp1 = k + 1
    kp2 = (k + 2).clamp(max=n)
    P0, P1, P2, P3 = take(stack, km1), take(stack, k), take(stack, kp1), take(stack, kp2)
    t_m1, t_0, t_1, t_2 = take(bounds, km1), take(bounds, k), take(bounds, kp1), take(bounds, kp2)
    h = t_1 - t_0
    m1 = (P2 - P0) / _bc((t_1 - t_m1).clamp_min(1e-6), P0)      # one-sided at the ends
    m2 = (P3 - P1) / _bc((t_2 - t_0).clamp_min(1e-6), P1)
    l = _bc(lam, P1)
    hh = _bc(h, P1)
    l2 = l * l
    l3 = l2 * l
    return ((2 * l3 - 3 * l2 + 1) * P1 + (l3 - 2 * l2 + l) * hh * m1
            + (-2 * l3 + 3 * l2) * P2 + (l3 - l2) * hh * m2)


def window(lam, kind):
    b = 4.0 * lam * (1.0 - lam)
    return b if kind == 'bump' else b * b


def subsegment_weights(lam, bps):
    """Approximate fraction of each bin's events before lambda (B, bps); used only when
    exact query features are not available (--query_events approx).  Triangular-kernel CDF,
    assumes a locally uniform event rate; lambda=0 -> 0, lambda=1 -> 1."""
    if bps == 1:
        return lam.view(-1, 1)
    j = torch.arange(bps, device=lam.device, dtype=lam.dtype).view(1, -1)
    d = lam.view(-1, 1) * (bps - 1) - j
    interior = torch.where(d < 0, 0.5 * (1 + d).clamp(min=0) ** 2, 1 - 0.5 * (1 - d).clamp(min=0) ** 2)
    first = torch.where(d < 0, torch.zeros_like(d), 1 - (1 - d).clamp(min=0) ** 2)
    last = torch.where(d < 0, (1 + d).clamp(min=0) ** 2, torch.ones_like(d))
    return torch.cat([first[:, :1], interior[:, 1:-1], last[:, -1:]], dim=1).clamp(0.0, 1.0)


def time_embedding(lam, n_freq, h, w):
    freqs = (2.0 ** torch.arange(n_freq, device=lam.device, dtype=lam.dtype)) * math.pi
    ang = lam.view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
    return emb.view(-1, 2 * n_freq, 1, 1).expand(-1, -1, h, w)


class QueryResidual(nn.Module):
    """g_theta: residual for the selected flows (and optionally contexts)."""

    def __init__(self, base_channel, bins_per_seg, n_flows, with_ctx, n_freq=4):
        super().__init__()
        c = base_channel
        self.n_freq = n_freq
        self.ev = nn.Sequential(
            nn.Conv2d(2 * bins_per_seg + 1, c, 3, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(c, 2 * c, 3, 2, 1), nn.LeakyReLU(0.2),
        )
        self.ctx_sq = nn.Conv2d(8 * c, 2 * c, 1)
        in_ch = 2 * c + 2 * c + 8 + 2 * n_freq + 1
        self.body = nn.Sequential(ResBlock(in_ch, 4 * c), ResBlock(4 * c, 4 * c), ResBlock(4 * c, 4 * c))
        self.flow_out = zero_init(nn.Conv2d(4 * c, 2 * n_flows, 3, 1, 1))
        self.ctx_out = zero_init(nn.Conv2d(4 * c, 8 * c, 3, 1, 1)) if with_ctx else None

    def forward(self, ev_before, ev_after, lam, dt, flows, c0, c1):
        b, _, h, w = c0.shape
        lam_full = lam.to(ev_before.dtype).view(-1, 1, 1, 1).expand(-1, 1, *ev_before.shape[-2:])
        e = self.ev(torch.cat([ev_before, ev_after, lam_full], 1))
        emb = time_embedding(lam.to(c0.dtype), self.n_freq, h, w)
        dt_map = dt.to(c0.dtype).view(-1, 1, 1, 1).expand(-1, 1, h, w)
        x = torch.cat([e, self.ctx_sq(torch.cat([c0, c1], 1))] + [flows[k] for k in FLOW_KEYS] + [emb, dt_map], 1)
        y = self.body(x)
        return self.flow_out(y), (self.ctx_out(y) if self.ctx_out is not None else None)


class ContinuousTimeQuery(nn.Module):
    def __init__(self, base_channel, bins_per_seg, base='linear', hermite_on='fwd', residual='bwd',
                 res_window='bump', res_ctx=False):
        super().__init__()
        assert base in BASES and residual in RESIDUALS and res_window in WINDOWS and hermite_on in ('fwd', 'all')
        self.base = base
        self.hermite_keys = FWD_KEYS if hermite_on == 'fwd' else FLOW_KEYS
        self.res_keys = () if residual == 'none' else (BWD_KEYS if residual == 'bwd' else FLOW_KEYS)
        self.res_window = res_window
        self.res_ctx = res_ctx and residual != 'none'
        self.bps = bins_per_seg
        self.c4 = 4 * base_channel
        if self.res_keys:
            self.residual = QueryResidual(base_channel, bins_per_seg, len(self.res_keys), self.res_ctx)

    def forward(self, anchors, bounds, t, seg_voxel=None, q_before=None):
        """q_before: (B, bps, H, W) exact events of t's segment before t (None -> approximation)."""
        k, lam = locate(bounds, t)
        out = {}
        for key in FLOW_KEYS + CTX_KEYS:
            if self.base == 'ofaf':
                out[key] = interp_ofaf(key, anchors[key], k, lam)
            elif self.base == 'hermite' and key in self.hermite_keys:
                out[key] = interp_hermite(anchors[key], bounds, k, lam)
            else:
                out[key] = interp_linear(anchors[key], k, lam)

        if self.res_keys:
            assert seg_voxel is not None, 'the CTQ residual needs the segment voxel'
            B, _, H, W = seg_voxel.shape
            n = bounds.shape[1] - 1
            vk = take(seg_voxel.reshape(B, n, self.bps, H, W), k)
            if q_before is None:
                wb = subsegment_weights(lam, self.bps).to(vk.dtype).view(B, self.bps, 1, 1)
                before = vk * wb
            else:
                before = q_before.to(vk.dtype)
            dt = (take(bounds, k + 1) - take(bounds, k)) * n
            d_flow, d_ctx = self.residual(before, vk - before, lam, dt,
                                          {key: out[key] for key in FLOW_KEYS}, out['c0'], out['c1'])
            wl = window(lam, self.res_window).to(d_flow.dtype).view(-1, 1, 1, 1)
            for i, key in enumerate(self.res_keys):
                out[key] = out[key] + wl * d_flow[:, 2 * i:2 * i + 2]
            if d_ctx is not None:
                out['c0'] = out['c0'] + wl * d_ctx[:, :self.c4]
                out['c1'] = out['c1'] + wl * d_ctx[:, self.c4:]

        out['k'] = k
        out['lam'] = lam
        return out

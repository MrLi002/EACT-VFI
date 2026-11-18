"""EBC-Splat helpers: event-integrated target luminance.

Per-pixel event model (valid regardless of motion):
    L_t(q) = L_0(q) + C(q) * S_{0->t}(q) = L_1(q) - C(q) * S_{t->1}(q)
with L = log(Y + eps) and S the signed polarity sum.  The two estimates are
fused with weights softmax(-beta * A), A = event activity of the integration
interval (longer/busier integration accumulates more noise and threshold error).
BiW then scores every source pixel p by
    Z(p) = -alpha * rho(L_src(p) - L_hat_t(p + F_{src->t}(p)))
and uses Z as the softmax-splatting metric.
"""
import math

import torch
from torch import nn

from .blocks import zero_init
from .CTQ import take, subsegment_weights


def rgb_to_logy(img, eps=1e-3):
    y = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    return torch.log(y.clamp_min(0.0) + eps)


class EventIntegral:
    """Event statistics of [0, t] and [t, 1] for every query time.

    exact  (default, --query_events exact): the dataset cuts the raw timestamps at t and
           provides the signed sum q_s0t and the true event count q_a0t (no +/- cancellation).
    approx (fallback): from the segment voxel; inside t's segment the bins are split with
           a triangular-kernel CDF and the activity is |bin|, a lower bound of the count
           (opposite polarities in the same bin cancel).  Kept for old checkpoints / caches.
    """

    def __init__(self, seg_voxel, n, bins_per_seg, qfeat=None):
        self.qfeat = qfeat
        if qfeat is not None:
            return
        B, _, H, W = seg_voxel.shape
        self.bps = bins_per_seg
        self.V = seg_voxel.reshape(B, n, bins_per_seg, H, W)
        seg_s = self.V.sum(2)
        seg_a = self.V.abs().sum(2)
        zero = torch.zeros_like(seg_s[:, :1])
        self.cs = torch.cat([zero, seg_s.cumsum(1)], 1)    # (B, n+1, H, W), cs[k] = sum over segments < k
        self.ca = torch.cat([zero, seg_a.cumsum(1)], 1)

    def query(self, k, lam, i=None):
        """k, lam from CTQ.locate; i = index of the query time (exact mode)."""
        u = lambda x: x.unsqueeze(1)
        if self.qfeat is not None:
            s0t = self.qfeat['q_s0t'][:, i]
            a0t = self.qfeat['q_a0t'][:, i]
            s_tot = self.qfeat['s_tot']
            a_tot = self.qfeat['a_tot']
            return dict(s0t=s0t, st1=s_tot - s0t, a0t=a0t, at1=a_tot - a0t, s_tot=s_tot, a_tot=a_tot)
        vk = take(self.V, k)
        w = subsegment_weights(lam, self.bps).to(vk.dtype)[:, :, None, None]
        s0t = take(self.cs, k) + (vk * w).sum(1)
        a0t = take(self.ca, k) + (vk.abs() * w).sum(1)
        s_tot = self.cs[:, -1]
        a_tot = self.ca[:, -1]
        return dict(s0t=u(s0t), st1=u(s_tot - s0t), a0t=u(a0t), at1=u(a_tot - a0t),
                    s_tot=u(s_tot), a_tot=u(a_tot))


class ContrastThreshold(nn.Module):
    """C(q) = c0 * exp(0.5 * tanh(net(stats)));  c0 learnable scalar, net zero-initialised."""

    def __init__(self, c_init=0.2, learn=True, per_pixel=True):
        super().__init__()
        self.log_c = nn.Parameter(torch.tensor(math.log(c_init)), requires_grad=learn)
        self.per_pixel = per_pixel and learn
        if self.per_pixel:
            self.net = nn.Sequential(
                nn.Conv2d(2, 16, 3, 1, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(16, 16, 3, 1, 1), nn.LeakyReLU(0.2),
                zero_init(nn.Conv2d(16, 1, 3, 1, 1)),
            )

    def forward(self, a_tot, s_tot):
        c = self.log_c.exp()
        if self.per_pixel:
            feat = torch.cat([torch.log1p(a_tot), torch.log1p(s_tot.abs())], 1)
            c = c * torch.exp(0.5 * torch.tanh(self.net(feat)))
        return c


class TargetLuminance(nn.Module):
    def __init__(self, c_init=0.2, learn_c=True, per_pixel_c=True, beta_init=0.05, fuse=True, eps=1e-3):
        super().__init__()
        self.threshold = ContrastThreshold(c_init, learn_c, per_pixel_c)
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta_init)))
        self.fuse = fuse
        self.lo = math.log(eps)
        self.hi = math.log(1.5 + eps)

    def forward(self, L0, L1, stats):
        """Returns (L_hat for the I0 side, L_hat for the I1 side)."""
        C = self.threshold(stats['a_tot'], stats['s_tot'])
        lh0 = (L0 + C * stats['s0t']).clamp(self.lo, self.hi)
        lh1 = (L1 - C * stats['st1']).clamp(self.lo, self.hi)
        if not self.fuse:
            return lh0, lh1
        beta = self.log_beta.exp()
        w = torch.softmax(torch.stack([-beta * stats['a0t'], -beta * stats['at1']], 0), 0)
        lh = w[0] * lh0 + w[1] * lh1
        return lh, lh

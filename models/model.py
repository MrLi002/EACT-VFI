"""EACT-Net: Event-Adaptive Continuous Trajectories with Brightness-Consistent Splatting.

Pipeline (OF-AF backbone, three changes marked [A1]/[A2]/[A3]):
    segment voxel (uniform or [A1] EATP boundaries)
      -> EventsEncoder / ImageEncoder
      -> BiFEB: n recurrent steps per direction, conditioned on segment length [A1]
      -> anchors {F_{0->b_k}, F_{1->b_k}, F_{b_k->0}, F_{b_k->1}, ctx} at k = 0..n
      -> for every query t: CTQ [A2] -> (optionally) event target luminance [A3]
      -> BiW with EBC-Splat [A3] -> I_t
The trajectory is estimated once; only CTQ + BiW run per target frame.
"""
import torch
from torch import nn

from .encoder import EventsEncoder, ImageEncoder
from .BiFEB import BidirFlow
from .CTQ import ContinuousTimeQuery
from .BiW import BiW
from .ebc import EventIntegral, TargetLuminance, rgb_to_logy


class EACTNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        c = 32
        self.base_channel = c
        self.voxel_bins = args.voxel_bins
        self.nb_of_flow = args.nb_of_flow
        assert self.voxel_bins % self.nb_of_flow == 0, 'voxel_bins must be a multiple of nb_of_flow'
        self.bins_per_seg = self.voxel_bins // self.nb_of_flow
        self.splat = args.splat

        self.events_encoder = EventsEncoder(self.voxel_bins, c, bins_per_seg=self.bins_per_seg)
        self.im_encoder = ImageEncoder(3, c)
        self.bidir_flow0 = BidirFlow(c, time_cond=args.time_cond)
        self.bidir_flow1 = BidirFlow(c, time_cond=args.time_cond)
        self.query = ContinuousTimeQuery(c, self.bins_per_seg, base=args.query, hermite_on=args.hermite_on,
                                         residual=args.query_res, res_window=args.res_window,
                                         res_ctx=args.query_res_ctx)
        if self.splat == 'ebc':
            self.target = TargetLuminance(c_init=args.ebc_c_init, learn_c=args.ebc_learn_c,
                                          per_pixel_c=args.ebc_pixel_c, fuse=args.ebc_fuse)
        self.biw = BiW(c, splat=args.splat, mask_mode=args.mask_mode, alpha_init=args.ebc_alpha,
                       diff_tau=args.diff_tau, diff_temp=args.diff_temp)

    def uniform_bounds(self, b, device, dtype):
        return torch.linspace(0.0, 1.0, self.nb_of_flow + 1, device=device, dtype=dtype).unsqueeze(0).repeat(b, 1)

    def trajectory(self, im0, im1, voxel, bounds):
        efs = self.events_encoder(im0, im1, voxel)                 # (B, n, 4c, h, w)
        if0_1, if0_4 = self.im_encoder(im0)
        if1_1, if1_4 = self.im_encoder(im1)
        b, n, _, h, w = efs.shape
        dts = (bounds[:, 1:] - bounds[:, :-1]) * n                 # (B, n), == 1 for uniform

        ift0, ift1 = if0_4, if1_4
        ctxt0, ctxt1 = if0_4, if1_4
        motion0t, motion1t = torch.zeros_like(if0_4), torch.zeros_like(if1_4)
        zero = if0_4.new_zeros((b, 2, h, w))
        flowt0, flowt1 = zero, zero
        f0t, ft0, c0 = [zero], [zero], [if0_4]
        f1t, ft1, c1 = [zero], [zero], [if1_4]
        for s in range(n):
            r = n - 1 - s
            dt0 = dts[:, s].view(b, 1, 1, 1).expand(b, 1, h, w).to(if0_4.dtype)
            dt1 = dts[:, r].view(b, 1, 1, 1).expand(b, 1, h, w).to(if0_4.dtype)
            ift0, motion0t, flow0t, flowt0, ctxt0 = self.bidir_flow0(if0_4, ift0, flowt0, motion0t, ctxt0, efs[:, s], dt0)
            ift1, motion1t, flow1t, flowt1, ctxt1 = self.bidir_flow1(if1_4, ift1, flowt1, motion1t, ctxt1, efs[:, r], dt1)
            f0t.append(flow0t)
            ft0.append(flowt0)
            c0.append(ctxt0)
            f1t.insert(0, flow1t)
            ft1.insert(0, flowt1)
            c1.insert(0, ctxt1)
        # index k of every stack <-> time b_k
        anchors = dict(f0t=torch.stack(f0t, 1), f1t=torch.stack(f1t, 1),
                       ft0=torch.stack(ft0, 1), ft1=torch.stack(ft1, 1),
                       c0=torch.stack(c0, 1), c1=torch.stack(c1, 1))
        return anchors, if0_1, if1_1

    def forward(self, im0, im1, voxel, interp_times, bounds=None, qfeat=None, return_aux=False):
        """
        im0, im1     : (B, 3, H, W) in [0, 1], H and W multiples of 4
        voxel        : (B, n * bins_per_seg, H, W) segment voxel built with `bounds`
        interp_times : (B, T) or (T,) query times in [0, 1]
        bounds       : (B, n + 1) segment boundaries; None -> uniform
        qfeat        : exact per-query event features from the dataset (see datasets/base.py):
                       q_before (B, T, bps, H, W), q_s0t / q_a0t (B, T, 1, H, W), s_tot / a_tot (B, 1, H, W).
                       None -> approximation from the segment voxel.
        returns      : (B, T, 3, H, W); with return_aux also a list (len T) of BiW diagnostics
        """
        b = im0.shape[0]
        if bounds is None:
            bounds = self.uniform_bounds(b, im0.device, im0.dtype)
        bounds = bounds.to(im0.dtype)
        if interp_times.dim() == 1:
            interp_times = interp_times.view(1, -1).expand(b, -1)
        if qfeat is not None:
            assert qfeat['q_before'].shape[1] == interp_times.shape[1], 'qfeat must match interp_times'

        anchors, if0_1, if1_1 = self.trajectory(im0, im1, voxel, bounds)
        if self.splat == 'ebc':
            integral = EventIntegral(voxel, self.nb_of_flow, self.bins_per_seg, qfeat)
            L0, L1 = rgb_to_logy(im0), rgb_to_logy(im1)

        mf0 = mf1 = None
        if self.splat == 'niklaus':                  # metric from key frames and F_{0->1} / F_{1->0}
            L0, L1 = rgb_to_logy(im0), rgb_to_logy(im1)
            mf0, mf1 = anchors['f0t'][:, -1], anchors['f1t'][:, 0]

        outs, auxs = [], []
        for i in range(interp_times.shape[1]):
            t = interp_times[:, i].to(im0.dtype)
            qb = qfeat['q_before'][:, i] if qfeat is not None else None
            q = self.query(anchors, bounds, t, voxel, qb)
            lh0 = lh1 = None
            if self.splat == 'ebc':
                lh0, lh1 = self.target(L0, L1, integral.query(q['k'], q['lam'], i))
            elif self.splat == 'niklaus':
                lh0, lh1 = L1, L0
            r = self.biw(im0, im1, t, q['f0t'], q['f1t'], q['ft0'], q['ft1'], q['c0'], q['c1'], if0_1, if1_1, lh0, lh1, mf0, mf1, return_aux)
            if return_aux:
                outs.append(r[0])
                auxs.append(r[1])
            else:
                outs.append(r)
        out = torch.stack(outs, 1)
        return (out, auxs) if return_aux else out

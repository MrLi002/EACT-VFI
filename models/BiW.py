"""Bidirectional Warping (OF-AF) with optional EBC-Splat.

Two independent switches:

splat       how the forward-splatted image I_{0->t} is formed
    'avg'     : summation splatting normalised by the splatted weights (OF-AF)
    'ebc'     : softmax splatting with the event brightness-consistency metric
                Z(p) = -alpha * rho(L_src(p) - L_hat_t(p + F_{src->t}(p)))
    'niklaus' : softmax splatting with the original metric
                Z(p) = -alpha * rho(L_0(p) - L_1(p + F_{0->1}(p)))      (ablation baseline)

mask_mode   how the two repair masks are formed; *identical for every splat mode*, so an
            avg / ebc / niklaus comparison only changes the splat metric
    'ofaf' : hole = 1 - s / (s + 1e-7), s = sum_splat(1, F)   (OF-AF; ~binary)
             diff = [ |I_{t<-0} - I_{0->t}|_1 > 0.1 ]           (hard threshold, no gradient)
             -> faithful OF-AF.  I_{0->t} gets no gradient; the forward flow head F_{0->t} only
                gets d/ds s/(s+1e-7) = 1e-7/(s+1e-7)^2: ~1e-7 on covered pixels, large only on
                the rare pixels with coverage s ~ 1e-6..1e-4 -> no useful learning signal.
    'geo'  : hole = 1 - min(1, sum_splat(1, F_{0->t}))   geometric coverage, differentiable in the flow
             diff = sigmoid((|I_{t<-0} - I_{0->t}|_1 - tau) / temp), tau learnable
             -> the forward flow and the splat metric (alpha, C) get a direct gradient through
                the diff mask; coverage stays a purely geometric signal and does not depend on the
                metric, so alpha / C cannot shrink or grow the repair region by themselves.

Earlier versions derived the hole from the splatted metric mass (1 - sum exp(Z)); that mixed
coverage with brightness confidence (one source with Z = -2 already gave a hole of 0.865) and is
no longer used.
"""
import math
import warnings

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import ResBlock, ResBlockIF, BackwardWarp, upsample_flow
from .softsplat import softsplat
from .ebc import rgb_to_logy

SPLATS = ('avg', 'ebc', 'niklaus')
MASK_MODES = ('ofaf', 'geo')


class RefBlock(nn.Module):
    def __init__(self, base_channel):
        super().__init__()
        c = base_channel
        self.conv0 = nn.Conv2d(c * 4, c * 2, 3, 1, 1)
        self.conv1 = nn.Conv2d(c * 4 + 1, c * 2, 3, 1, 1)
        self.res = ResBlock(c * 8 + 1, c * 4 + 1)

    def forward(self, mask, feature):
        s = self.conv0(mask * feature)
        w = self.conv1(torch.cat([feature, mask], dim=1))
        out = self.res(torch.cat([s, w, feature, mask], dim=1))
        return mask + out[:, :1], out[:, 1:]


def coverage(flow):
    """Number of source pixels landing on each target pixel (bilinear weights), (B, 1, H, W)."""
    ones = flow.new_ones((flow.shape[0], 1, flow.shape[2], flow.shape[3]))
    return softsplat(ones, flow, None, 'sum')


class MaskGuideNet(nn.Module):
    def __init__(self, base_channel, mask_mode='geo', diff_tau=0.1, diff_temp=0.02):
        super().__init__()
        assert mask_mode in MASK_MODES, mask_mode
        c = base_channel
        self.mask_mode = mask_mode
        self.bwarp = BackwardWarp()
        self.refblock0 = RefBlock(c)
        self.refblock1 = RefBlock(c)
        self.res0 = ResBlock(c * 8 + 3, c * 4)
        self.conv0 = nn.Conv2d(3 + 3 + 1, c * 4, 3, 1, 1)
        self.conv1 = nn.Conv2d(c * 8, 3, 3, 1, 1)
        if mask_mode == 'geo':
            self.diff_tau = nn.Parameter(torch.tensor(float(diff_tau)))
            self.diff_temp = float(diff_temp)

    def forward(self, im0, im1, flow0t, flowt0, ctxt0, ctxt1, metric=None, return_aux=False):
        b, _, h, w = flow0t.shape
        Flowt0 = upsample_flow(flowt0)
        Flow0t = upsample_flow(flow0t)
        imt0 = self.bwarp(im0, Flowt0)

        ne = 'add' if self.mask_mode == 'ofaf' else 'zero'       # 'add' = OF-AF normalisation
        if metric is None:
            im0t = softsplat(im0, Flow0t, None, 'avg', norm_eps=ne)
        else:
            im0t = softsplat(im0, Flow0t, metric, 'soft', norm_eps=ne)

        diff_map = torch.sum(torch.abs(imt0 - im0t), dim=1, keepdim=True)
        if self.mask_mode == 'ofaf':
            ones = torch.ones((b, 1, h, w), device=im0.device, dtype=im0.dtype)
            hole_mask = 1 - softsplat(ones, flow0t, None, 'avg', norm_eps='add')
            diff_mask = (diff_map > 0.1).to(im0.dtype)
        else:
            hole_mask = 1 - coverage(flow0t).clamp(max=1.0)
            diff_mask = torch.sigmoid((diff_map - self.diff_tau) / self.diff_temp)

        hole_mask, ref0 = self.refblock0(hole_mask, ctxt0)
        hole_mask, ref1 = self.refblock1(hole_mask, ctxt1)
        im1_4 = F.interpolate(im1, scale_factor=0.25, mode='bilinear', align_corners=False)
        ref = self.res0(torch.cat([ref0, ref1, im1_4], dim=1))
        Ref = F.interpolate(ref, scale_factor=4, mode='bilinear', align_corners=False)
        Mask0t = F.interpolate(hole_mask, scale_factor=4, mode='bilinear', align_corners=False) + diff_mask
        Ref_source = self.conv0(torch.cat([imt0 * Mask0t, imt0, Mask0t], dim=1))
        imt0 = imt0 + self.conv1(torch.cat([Ref, Ref_source], dim=1))
        if not return_aux:
            return imt0, im0t, Mask0t
        with torch.no_grad():
            aux = dict(splat=im0t.detach(), cov=coverage(Flow0t.detach()), diff_mask=diff_mask.detach())
        return imt0, im0t, Mask0t, aux


class BiW(nn.Module):
    def __init__(self, base_channel, splat='avg', mask_mode='geo', alpha_init=5.0, metric_smooth=True,
                 diff_tau=0.1, diff_temp=0.02):
        super().__init__()
        assert splat in SPLATS, splat
        assert mask_mode in MASK_MODES, mask_mode
        if splat != 'avg' and mask_mode == 'ofaf':
            warnings.warn(f"splat='{splat}' with mask_mode='ofaf': the splatted image only reaches the output "
                          f"through a hard threshold, so alpha and C get no gradient. Use --mask_mode geo.")
        c = base_channel
        self.splat = splat
        self.mask_mode = mask_mode
        self.metric_smooth = metric_smooth
        self.bwarp = BackwardWarp()
        self.fuse_warp = MaskGuideNet(c, mask_mode, diff_tau, diff_temp)
        if splat != 'avg':
            self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha_init)))
        self.channel_squeeze0 = nn.Conv2d(4 * c, c, 1, 1, 0, bias=True)
        self.channel_squeeze1 = nn.Conv2d(4 * c, c, 1, 1, 0, bias=True)
        self.refine_blocks = ResBlockIF(3 + 4 * c, 2 * c)
        self.conv_out = nn.Conv2d(2 * c, 3, 1, 1, bias=True)

    def ebc_metric(self, src, flow_lr, lhat, eps=1e-2):
        """Z(p) = -alpha * rho(L_src(p) - L_hat(p + F_{src->t}(p))) <= 0."""
        ls = rgb_to_logy(src)
        lt = self.bwarp(lhat, upsample_flow(flow_lr))
        r = torch.sqrt((ls - lt) ** 2 + eps ** 2) - eps
        z = -self.log_alpha.exp() * r
        if self.metric_smooth:
            z = F.avg_pool2d(z, 3, 1, 1, count_include_pad=False)
        return z

    def forward(self, im0, im1, t, flow0t, flow1t, flowt0, flowt1, ctxt0, ctxt1, if0_1, if1_1,
                lhat0=None, lhat1=None, metric_flow0=None, metric_flow1=None, return_aux=False):
        """lhat*: target luminance for each side; metric_flow*: flow used to look it up
        (defaults to F_{0->t} / F_{1->t}; the 'niklaus' baseline passes F_{0->1} / F_{1->0})."""
        z0 = z1 = None
        if self.splat != 'avg':
            z0 = self.ebc_metric(im0, flow0t if metric_flow0 is None else metric_flow0, lhat0)
            z1 = self.ebc_metric(im1, flow1t if metric_flow1 is None else metric_flow1, lhat1)
        r0 = self.fuse_warp(im0, im1, flow0t, flowt0, ctxt0, ctxt1, z0, return_aux)
        r1 = self.fuse_warp(im1, im0, flow1t, flowt1, ctxt1, ctxt0, z1, return_aux)
        imt0, mask0t = r0[0], r0[2]
        imt1, mask1t = r1[0], r1[2]
        tt = t.to(im0.dtype).view(-1, 1, 1, 1)
        mask = torch.softmax(torch.cat([(1 - mask0t) * (1 - tt), (1 - mask1t) * tt], 1), 1)[:, :1]
        fuse_out = imt0 * mask + (1 - mask) * imt1

        f0 = F.interpolate(self.channel_squeeze0(ctxt0), scale_factor=4, mode='bilinear', align_corners=False)
        f1 = F.interpolate(self.channel_squeeze1(ctxt1), scale_factor=4, mode='bilinear', align_corners=False)
        refine_in = torch.cat((f0, f1, fuse_out, if0_1, if1_1), 1)
        out = self.conv_out(self.refine_blocks(refine_in)) + fuse_out
        if not return_aux:
            return out
        return out, dict(splat0=r0[3]['splat'], splat1=r1[3]['splat'], cov0=r0[3]['cov'], cov1=r1[3]['cov'],
                         diff0=r0[3]['diff_mask'], diff1=r1[3]['diff_mask'])

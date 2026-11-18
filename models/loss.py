import warnings

import torch
from torch import nn


class VFILoss(nn.Module):
    """Charbonnier + lpips_weight * LPIPS, as in OF-AF (LPIPS is optional).

    OF-AF feeds [0, 1] images to LPIPS without normalize=True; here the images
    are normalized to [-1, 1] as the lpips package expects.
    """

    def __init__(self, lpips_weight=0.1, lpips_net='vgg'):
        super().__init__()
        self.lpips_weight = lpips_weight
        self.lpips = None
        if lpips_weight > 0:
            try:
                import lpips
                self.lpips = lpips.LPIPS(net=lpips_net, verbose=False).eval()
                for p in self.lpips.parameters():
                    p.requires_grad_(False)
            except Exception as e:  # pragma: no cover
                warnings.warn(f'[loss] LPIPS disabled ({e}); training with Charbonnier only.')

    def forward(self, pred, gt):
        p = pred.flatten(0, 1)
        g = gt.flatten(0, 1)
        charb = torch.sqrt((p - g) ** 2 + 1e-6).mean()
        total = charb
        logs = {'charb': float(charb.detach())}
        if self.lpips is not None:
            lp = self.lpips(p, g, normalize=True).mean()
            total = total + self.lpips_weight * lp
            logs['lpips'] = float(lp.detach())
        logs['total'] = float(total.detach())
        return total, logs

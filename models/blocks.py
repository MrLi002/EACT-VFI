import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Same layout as OF-AF models/blocks.py ResBlock."""

    def __init__(self, in_channel, base_channel):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channel, base_channel, 3, 1, 1, bias=True)
        self.conv_block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x_ = self.conv_in(x)
        return self.conv_block(x_) + x_


class ResBlockIF(nn.Module):
    """Same layout as OF-AF models/blocks.py ResBlockIF (two residual stages)."""

    def __init__(self, in_channel, base_channel):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channel, base_channel, 3, 1, 1, bias=True)

        def stage():
            return nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
                nn.LeakyReLU(0.2),
                nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
                nn.LeakyReLU(0.2),
                nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
                nn.LeakyReLU(0.2),
            )

        self.conv_block = stage()
        self.conv_block1 = stage()

    def forward(self, x):
        x_ = self.conv_in(x)
        x_res = self.conv_block(x_) + x_
        return self.conv_block1(x_res) + x_res


_GRID_CACHE = {}


def base_grid(H, W, device, dtype):
    key = (H, W, str(device), dtype)
    g = _GRID_CACHE.get(key)
    if g is None:
        gy, gx = torch.meshgrid(torch.arange(H, device=device, dtype=dtype),
                                torch.arange(W, device=device, dtype=dtype), indexing='ij')
        g = (gx, gy)
        _GRID_CACHE[key] = g
    return g


class BackwardWarp(nn.Module):
    """out(p) = img(p + flow(p)), bilinear, border padding, pixel-exact coordinates."""

    def forward(self, img, flow):
        B, _, H, W = img.shape
        gx, gy = base_grid(H, W, img.device, img.dtype)
        x = gx.unsqueeze(0) + flow[:, 0]
        y = gy.unsqueeze(0) + flow[:, 1]
        x = 2.0 * x / max(W - 1, 1) - 1.0
        y = 2.0 * y / max(H - 1, 1) - 1.0
        grid = torch.stack((x, y), dim=3)
        return F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)


def upsample_flow(flow, scale=4):
    return F.interpolate(flow, scale_factor=scale, mode='bilinear', align_corners=False) * scale


def zero_init(module):
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module

import torch
from torch import nn


class EncResBlock(nn.Module):
    """OF-AF models/encoder.py ResBlock."""

    def __init__(self, in_channel, base_channel):
        super().__init__()
        self.conv_in = nn.Sequential(nn.Conv2d(in_channel, base_channel, 3, 1, 1, bias=True), nn.LeakyReLU(0.2))
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_in(x)


class EventsEncoder(nn.Module):
    """Encodes the segment voxel (B, n * bins_per_seg, H, W) into (B, n, 4c, H/4, W/4).

    Identical to OF-AF except that the input is the segment-wise voxel, so the
    k-th group of `bins_per_seg` channels covers exactly [b_k, b_{k+1}]
    (uniform or EATP boundaries).
    """

    def __init__(self, voxel_bins, base_channel, bins_per_seg=8):
        super().__init__()
        self.bins_per_seg = bins_per_seg
        self.pre_conv = EncResBlock(voxel_bins + 2 * 3, voxel_bins)
        self.ct = nn.Parameter(torch.ones((1, voxel_bins, 1, 1)) * 0.2, requires_grad=True)
        self.down0 = nn.Sequential(nn.Conv2d(bins_per_seg, 2 * base_channel, 3, 2, 1, bias=True), nn.LeakyReLU(0.2))
        self.down1 = nn.Sequential(nn.Conv2d(2 * base_channel, 4 * base_channel, 3, 2, 1, bias=True), nn.LeakyReLU(0.2))

    def forward(self, im0, im1, events):
        pre = self.pre_conv(torch.cat((im0, events, im1), 1)) * self.ct
        chunks = pre.split(self.bins_per_seg, 1)
        b = chunks[0].shape[0]
        e = self.down1(self.down0(torch.cat(chunks, 0)))
        return torch.stack(e.split(b, 0), 1)


class ImageEncoder(nn.Module):
    def __init__(self, img_ch, base_channel):
        super().__init__()
        self.preblock = EncResBlock(img_ch, base_channel)
        self.down0 = nn.Conv2d(base_channel, 2 * base_channel, 3, 2, 1, bias=True)
        self.down1 = nn.Conv2d(2 * base_channel, 4 * base_channel, 3, 2, 1, bias=True)

    def forward(self, img):
        f1 = self.preblock(img)
        f4 = self.down1(self.down0(f1))
        return f1, f4

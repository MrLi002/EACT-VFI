"""Bidirectional Flow Estimation Block (OF-AF) with segment-length conditioning (EATP).

With EATP the segments have different durations.  The recurrent step therefore
receives dt = (b_{k+1} - b_k) * n as an extra constant map, because the
displacement produced by one step scales with the segment duration.  With
uniform boundaries dt == 1 everywhere and the block reduces to OF-AF's.
"""
import torch
from torch import nn

from .blocks import ResBlock, BackwardWarp


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super().__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h


class BidirFlow(nn.Module):
    def __init__(self, base_channel, time_cond=True):
        super().__init__()
        c = base_channel
        self.time_cond = time_cond
        extra = 1 if time_cond else 0
        self.bwarp = BackwardWarp()
        self.res0 = ResBlock(c * 12 + extra, c * 4)
        self.res1 = ResBlock(c * 12 + extra, c * 4)
        self.res2 = ResBlock(c * 16 + 6, c * 4)
        self.gru = SepConvGRU(c * 4, c * 8)
        self.conv0 = nn.Conv2d(c * 8, 2, 3, 1, 1)
        self.conv1 = nn.Conv2d(c * 12, 2, 3, 1, 1)
        self.conv2 = nn.Conv2d(c * 4, 2, 3, 1, 1)
        self.flow = nn.Sequential(nn.Conv2d(c * 4, c * 8, 3, 1, 1), nn.ReLU(), nn.Conv2d(c * 8, 2, 3, 1, 1))

    def forward(self, if0, ift0, flowt0, motion0t, ctxt0, voxel, dt=None):
        """One recurrent step over one segment.

        if0      : key-frame features (B, 4c, h, w)
        ift0     : key-frame features warped to the current anchor time
        flowt0   : backward flow (anchor -> key frame) accumulated so far
        motion0t : motion state
        ctxt0    : context state
        voxel    : event features of this segment (B, 4c, h, w)
        dt       : (B, 1, h, w) segment length * n (EATP); ignored if time_cond=False
        """
        if self.time_cond:
            if dt is None:
                dt = torch.ones_like(voxel[:, :1])
            ctxt0_cur = self.res0(torch.cat([ctxt0, voxel, ift0, dt], dim=1))
            motion0t_cur = self.res1(torch.cat([ctxt0_cur, motion0t, voxel, dt], dim=1))
        else:
            ctxt0_cur = self.res0(torch.cat([ctxt0, voxel, ift0], dim=1))
            motion0t_cur = self.res1(torch.cat([ctxt0_cur, motion0t, voxel], dim=1))

        flow0 = self.conv0(torch.cat([ift0, ctxt0_cur], dim=1))
        ift0_cur0 = self.bwarp(ift0, flow0)
        flow1 = flowt0 + flow0
        ift0_cur1 = self.bwarp(if0, flow1)
        flow2 = self.conv1(torch.cat([if0, motion0t_cur, ift0_cur0], dim=1))
        ift0_cur2 = self.bwarp(if0, flow2)

        final = self.res2(torch.cat([if0, ift0_cur0, ift0_cur1, ift0_cur2, flow0, flow1, flow2], dim=1))
        flowt0 = self.conv2(final) + flow2
        ift0_cur = self.bwarp(if0, flowt0)

        flow0t = self.flow(self.gru(if0, torch.cat([ctxt0_cur, motion0t_cur], dim=1)))
        return ift0_cur, motion0t_cur, flow0t, flowt0, ctxt0_cur

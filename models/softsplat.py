"""Forward warping (splatting) in pure PyTorch.

Drop-in replacement for the cupy kernel used by OF-AF: same call signature
softsplat(tenIn, tenFlow, tenMetric, strMode) and the same modes

    'sum'    : sum of bilinearly splatted values
    'avg'    : summation splatting normalized by the splatted weights
    'linear' : weighted by tenMetric
    'soft'   : weighted by exp(tenMetric)            (softmax splatting)

Gradients flow to tenIn, tenFlow (through the bilinear weights) and tenMetric.
No CUDA extension or cupy is needed, so it also runs on CPU for testing.
Reference: Niklaus & Liu, Softmax Splatting for Video Frame Interpolation, CVPR 2020.
"""
import torch

from .blocks import base_grid


def _forward_splat(tenIn, tenFlow):
    B, C, H, W = tenIn.shape
    gx, gy = base_grid(H, W, tenIn.device, tenIn.dtype)
    flow = torch.nan_to_num(tenFlow)
    x = gx.unsqueeze(0) + flow[:, 0]
    y = gy.unsqueeze(0) + flow[:, 1]
    x0 = torch.floor(x)
    y0 = torch.floor(y)

    src = tenIn.permute(0, 2, 3, 1).reshape(B * H * W, C)
    bidx = torch.arange(B, device=tenIn.device).view(B, 1, 1).expand(B, H, W)
    out = tenIn.new_zeros(B * H * W, C)
    for ox, oy in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
        xi = x0 + ox
        yi = y0 + oy
        w = (1.0 - (x - xi).abs()) * (1.0 - (y - yi).abs())
        valid = ((xi >= 0) & (xi <= W - 1) & (yi >= 0) & (yi <= H - 1)).reshape(-1)
        idx = ((bidx * H + yi.clamp(0, H - 1).long()) * W + xi.clamp(0, W - 1).long()).reshape(-1)
        out = out.index_add(0, idx[valid], (src * w.reshape(-1, 1))[valid])
    return out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()


def softsplat(tenIn, tenFlow, tenMetric=None, strMode='avg', eps=1e-7, return_norm=False, norm_eps='zero'):
    """norm_eps: 'zero' -> pixels with weight <= eps are set to 0, others divided exactly;
                 'add'  -> divide by (weight + eps), the default of OF-AF's cupy softsplat
                           (so avg_splat(1) = s / (s + 1e-7) is not exactly binary)."""
    assert strMode in ('sum', 'avg', 'linear', 'soft'), strMode
    assert norm_eps in ('zero', 'add'), norm_eps
    if strMode in ('sum', 'avg'):
        assert tenMetric is None
    else:
        assert tenMetric is not None

    if strMode == 'avg':
        tenIn = torch.cat([tenIn, torch.ones_like(tenIn[:, :1])], 1)
    elif strMode == 'linear':
        tenIn = torch.cat([tenIn * tenMetric, tenMetric], 1)
    elif strMode == 'soft':
        e = tenMetric.clamp(max=30.0).exp()
        tenIn = torch.cat([tenIn * e, e], 1)

    out = _forward_splat(tenIn, tenFlow)
    if strMode == 'sum':
        return out

    norm = out[:, -1:]
    if norm_eps == 'add':
        res = out[:, :-1] / (norm + eps)
        return (res, norm) if return_norm else res
    safe = torch.where(norm > eps, norm, torch.ones_like(norm))
    res = torch.where(norm > eps, out[:, :-1] / safe, torch.zeros_like(out[:, :-1]))
    if return_norm:
        return res, norm
    return res

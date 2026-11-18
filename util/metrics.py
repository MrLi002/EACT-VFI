"""Evaluation metrics.

Image quality : PSNR, SSIM (scikit-image), LPIPS (optional, `pip install lpips`).
Temporal      : tOF (needs GT frames) and consecutive warping error (needs no GT;
                used for densely interpolated sequences at off-grid times).
Flows for the temporal metrics come from OpenCV DIS so that no extra network is
needed; report which flow was used when comparing with published tOF numbers.
"""
import warnings

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(gt, pred):
    return float(peak_signal_noise_ratio(gt, pred, data_range=255))


def ssim(gt, pred):
    return float(structural_similarity(gt, pred, data_range=255, channel_axis=2))


def to_uint8(img_chw, rounding=False):
    """(3, H, W) float tensor/array in [0, 1] -> (H, W, 3) uint8.

    rounding=False truncates like OF-AF's evaluation ((x * 255).astype(uint8)),
    keeping PSNR/SSIM comparable with the numbers reported there.
    """
    if hasattr(img_chw, 'detach'):
        img_chw = img_chw.detach().float().cpu().numpy()
    img = np.clip(img_chw.transpose(1, 2, 0), 0.0, 1.0) * 255.0
    return (img + 0.5 if rounding else img).astype(np.uint8)


class LPIPSMetric:
    """Thin wrapper; disabled automatically if the lpips package is missing."""

    def __init__(self, net='alex', device='cuda'):
        self.fn = None
        try:
            import lpips
            import torch  # noqa: F401
            self.fn = lpips.LPIPS(net=net, verbose=False).to(device).eval()
        except Exception as e:  # pragma: no cover
            warnings.warn(f"[metrics] LPIPS disabled ({e}).")
        self.device = device

    def __call__(self, pred, gt):
        """pred, gt: (1, 3, H, W) torch tensors in [0, 1]."""
        if self.fn is None:
            return float('nan')
        import torch
        with torch.no_grad():
            return float(self.fn(pred.to(self.device), gt.to(self.device), normalize=True).mean().item())


_DIS = [None]


def _dis():
    if _DIS[0] is None:
        _DIS[0] = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _DIS[0]


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def dis_flow(a, b):
    """Flow from frame a to frame b (both HxWx3 uint8 RGB)."""
    return _dis().calc(_gray(a), _gray(b), None)


def _warp(img, flow):
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def tof(preds, gts):
    """Temporal optical-flow difference between consecutive predicted and GT frames."""
    vals = []
    for i in range(len(preds) - 1):
        fp = dis_flow(preds[i], preds[i + 1])
        fg = dis_flow(gts[i], gts[i + 1])
        vals.append(float(np.mean(np.linalg.norm(fp - fg, axis=2))))
    return float(np.mean(vals)) if vals else float('nan')


def consecutive_warp_error(frames, fb_thresh=1.0):
    """Mean |I_t - warp(I_{t+1}, F_{t->t+1})| over non-occluded pixels (0-255 scale).

    Lower is smoother.  Occlusions are removed with a forward-backward check.
    """
    vals = []
    for i in range(len(frames) - 1):
        a, b = frames[i], frames[i + 1]
        f_ab = dis_flow(a, b)
        f_ba = dis_flow(b, a)
        f_ba_w = _warp(f_ba, f_ab)
        fb = np.linalg.norm(f_ab + f_ba_w, axis=2)
        valid = fb < fb_thresh + 0.05 * (np.linalg.norm(f_ab, axis=2) + np.linalg.norm(f_ba_w, axis=2))
        b_w = _warp(b, f_ab).astype(np.float32)
        err = np.abs(a.astype(np.float32) - b_w).mean(axis=2)
        if valid.any():
            vals.append(float(err[valid].mean()))
    return float(np.mean(vals)) if vals else float('nan')


def grid_fraction(t, n):
    """Fractional position of time t inside the uniform 1/n grid (0 = on a grid point)."""
    x = float(t) * n
    frac = x - np.floor(x)
    return 0.0 if (frac < 1e-6 or 1.0 - frac < 1e-6) else float(frac)

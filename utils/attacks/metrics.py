# attack/metrics.py
"""
Visual similarity metrics used to bound visibility of perturbations.
- SSIM: implemented in pure PyTorch (channel-wise, Gaussian window).
- LPIPS: optional (returns None unless the 'lpips' package is available).
- FID/KID: placeholders (return (None, None) for Milestone-1).
"""
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

# ---------------- SSIM ----------------

def _gaussian_kernel(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype):
    x = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel2d = (g.view(1, 1, -1, 1) @ g.view(1, 1, 1, -1)).squeeze(0)  # [1, 1, w, w] -> [w, w]
    return kernel2d.expand(channels, 1, window_size, window_size).contiguous()

def _ssim_per_channel(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5):
    """
    x,y: float tensors in [0,1], shape [B, C, H, W]
    returns: [B, C] SSIM per channel
    """
    assert x.shape == y.shape and x.ndim == 4, "SSIM expects [B,C,H,W] tensors with same shape"
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype
    kernel = _gaussian_kernel(window_size, sigma, C, device, dtype)

    mu_x = F.conv2d(x, kernel, padding=window_size // 2, groups=C)
    mu_y = F.conv2d(y, kernel, padding=window_size // 2, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=window_size // 2, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=window_size // 2, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=window_size // 2, groups=C) - mu_xy

    L = 1.0
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map.mean(dim=[2, 3])  # [B, C]

def batch_ssim(x: torch.Tensor, y: torch.Tensor) -> Optional[float]:
    """
    Returns mean SSIM over batch and channels, or None on failure.
    Expects x,y in [0,1] float, shape [B,C,H,W].
    """
    try:
        x = x.clamp(0, 1)
        y = y.clamp(0, 1)
        ssim_bc = _ssim_per_channel(x, y)  # [B, C]
        ssim = ssim_bc.mean(dim=1).mean()
        return float(ssim.detach().cpu())
    except Exception:
        return None

# ---------------- LPIPS (optional) ----------------

_lpips_net = None

def _get_lpips():
    global _lpips_net
    if _lpips_net is not None:
        return _lpips_net
    try:
        import lpips  # type: ignore
        _lpips_net = lpips.LPIPS(net='alex').eval().to('cpu')
        return _lpips_net
    except Exception:
        return None

@torch.no_grad()
def batch_lpips(x: torch.Tensor, y: torch.Tensor) -> Optional[float]:
    """
    Returns mean LPIPS if the 'lpips' package is available; otherwise None.
    Expects x,y in [0,1] float, shape [B,C,H,W].
    """
    net = _get_lpips()
    if net is None:
        return None
    try:
        def to_m1p1(t):  # [0,1] -> [-1,1]
            return t.clamp(0, 1) * 2.0 - 1.0
        d = net(to_m1p1(x.cpu()), to_m1p1(y.cpu()))
        return float(d.mean().detach().cpu())
    except Exception:
        return None

# ---------------- FID/KID placeholders ----------------

def batch_fid_kid(x: torch.Tensor, y: torch.Tensor) -> Tuple[Optional[float], Optional[float]]:
    """
    Placeholders for Milestone-1: computing FID/KID reliably requires a dedicated
    pipeline and cached activations; return (None, None) to avoid misuse.
    """
    return None, None

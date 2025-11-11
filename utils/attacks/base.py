import time
from typing import Tuple, Dict, Any, Optional
import torch

from . import metrics as vis_metrics


class AttackBase:
    """
    Minimal no-op attack with accounting for:
      - Impact budget: edited fraction, eps, steps, queries, added time
      - Visibility: stores small pairs of (orig, edited) tensors and (optionally) computes SSIM/LPIPS

    Notes for visibility:
      • We store raw float CPU tensors (no clamping at cache-time).
      • During summarize(..., compute_visibility=True) we convert to pixel space [0,1] if possible.
      • To enable correct de-normalization, call `set_visibility_normalization(mean, std)` once
        (e.g., from the training loop or the attack when dataset stats are available).
      • If we cannot safely convert to [0,1], visibility metrics return None (no crashes).
    """
    def __init__(self, args):
        self.args = args
        self.eps = float(getattr(args, 'eps', 0.0) or 0.0)
        self.steps = int(getattr(args, 'attack_steps', 0) or 0)
        self.queries = int(getattr(args, 'attack_queries', 0) or 0)
        self.rho = float(getattr(args, 'rho', 0.0) or 0.0)
        self._stats = {
            'wake': {'seen': 0, 'edited': 0, 'time': 0.0, 'queries': 0},
            'nrem': {'seen': 0, 'edited': 0, 'time': 0.0, 'queries': 0},
        }
        # Visibility storage
        self._vis_cap = int(getattr(args, 'vis_cap', 512))  # cap total stored samples per phase
        self._vis = {'wake': [], 'nrem': []}                # list of (orig, adv) raw float CPU tensors
        # Optional normalization used to convert normalized tensors to pixel-space [0,1]
        self._vis_norm_mean: Optional[torch.Tensor] = None  # shape [1,C,1,1] on CPU
        self._vis_norm_std: Optional[torch.Tensor] = None   # shape [1,C,1,1] on CPU

    # ----------------- visibility normalization API -----------------

    def set_visibility_normalization(self, mean, std, device: str = 'cpu'):
        """
        Register per-channel mean/std used for de-normalization in visibility computations.
        `mean` and `std` can be lists/tuples or tensors of shape [C] or [1,C,1,1].
        """
        mean_t = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1).to(device)
        std_t = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1).to(device)
        # Store on CPU to keep memory light and avoid device mismatches later
        self._vis_norm_mean = mean_t.cpu()
        self._vis_norm_std = std_t.cpu()

    # ----------------- accounting helpers -----------------

    def _start(self):
        return time.perf_counter()

    def _end(self, t0):
        return time.perf_counter() - t0

    def _record(self, phase: str, n_seen: int, n_edited: int, dt: float, queries: int = 0):
        st = self._stats[phase]
        st['seen'] += int(n_seen)
        st['edited'] += int(n_edited)
        st['time'] += float(dt)
        st['queries'] += int(queries)

    # ----------------- visibility caching -----------------

    @torch.no_grad()
    def cache_visibility(self, phase: str, orig: torch.Tensor, adv: torch.Tensor):
        """
        Cache a small subset of (orig, adv) pairs for later visibility computation.
        We store raw float CPU tensors and convert to [0,1] pixel-space during summarize().

        NOTE: If your pipeline knows mean/std, call `set_visibility_normalization(mean, std)` once;
              otherwise visibility metrics may return None for normalized inputs.
        """
        if phase not in self._vis:
            return
        need = self._vis_cap - len(self._vis[phase])
        if need <= 0:
            return
        k = min(need, orig.shape[0])

        def _to_float_cpu(x: torch.Tensor) -> torch.Tensor:
            if x.dtype == torch.uint8:
                return x[:k].float().cpu() / 255.0
            return x[:k].detach().float().cpu()

        o = _to_float_cpu(orig)
        a = _to_float_cpu(adv)
        self._vis[phase].append((o, a))

    # ----------------- default no-op hooks -----------------

    def on_wake_batch(self, inputs: torch.Tensor, labels: torch.Tensor, **ctx) -> Tuple[torch.Tensor, torch.Tensor]:
        t0 = self._start()
        self._record('wake', n_seen=inputs.size(0), n_edited=0, dt=self._end(t0), queries=0)
        return inputs, labels

    def on_nrem_batch(self, inputs: torch.Tensor, labels: torch.Tensor, **ctx) -> Tuple[torch.Tensor, torch.Tensor]:
        t0 = self._start()
        self._record('nrem', n_seen=inputs.size(0), n_edited=0, dt=self._end(t0), queries=0)
        return inputs, labels

    # ----------------- summarization & visibility -----------------

    def _looks_normalized(self, x: torch.Tensor) -> bool:
        # Heuristic: normalized tensors often have values outside [0,1]
        xmin, xmax = float(x.min()), float(x.max())
        return (xmin < -0.05) or (xmax > 1.05)

    def _to_pixel_for_vis(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Convert a cached raw float CPU tensor to pixel-space [0,1] if possible.
        Returns None if we cannot safely convert (e.g., normalized but no mean/std provided).
        """
        if x.ndim != 4:
            return None
        # Already [0,1]?
        xmin, xmax = float(x.min()), float(x.max())
        if -0.01 <= xmin and xmax <= 1.01:
            return x.clamp(0.0, 1.0)
        # Looks normalized? try de-normalize if stats are provided
        if self._looks_normalized(x):
            if self._vis_norm_mean is None or self._vis_norm_std is None:
                return None
            mean = self._vis_norm_mean
            std = self._vis_norm_std
            # Broadcast to batch
            mean_b = mean.expand(x.size(0), -1, -1, -1)
            std_b = std.expand(x.size(0), -1, -1, -1)
            x_pix = (x * std_b) + mean_b
            return x_pix.clamp(0.0, 1.0)
        # Values > 1 but not normalized (e.g., [0,255] scaled float)
        if xmax > 1.5 and xmin >= 0.0:
            return (x / 255.0).clamp(0.0, 1.0)
        # Fallback: clamp (may be inaccurate), better than crashing
        return x.clamp(0.0, 1.0)

    def summarize(self, reset: bool = True, compute_visibility: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for ph in ('wake', 'nrem'):
            st = self._stats[ph]
            seen = max(1, st['seen'])
            out[f'ATTACK/{ph}_edited_frac'] = float(st['edited']) / float(seen)
            out[f'ATTACK/{ph}_edited_count'] = st['edited']
            out[f'ATTACK/{ph}_seen'] = seen
            out[f'ATTACK/{ph}_time_sec'] = st['time']
            out[f'ATTACK/{ph}_queries'] = st['queries']
        out['ATTACK/eps'] = self.eps
        out['ATTACK/steps'] = self.steps
        out['ATTACK/queries_budget'] = self.queries
        out['ATTACK/rho'] = self.rho
        out['ATTACK/name'] = getattr(self.args, 'attack', 'none')
        out['ATTACK/phase'] = getattr(self.args, 'attack_phase', 'none')

        if compute_visibility:
            for ph in ('wake', 'nrem'):
                pairs = self._vis[ph]
                if not pairs:
                    continue
                # Concatenate cached pairs
                origs_raw = torch.cat([p[0] for p in pairs], dim=0)
                advs_raw = torch.cat([p[1] for p in pairs], dim=0)
                # Convert to pixel space if possible
                origs = self._to_pixel_for_vis(origs_raw)
                advs = self._to_pixel_for_vis(advs_raw)
                if origs is None or advs is None:
                    out[f'VIS/{ph}_ssim'] = None
                    out[f'VIS/{ph}_lpips'] = None
                else:
                    ssim = vis_metrics.batch_ssim(origs, advs)
                    lpips = vis_metrics.batch_lpips(origs, advs)
                    out[f'VIS/{ph}_ssim'] = None if ssim is None else float(ssim)
                    out[f'VIS/{ph}_lpips'] = None if lpips is None else float(lpips)

        if reset:
            self._stats = {k: {'seen': 0, 'edited': 0, 'time': 0.0, 'queries': 0} for k in self._stats}
            self._vis = {'wake': [], 'nrem': []}
        return out

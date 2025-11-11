

import sys
from argparse import Namespace
from contextlib import suppress
from typing import List
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from utils.conf import get_device
from utils.magic import persistent_locals
from datasets import get_dataset
from torchvision import transforms

with suppress(ImportError):
    import wandb


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME: str
    COMPATIBILITY: List[str]

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                 args: Namespace, transform: nn.Module) -> None:
        super(ContinualModel, self).__init__()

        ds = get_dataset(args)
        self.cpt = ds.N_CLASSES_PER_TASK
        self.n_task = ds.N_TASKS
        self.num_classes = self.n_task * self.cpt
        
        self.train_transform = ds.TRANSFORM
        self.test_transform = ds.TEST_TRANSFORM if hasattr(ds, 'TEST_TRANSFORM') else transforms.Compose(
            [transforms.ToTensor(), ds.get_normalization_transform()])

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        self.device = self.args.device

        # ---- SCDT/P-SCDT (stealth-constrained replay) lightweight state (kept inert unless enabled) ----
        # Rolling window of kept-replay histograms (GLOBAL basis; CPU tensors)
        self._scdt_hist_window = deque(maxlen=0)  # each entry: shape [self.num_classes], sums to 1 on CPU
        # Optional nominal histogram EMA (only used if args.scdt_nominal == 'ema'; present-class basis)
        self._scdt_nominal_ema = None
        # Optional per-class loss EMA for preference tilt (GLOBAL basis over self.num_classes)
        self._scdt_class_loss_ma = None
        self._scdt_class_loss_mom = 0.9  # default momentum if not overridden
        # Last-step audit dictionary for logging (training loop reads this)
        self._scdt_last_audit = {}
        # -----------------------------------------------------------------------------------------------

        if not self.NAME or not self.COMPATIBILITY:
            raise NotImplementedError('Please specify the name and the compatibility of the model.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def forward_all(self, x: torch.Tensor):
        """
        Return (logits, penultimate_features) using a classifier-input forward hook.
        This is model-agnostic and works for any backbone with a final Linear 'classifier'
        (or 'fc'); if not found, it falls back to the last nn.Linear layer.

        The method leaves the model's train/eval mode unchanged.
        """
        feats = {}

        # Prefer explicit classifier module
        base = self.net
        hook_mod = None
        if hasattr(base, "classifier"):
            hook_mod = base.classifier
        elif hasattr(base, "fc"):
            hook_mod = getattr(base, "fc")
        else:
            # Fallback: try the last Linear layer found
            for m in base.modules():
                if isinstance(m, nn.Linear):
                    hook_mod = m

        if hook_mod is not None:
            def _hook(m, inp, out):
                # Capture input to the classifier: penultimate features
                feats["f"] = inp[0]
            h = hook_mod.register_forward_hook(_hook)
            try:
                logits = self.net(x)
            finally:
                h.remove()
            if "f" in feats:
                return logits, feats["f"]

        # Last resort: return logits for both to avoid crashes (not ideal for feature attacks)
        logits = self.net(x)
        return logits, logits

    def meta_observe(self, *args, **kwargs):
        if 'wandb' in sys.modules and not self.args.nowand:
            pl = persistent_locals(self.observe)
            ret = pl(*args, **kwargs)
            self.autolog_wandb(pl.locals)
        else:
            ret = self.observe(*args, **kwargs)
        return ret

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        raise NotImplementedError

    def autolog_wandb(self, locals):
        """
        All variables starting with "_wandb_" or "loss" in the observe function
        are automatically logged to wandb upon return if wandb is installed.
        """
        if not self.args.nowand and not self.args.debug_mode:
            wandb.log({k: (v.item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v)
                      for k, v in locals.items() if k.startswith('_wandb_') or k.startswith('loss')})
            
    @torch.no_grad()
    def apply_transform(self, inputs, transform, device=None, add_pil_transforms=True):
        tr = transforms.Compose([transforms.ToPILImage()] + transform.transforms) if add_pil_transforms else transform
        device = self.device if device is None else device
        if len(inputs.shape) == 3:
            return tr(inputs)
        return torch.stack([tr(inp) for inp in inputs.cpu()], dim=0).to(device)

    @torch.no_grad()
    def aug_batch(self, not_aug_inputs, device=None):
        """
        Full train transform 
        """
        return self.apply_transform(not_aug_inputs, self.train_transform, device=device)

    @torch.no_grad()
    def test_data_aug(self, inputs, device=None):
        """
        Test transform
        """
        return self.apply_transform(inputs, self.test_transform, device=device)
    

    def reset_classifier(self):
        self.net.classifier = torch.nn.Linear(
                self.net.classifier.in_features, self.net.num_classes).to(self.device)
        self.reset_opt()

    def reset_opt(self):
        self.opt = get_dataset(self.args).get_optimizer(self.net.parameters(), self.args)
        return self.opt

    # -----------------------------------------------------------------------------------
    # SCDT / P-SCDT helpers
    # -----------------------------------------------------------------------------------
    @torch.no_grad()
    def _scdt_update_class_loss_ema(self, labels: torch.Tensor, per_sample_ce: torch.Tensor,
                                    momentum: float = None) -> None:
        """
        Update a per-class EMA of cross-entropy on the *current* batch (for preference u).
        Called by training code in ER / DER++ / ER-ACE after computing per-sample CE.
        """
        if labels is None or per_sample_ce is None or labels.numel() == 0:
            return
        if momentum is None:
            momentum = float(getattr(self.args, 'scdt_util_mom', self._scdt_class_loss_mom))
        if self._scdt_class_loss_ma is None or self._scdt_class_loss_ma.numel() != self.num_classes:
            self._scdt_class_loss_ma = torch.zeros(self.num_classes, dtype=torch.float, device=self.device)
        # compute per-class means on this mini-batch
        uniq, inv = labels.unique(return_inverse=True)
        for i, c in enumerate(uniq):
            mask = (inv == i)
            if torch.any(mask):
                v = per_sample_ce[mask].mean()
                c_int = int(c.item())
                self._scdt_class_loss_ma[c_int] = momentum * self._scdt_class_loss_ma[c_int] + (1 - momentum) * v

    @staticmethod
    def _kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
        p = p.clamp(min=1e-12)
        q = q.clamp(min=1e-12)
        return float((p * (p / q).log()).sum().item())

    @staticmethod
    def _tv_distance(p: torch.Tensor, q: torch.Tensor) -> float:
        return float(0.5 * (p - q).abs().sum().item())

    @staticmethod
    def _largest_remainder_rounding(t: torch.Tensor, m: int) -> torch.LongTensor:
        q = torch.floor(t).long()
        R = int(m - int(q.sum().item()))
        if R <= 0:
            return q
        frac = (t - q.float())
        order = torch.argsort(frac, descending=True)
        q[order[:R]] += 1
        return q

    @torch.no_grad()
    def _integerize_with_caps(self, p_star: torch.Tensor, m: int,
                              upper_cap: torch.LongTensor, u: torch.Tensor):
        """
        Hamilton rounding + availability caps; leftover assigned to highest-u classes.
        Returns (q, rounding_L1_gap).
        """
        t = p_star * float(m)
        q = self._largest_remainder_rounding(t, m).to(upper_cap.device)
        rounding_l1_gap = float((q.float() / max(1, m) - p_star).abs().sum().item())
        q = torch.minimum(q, upper_cap)
        s = int(q.sum().item())
        if s < m:
            R = int(m - s)
            cap = (upper_cap - q)
            if int(cap.sum().item()) > 0:
                order = torch.argsort(u, descending=True)
                for idx in order.tolist():
                    if R <= 0:
                        break
                    add = int(min(int(cap[idx].item()), R))
                    if add > 0:
                        q[idx] += add
                        R -= add
        return q, rounding_l1_gap

    @torch.no_grad()
    def _post_audit_transfer(self, q: torch.LongTensor, pi_nom: torch.Tensor, delta: float,
                             upper_cap: torch.LongTensor, u: torch.Tensor, div: str) -> torch.LongTensor:
        """
        Greedy unit transfers to restore Div(q/m || pi_nom) ≤ delta after discretization/availability.
        """
        m = int(q.sum().item())
        if m <= 0:
            return q
        for _ in range(4 * m + 10):
            p_bar = q.float() / float(m)
            D = self._tv_distance(p_bar, pi_nom) if div == 'TV' else self._kl_divergence(p_bar, pi_nom)
            if D <= delta + 1e-12:
                break
            dev = p_bar - pi_nom
            donors = torch.nonzero((dev > 0.0) & (q > 0), as_tuple=False).squeeze(1)
            receivers = torch.nonzero((dev < 0.0) & (q < upper_cap), as_tuple=False).squeeze(1)
            if donors.numel() == 0 or receivers.numel() == 0:
                break
            d_idx = donors[torch.argsort(u[donors], descending=False)[0]].item()
            r_idx = receivers[torch.argsort(u[receivers], descending=True)[0]].item()
            if d_idx == r_idx:
                break
            q[d_idx] -= 1
            q[r_idx] += 1
        return q

    @torch.no_grad()
    def _pscdt_kl_optimal(self, pi_nom: torch.Tensor, u: torch.Tensor, delta: float) -> torch.Tensor:
        """
        KL-ball optimizer: p*(c) ∝ pi_nom(c) * exp(alpha * u_c) with alpha≥0 tuned to KL=delta.
        """
        C = int(pi_nom.numel())
        if delta <= 1e-12 or C == 0 or float(u.var().item()) <= 1e-16:
            return pi_nom.clone()
        # KL requires positive support
        if torch.any(pi_nom <= 0):
            eps = 1e-6
            pi_nom = (1 - eps) * pi_nom + eps / float(max(1, C))
        def tilt(alpha: float) -> torch.Tensor:
            z = torch.exp(alpha * u)
            p = (pi_nom * z)
            return p / p.sum().clamp(min=1e-12)
        hi = 1.0
        p_hi = tilt(hi)
        kl_hi = self._kl_divergence(p_hi, pi_nom)
        it = 0
        while kl_hi < delta and it < 40:
            hi *= 2.0
            p_hi = tilt(hi)
            kl_hi = self._kl_divergence(p_hi, pi_nom)
            it += 1
            if hi > 1e6:
                break
        if kl_hi < delta:
            return p_hi
        lo = 0.0
        p_star = p_hi
        for _ in range(35):
            mid = 0.5 * (lo + hi)
            pm = tilt(mid)
            km = self._kl_divergence(pm, pi_nom)
            if km > delta:
                hi = mid
                p_star = pm
            else:
                lo = mid
        return p_star

    @torch.no_grad()
    def _pscdt_tv_optimal(self, pi_nom: torch.Tensor, u: torch.Tensor, delta: float) -> torch.Tensor:
        """
        TV-ball optimizer: two-sided water-filling that moves total mass delta
        from low-u to high-u classes; saturate at [0,1] and renormalize.
        """
        C = int(pi_nom.numel())
        if delta <= 1e-12 or C == 0:
            return pi_nom.clone()
        p = pi_nom.clone()
        order_desc = torch.argsort(u, descending=True)
        order_asc = torch.flip(order_desc, dims=[0])
        i, j, moved = 0, 0, 0.0
        while moved < delta - 1e-12 and i < C and j < C:
            r = int(order_desc[i].item())
            d = int(order_asc[j].item())
            if r == d:
                j += 1
                continue
            inc_cap = float(1.0 - p[r].item())
            dec_cap = float(p[d].item())
            if inc_cap <= 1e-12:
                i += 1; continue
            if dec_cap <= 1e-12:
                j += 1; continue
            mv = min(delta - moved, inc_cap, dec_cap)
            if mv <= 0:
                break
            p[r] += mv
            p[d] -= mv
            moved += mv
            if p[r] >= 1.0 - 1e-12:
                i += 1
            if p[d] <= 1e-12:
                j += 1
        p = p.clamp(min=0.0)
        p = p / p.sum().clamp(min=1e-12)
        return p

    @staticmethod
    def _blend(pi_nom: torch.Tensor, p_star: torch.Tensor, lam: float) -> torch.Tensor:
        """
        Blend toward the optimum: p_lam ∝ (1-λ)*pi_nom + λ*p_star (then renormalize).
        """
        p = (1.0 - lam) * pi_nom + lam * p_star
        return p / p.sum().clamp(min=1e-12)

    # -----------------------------------------------------------------------------------
    # SCDT / P-SCDT: central replay-batch helper (minimal, opt-in via args.scdt == 1)
    # -----------------------------------------------------------------------------------
    @torch.no_grad()
    def _get_replay_batch(self, n: int, include_logits: bool = False):
        """
        Return a replay mini-batch of size n. If args.scdt==1, select indices by
        Stealth-Constrained Distributional Trimming (SCDT) / P-SCDT with optional preference
        tilt; otherwise, delegate to buffer.get_data(...) when available, falling
        back to uniform random sampling.

        Output shape/contract:
            returns (buf_inputs, buf_labels, buf_logits_or_None)
        """
        # Sanity checks
        if not hasattr(self, 'buffer') or self.buffer is None:
            raise RuntimeError("Replay requested but `self.buffer` is not set on the model.")

        # If SCDT disabled, prefer the original API to preserve buffer semantics.
        scdt_enabled = int(getattr(self.args, 'scdt', 0)) == 1
        if not scdt_enabled:
            if hasattr(self.buffer, 'get_data'):
                out = self.buffer.get_data(n, transform=self.transform)
                # Normalize return to 3-tuple
                if isinstance(out, (list, tuple)):
                    if len(out) == 3:
                        return out[0], out[1], out[2]
                    elif len(out) == 2:
                        return out[0], out[1], None
            # Fallback: uniform random without replacement
            return self._legacy_uniform_replay(n, include_logits=include_logits)

        # ---------- SCDT/P-SCDT path ----------
        # Pull pool tensors
        assert hasattr(self, 'examples') or hasattr(self.buffer, 'examples')
        assert hasattr(self.buffer, 'examples') and hasattr(self.buffer, 'labels'), \
            "Buffer must expose .examples and .labels for SCDT sampling."
        X_pool = self.buffer.examples
        Y_pool = self.buffer.labels
        N_pool = int(Y_pool.shape[0])
        if N_pool <= 0:
            # Nothing to sample
            return self._legacy_uniform_replay(n, include_logits=include_logits)

        n = int(min(n, N_pool))
        device = Y_pool.device if isinstance(Y_pool, torch.Tensor) else self.device

        # Build class partitions over the pool
        classes, counts = Y_pool.unique(return_counts=True)
        C = int(classes.numel())
        classes = classes.long()
        classes_cpu = classes.cpu()  # for global-basis ops

        # Precompute per-class index lists
        per_class_idxs = []
        for k in range(C):
            mask_k = (Y_pool == classes[k])
            per_class_idxs.append(torch.nonzero(mask_k, as_tuple=False).squeeze(1))

        # Nominal histogram π_nom (present-class basis)
        nominal_mode = getattr(self.args, 'scdt_nominal', 'buffer')
        if nominal_mode == 'uniform':
            pi_nom = torch.full((C,), 1.0 / max(1, C), dtype=torch.float, device=device)
        elif nominal_mode == 'ema':
            # Minimal EMA support: initialize full-vector EMA lazily and update on observed classes
            if self._scdt_nominal_ema is None or self._scdt_nominal_ema.numel() != C:
                self._scdt_nominal_ema = (counts.float() / max(1, counts.sum())).to(device)
            else:
                alpha = 0.1
                current = (counts.float() / max(1, counts.sum())).to(device)
                self._scdt_nominal_ema = (1 - alpha) * self._scdt_nominal_ema + alpha * current
            pi_nom = self._scdt_nominal_ema / self._scdt_nominal_ema.sum().clamp(min=1e-12)
        else:
            # 'buffer' (default): use the buffer composition
            pi_nom = counts.float() / max(1, counts.sum())

        # Preference (P-SCDT) scores u (class-level). Sources: EMA or pool logits.
        u = torch.zeros_like(pi_nom, dtype=torch.float, device=device)
        # (a) class CE EMA from training (GLOBAL basis) if available
        if self._scdt_class_loss_ma is not None and self._scdt_class_loss_ma.numel() == self.num_classes:
            u = self._scdt_class_loss_ma.index_select(0, classes)
        # (b) else: derive class means from stored logits if available
        elif hasattr(self.buffer, 'logits') and self.buffer.logits is not None:
            pool_logits = self.buffer.logits
            ce_all = F.cross_entropy(pool_logits, Y_pool, reduction='none')
            for k in range(C):
                idxs = per_class_idxs[k]
                if idxs.numel() > 0:
                    u[k] = ce_all.index_select(0, idxs).mean()
        # scale/center (optional): center to stabilize α search
        if float(u.var().item()) > 0:
            u = u - u.mean()
        eta_cfg = float(getattr(self.args, 'scdt_pref_eta', 0.0))
        if eta_cfg != 0.0:
            u = u * eta_cfg

        # (sample-level utilities for within-class selection, if logits available)
        per_sample_util = None
        if hasattr(self.buffer, 'logits') and self.buffer.logits is not None:
            per_sample_util = F.cross_entropy(self.buffer.logits, Y_pool, reduction='none')

        # Divergence type & budget
        div = getattr(self.args, 'scdt_divergence', 'FAIRNESS').upper()
        eps = float(getattr(self.args, 'scdt_budget', 0.05))

        # Availability caps per class
        upper_cap = counts.clone().long()

        # Compute real-valued optimum p*, then integerize to q (sum=n) with caps and audit
        audit = {}
        W = int(getattr(self.args, 'scdt_window', 0))
        if W > 0 and self._scdt_hist_window.maxlen != W:
            self._scdt_hist_window = deque(maxlen=W)  # GLOBAL basis

        # ---- Prepare GLOBAL pi_nom for window divergence checks ----
        # shape [num_classes], mass only on `classes`
        pi_nom_global_cpu = torch.zeros(self.num_classes, dtype=torch.float)
        pi_nom_global_cpu.index_copy_(0, classes_cpu, pi_nom.detach().cpu())

        if div == 'FAIRNESS':
            # Legacy heuristic: per-class probability band ±eps around π_nom
            q_lower = torch.clamp(((pi_nom - eps).clamp(min=0.0) * n).floor().long(), min=0)
            q_upper = torch.clamp(((pi_nom + eps).clamp(max=1.0) * n).ceil().long(), max=upper_cap)
            q_lower = torch.min(q_lower, upper_cap)
            q_upper = torch.max(q_upper, q_lower)
            q = q_lower.clone()
            R = int(n - int(q.sum().item()))
            cap = (q_upper - q)
            order = torch.argsort(u, descending=False)  # legacy: fill low-u first
            if R > 0 and int(cap.sum().item()) > 0:
                for idx in order.tolist():
                    if R <= 0:
                        break
                    add = int(min(int(cap[idx].item()), R))
                    if add > 0:
                        q[idx] += add
                        R -= add
            if R != 0:
                # fallback to uniform
                return self._legacy_uniform_replay(n, include_logits=include_logits)
            rounding_l1_gap = 0.0
            p_bar = q.float() / float(max(1, int(q.sum().item())))
            audit.update({
                'scdt_div': 'FAIRNESS_BAND',
                'scdt_budget': eps,
                'scdt_div_batch': self._tv_distance(p_bar, pi_nom)
            })

        else:
            # Exact solvers (P-SCDT)
            if div == 'TV':
                p_star = self._pscdt_tv_optimal(pi_nom, u, eps)
            elif div == 'KL':
                p_star = self._pscdt_kl_optimal(pi_nom, u, eps)
            else:
                # Unknown divergence type → uniform fallback
                return self._legacy_uniform_replay(n, include_logits=include_logits)

            # ----- Window constraint via λ-blend toward pi_nom (fixed) -----
            def window_div_if_take(q_counts: torch.LongTensor) -> float:
                """
                Compute divergence between the window-average (GLOBAL basis)
                that includes the candidate batch (q_counts over `classes`) and
                the GLOBAL nominal distribution.
                """
                # q_counts is present-class basis -> map to GLOBAL
                p_curr_global = torch.zeros(self.num_classes, dtype=torch.float)
                if int(q_counts.sum().item()) > 0:
                    p_curr_global.index_copy_(
                        0, classes_cpu,
                        (q_counts.float() / float(int(q_counts.sum().item()))).detach().cpu()
                    )
                if W <= 0 or len(self._scdt_hist_window) == 0:
                    win = p_curr_global
                else:
                    hist_sum = None
                    for h in self._scdt_hist_window:
                        hist_sum = h if hist_sum is None else (hist_sum + h)
                    win = (hist_sum + p_curr_global) / float(len(self._scdt_hist_window) + 1)

                # Divergence on GLOBAL basis
                if div == 'TV':
                    return float(0.5 * (win - pi_nom_global_cpu).abs().sum().item())
                else:
                    w = win.clamp(min=1e-12)
                    q0 = pi_nom_global_cpu.clamp(min=1e-12)
                    return float((w * (w / q0).log()).sum().item())

            # Decide if we need window enforcement
            window_enforce = (W > 1 and len(self._scdt_hist_window) > 0)

            if not window_enforce:
                # No window constraint active → use the optimum p_star directly (bug fix)
                p_use = p_star
                q, rounding_l1_gap = self._integerize_with_caps(p_use, n, upper_cap, u)
                q = self._post_audit_transfer(q, pi_nom, eps, upper_cap, u, div=('TV' if div == 'TV' else 'KL'))
            else:
                # Bisection over λ ∈ [0,1]; feasible -> raise lower bound (bug fix)
                lam_lo, lam_hi = 0.0, 1.0
                best_q, best_gap = None, None
                for _ in range(16):
                    lam = 0.5 * (lam_lo + lam_hi)
                    p_use = self._blend(pi_nom, p_star, lam)
                    q_try, gap_try = self._integerize_with_caps(p_use, n, upper_cap, u)
                    q_try = self._post_audit_transfer(q_try, pi_nom, eps, upper_cap, u, div=('TV' if div == 'TV' else 'KL'))
                    win_D = window_div_if_take(q_try)
                    if win_D <= eps + 1e-12:
                        best_q, best_gap = q_try, gap_try
                        lam_lo = lam  # push toward p_star
                    else:
                        lam_hi = lam
                if best_q is None:
                    # Conservative fallback: λ=0 (pi_nom) to minimize window divergence
                    p_fallback = pi_nom
                    best_q, best_gap = self._integerize_with_caps(p_fallback, n, upper_cap, u)
                    best_q = self._post_audit_transfer(best_q, pi_nom, eps, upper_cap, u, div=('TV' if div == 'TV' else 'KL'))
                q, rounding_l1_gap = best_q, float(best_gap)

            p_bar = q.float() / float(max(1, int(q.sum().item())))
            audit.update({
                'scdt_div': div,
                'scdt_budget': eps,
                'scdt_div_batch': (self._tv_distance(p_bar, pi_nom) if div == 'TV'
                                   else self._kl_divergence(p_bar, pi_nom)),
                'scdt_rounding_L1_gap': float(rounding_l1_gap)
            })

        # Sample within each class according to q (without replacement)
        kept_list = []
        sample_eta = float(getattr(self.args, 'scdt_sample_eta', 0.0))
        for k in range(C):
            need = int(q[k].item())
            if need <= 0:
                continue
            pool_k = per_class_idxs[k]
            if need >= int(pool_k.numel()):
                kept_list.append(pool_k)
            else:
                if per_sample_util is None or pool_k.numel() == 0:
                    perm_k = torch.randperm(int(pool_k.numel()), device=device)[:need]
                    kept_list.append(pool_k.index_select(0, perm_k))
                else:
                    util_k = per_sample_util.index_select(0, pool_k)
                    if sample_eta > 0.0:
                        # probabilistic without replacement with weights ∝ exp(η u_i)
                        w = torch.exp(sample_eta * (util_k - util_k.mean()))
                        w = (w / w.sum().clamp(min=1e-12)).clamp(min=1e-12)
                        sel_local = torch.multinomial(w, num_samples=need, replacement=False)
                        kept_list.append(pool_k.index_select(0, sel_local))
                    else:
                        # top-k by utility
                        order = torch.argsort(util_k, descending=True)[:need]
                        kept_list.append(pool_k.index_select(0, order))

        if len(kept_list) == 0:
            return self._legacy_uniform_replay(n, include_logits=include_logits)
        sel = torch.cat(kept_list, dim=0)

        # Materialize batch
        not_aug = X_pool.index_select(0, sel)
        # Use the dataset-provided transform as-is (no auto ToPILImage)
        buf_inputs = self.apply_transform(not_aug, self.transform, device=self.device, add_pil_transforms=False)
        buf_labels = Y_pool.index_select(0, sel)
        buf_logits = None
        if include_logits and hasattr(self, 'buffer') and hasattr(self.buffer, 'logits') and self.buffer.logits is not None:
            buf_logits = self.buffer.logits.index_select(0, sel)

        # ---- Audits & window bookkeeping ----
        # Present-class batch histogram (for per-batch audit)
        kept_counts_present = torch.zeros_like(counts, dtype=torch.float, device=device)
        sel_labels = Y_pool.index_select(0, sel)
        for k in range(C):
            kept_counts_present[k] = (sel_labels == classes[k]).sum()
        pi_hat_present = kept_counts_present / max(1.0, float(kept_counts_present.sum().item()))

        # ---- GLOBAL-BASIS FIX: store window histograms in GLOBAL basis ----
        # Build GLOBAL histogram for selected labels
        kept_counts_global_cpu = torch.bincount(sel_labels.cpu(), minlength=self.num_classes).float()
        if kept_counts_global_cpu.sum().item() > 0:
            pi_hat_global_cpu = kept_counts_global_cpu / kept_counts_global_cpu.sum().item()
        else:
            pi_hat_global_cpu = kept_counts_global_cpu  # all zeros; shouldn't happen

        if self._scdt_hist_window.maxlen > 0:
            self._scdt_hist_window.append(pi_hat_global_cpu)
            # compute window divergence (GLOBAL basis)
            hist_sum = None
            for h in self._scdt_hist_window:
                hist_sum = h if hist_sum is None else (hist_sum + h)
            win = hist_sum / float(len(self._scdt_hist_window))
            if audit.get('scdt_div') == 'TV':
                audit['scdt_div_window'] = float(0.5 * (win - pi_nom_global_cpu).abs().sum().item())
            elif audit.get('scdt_div') == 'KL':
                w = win.clamp(min=1e-12)
                q0 = pi_nom_global_cpu.clamp(min=1e-12)
                audit['scdt_div_window'] = float((w * (w / q0).log()).sum().item())

        # Correlation between class deviations and u (detectability proxy; present-class basis)
        if 'scdt_div' in audit:
            dev = (pi_hat_present - pi_nom)
            if float(u.var().item()) > 0:
                # torch.corrcoef requires at least 2 dims
                M = torch.stack([dev - dev.mean(), u - u.mean()])
                with suppress(Exception):
                    r = float(torch.corrcoef(M)[0, 1].item())
                    audit['scdt_u_corr'] = r

        # (removed) Sampler should not report KD mass deviation here.

        # Expose for training loop logging
        self._scdt_last_audit = audit

        return buf_inputs, buf_labels, buf_logits

    @torch.no_grad()
    def _legacy_uniform_replay(self, n: int, include_logits: bool = False):
        """
        Backward-compatible fallback: use buffer.get_data(...) if available,
        otherwise uniform random without replacement from buffer tensors.
        """
        if hasattr(self, 'buffer') and hasattr(self.buffer, 'get_data'):
            out = self.buffer.get_data(n, transform=self.transform)
            if isinstance(out, (list, tuple)):
                if len(out) == 3:
                    return out[0], out[1], out[2]
                elif len(out) == 2:
                    return out[0], out[1], None

        # Direct uniform sampling
        X_pool = self.buffer.examples
        Y_pool = self.buffer.labels
        N_pool = int(Y_pool.shape[0])
        n = int(min(n, N_pool))
        device = Y_pool.device if isinstance(Y_pool, torch.Tensor) else self.device
        sel = torch.randperm(N_pool, device=device)[:n]
        not_aug = X_pool.index_select(0, sel)
        # *** Keep transform usage consistent (no extra ToPILImage)
        buf_inputs = self.apply_transform(
            not_aug, self.transform, device=self.device, add_pil_transforms=False
        )
        buf_labels = Y_pool.index_select(0, sel)
        buf_logits = None
        if include_logits and hasattr(self, 'buffer') and hasattr(self.buffer, 'logits') and self.buffer.logits is not None:
            buf_logits = self.buffer.logits.index_select(0, sel)
        # clear last audit in legacy path
        self._scdt_last_audit = {}
        return buf_inputs, buf_labels, buf_logits

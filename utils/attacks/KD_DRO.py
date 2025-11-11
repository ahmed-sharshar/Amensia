

import os
import csv
from argparse import Namespace
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn.functional as F


__all__ = [
    "_is_aux_trim_enabled",
    "_maybe_trim_aux",
    "_apply_lr_scale_to_optimizer",
    "_restore_lr_of_optimizer",
    "_aux_debug_log",
]


# ------------------------
# small helpers (local)
# ------------------------

def _safe_var(x: torch.Tensor) -> torch.Tensor:
    if x is None or x.numel() < 2:
        return x.new_tensor(0.0) if x is not None else torch.tensor(0.0)
    return x.float().var(correction=0)

def _has_variance(x: torch.Tensor, eps: float = 1e-16) -> bool:
    return x is not None and x.numel() >= 2 and float(_safe_var(x).item()) > eps

def _safe_pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    if a is None or b is None:
        return 0.0
    if a.numel() < 2 or b.numel() < 2:
        return 0.0
    a = a.float(); b = b.float()
    a_c = a - a.mean(); b_c = b - b.mean()
    denom = (a_c.pow(2).mean().sqrt() * b_c.pow(2).mean().sqrt())
    if float(denom.item()) <= 0.0:
        return 0.0
    num = (a_c * b_c).mean()
    return float((num / denom).item())

def _kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.clamp(min=1e-12); q = q.clamp(min=1e-12)
    return float((p * (p / q).log()).sum().item())

def _tv_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    return float(0.5 * (p - q).abs().sum().item())

def _largest_remainder_rounding(t: torch.Tensor, m: int) -> torch.LongTensor:
    q = torch.floor(t).long()
    R = int(m - int(q.sum().item()))
    if R <= 0:
        return q
    frac = (t - q.float())
    order = torch.argsort(frac, descending=True)
    if R > 0:
        q[order[:R]] += 1
    return q

def _integerize_with_caps(p_star: torch.Tensor,
                          m: int,
                          upper_cap: torch.LongTensor,
                          u: torch.Tensor) -> Tuple[torch.LongTensor, float]:
    device = p_star.device
    t = (p_star * float(m))
    q = _largest_remainder_rounding(t, m).to(device)
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

def _post_audit_transfer(q: torch.LongTensor,
                         pi_nom: torch.Tensor,
                         delta: float,
                         upper_cap: torch.LongTensor,
                         u: torch.Tensor,
                         div: str) -> torch.LongTensor:
    m = int(q.sum().item())
    if m <= 0:
        return q
    for _ in range(4 * m + 10):
        p_bar = q.float() / float(m)
        D = _tv_distance(p_bar, pi_nom) if div == 'TV' else _kl_divergence(p_bar, pi_nom)
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

def _pscdt_kl_optimal(pi_nom: torch.Tensor,
                      u: torch.Tensor,
                      delta: float) -> Tuple[torch.Tensor, float]:
    C = int(pi_nom.numel())
    if delta <= 1e-12 or C == 0:
        return pi_nom.clone(), 0.0

    smoothing_eps = 0.0
    if torch.any(pi_nom <= 0):
        eps = 1e-6
        pi_nom = (1 - eps) * pi_nom + eps / float(max(1, C))
        smoothing_eps = eps

    if not _has_variance(u):
        return pi_nom.clone(), smoothing_eps

    def _tilt(alpha: float) -> torch.Tensor:
        z = torch.exp(alpha * u)
        p = (pi_nom * z)
        p = p / p.sum().clamp(min=1e-12)
        return p

    # Find a high alpha with KL >= delta
    hi = 1.0
    p_hi = _tilt(hi)
    kl_hi = _kl_divergence(p_hi, pi_nom)
    it = 0
    while kl_hi < delta and it < 40:
        hi *= 2.0
        p_hi = _tilt(hi)
        kl_hi = _kl_divergence(p_hi, pi_nom)
        it += 1
        if hi > 1e6:
            break
    # Bisection on [lo, hi], always keeping the last feasible (<=delta)
    lo = 0.0
    p_lo = _tilt(lo)
    kl_lo = _kl_divergence(p_lo, pi_nom)  # = 0 at lo=0
    p_best = p_lo  # guaranteed feasible
    if kl_hi < delta:
        # Could not reach the boundary; return the best we have (feasible side)
        return p_hi, smoothing_eps
    for _ in range(35):
        mid = 0.5 * (lo + hi)
        pm = _tilt(mid)
        km = _kl_divergence(pm, pi_nom)
        if km <= delta:
            lo = mid
            p_best = pm  # keep feasible candidate
        else:
            hi = mid
    return p_best, smoothing_eps

def _pscdt_tv_optimal(pi_nom: torch.Tensor,
                      u: torch.Tensor,
                      delta: float) -> torch.Tensor:
    C = int(pi_nom.numel())
    if delta <= 1e-12 or C == 0:
        return pi_nom.clone()

    p = pi_nom.clone()
    order_desc = torch.argsort(u, descending=True)
    order_asc = torch.flip(order_desc, dims=[0])

    i = 0; j = 0; moved = 0.0
    while moved < delta - 1e-12 and i < C and j < C:
        r = order_desc[i].item()
        d = order_asc[j].item()
        if r == d:
            j += 1; continue
        inc_cap = float(1.0 - p[r].item())
        dec_cap = float(p[d].item())
        if inc_cap <= 1e-12:
            i += 1; continue
        if dec_cap <= 1e-12:
            j += 1; continue
        mv = min(delta - moved, inc_cap, dec_cap)
        if mv <= 0:
            break
        p[r] += mv; p[d] -= mv; moved += mv
        if p[r] >= 1.0 - 1e-12: i += 1
        if p[d] <= 1e-12: j += 1
    p = p.clamp(min=0.0)
    p = p / p.sum().clamp(min=1e-12)
    return p

def _select_indices_within_class(pool_idx: torch.Tensor,
                                 per_sample_util: Optional[torch.Tensor],
                                 need: int,
                                 device: torch.device) -> torch.Tensor:
    if need <= 0 or pool_idx.numel() == 0:
        return torch.empty(0, dtype=pool_idx.dtype, device=device)
    if need >= int(pool_idx.numel()):
        return pool_idx
    if per_sample_util is None:
        perm = torch.randperm(int(pool_idx.numel()), device=device)[:need]
        return pool_idx.index_select(0, perm)
    order = torch.argsort(per_sample_util, descending=True)[:need]
    return pool_idx.index_select(0, order)

def _select_indices_with_temp(pool_idx: torch.Tensor,
                              util_local: Optional[torch.Tensor],
                              need: int,
                              device: torch.device,
                              sample_eta: float) -> torch.Tensor:
    """
    Within-class selection with optional probabilistic weights:
      - if util unavailable -> random
      - if sample_eta>0 -> weighted without replacement, weights ∝ exp(eta*(u_i - mean(u)))
      - else -> top-k by util
    """
    if need <= 0 or pool_idx.numel() == 0:
        return torch.empty(0, dtype=pool_idx.dtype, device=device)
    if need >= int(pool_idx.numel()):
        return pool_idx
    if util_local is None:
        perm = torch.randperm(int(pool_idx.numel()), device=device)[:need]
        return pool_idx.index_select(0, perm)
    if sample_eta > 0.0:
        w = torch.exp(sample_eta * (util_local - util_local.mean()))
        w = (w / w.sum().clamp(min=1e-12)).clamp(min=1e-12)
        sel_local = torch.multinomial(w, num_samples=need, replacement=False)
        return pool_idx.index_select(0, sel_local)
    # top-k fallback
    order = torch.argsort(util_local, descending=True)[:need]
    return pool_idx.index_select(0, order)

def _compute_delta_t(eps: float, ring: list, W: int) -> float:
    """
    Online tightened budget δ_t per the scheduler:
      δ_t = max{0, min_{1 ≤ L ≤ min(W-1, len(ring))} [ L*eps - sum_{last L} D_j ] }
    """
    if W <= 1 or len(ring) == 0:
        return eps
    L_max = min(W - 1, len(ring))
    best = eps
    for L in range(1, L_max + 1):
        sL = float(sum(ring[-L:]))
        val = max(0.0, L * eps - sL)
        if val < best:
            best = val
    return best


# ------------------------
# Public API (unchanged)
# ------------------------

def _is_aux_trim_enabled(args: Namespace) -> bool:
    if int(getattr(args, 'aux_trim', 0)) == 1:
        return True
    f = getattr(args, 'aux_keep_frac', None)
    try:
        return f is not None and 0.0 <= float(f) < 1.0
    except Exception:
        return False

def _compute_aux_target(n_curr: int, n_aux_avail: int, target_frac: float) -> int:
    if n_curr < 0 or n_aux_avail <= 0 or target_frac <= 0.0:
        return 0
    denom = max(1e-8, (1.0 - target_frac))
    target = int((target_frac / denom) * n_curr)
    return max(0, min(n_aux_avail, target))

def _apply_lr_scale_to_optimizer(opt, scale: float) -> Optional[list]:
    if opt is None or abs(scale - 1.0) < 1e-8:
        return None
    old_lrs = [pg['lr'] for pg in opt.param_groups]
    for pg in opt.param_groups:
        pg['lr'] = pg['lr'] * scale
    return old_lrs

def _restore_lr_of_optimizer(opt, old_lrs: Optional[list]) -> None:
    if opt is None or old_lrs is None:
        return
    for pg, lr in zip(opt.param_groups, old_lrs):
        pg['lr'] = lr

def _aux_debug_log(args: Namespace, row: Dict[str, Any]) -> None:
    path = getattr(args, 'aux_debug_log', '')
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    file_exists = os.path.isfile(path)
    with open(path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _maybe_trim_aux(inputs: torch.Tensor,
                    labels: torch.Tensor,
                    not_aug_inputs: torch.Tensor,
                    current_task_labels: list,
                    args: Namespace,
                    logits: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    meta = {
        'lr_scale': 1.0,
        'n_before': int(labels.shape[0]),
        'n_after': int(labels.shape[0]),
        'n_curr': 0,
        'n_aux': 0,
        'n_aux_kept': 0,
        'realized_aux_frac': 0.0  # (legacy) AUX share in final kept batch
    }

    # If trimming is disabled or no task label info: not applicable
    if not _is_aux_trim_enabled(args) or len(current_task_labels) == 0:
        if labels.numel() > 0:
            meta['realized_aux_frac'] = 0.0
        # Explicitly mark as "not applicable" for AUX budget accounting
        meta['scdt_aux_defined'] = 0
        return inputs, labels, not_aug_inputs, logits, meta

    device = labels.device
    curr_set = torch.tensor(current_task_labels, device=device, dtype=labels.dtype)
    try:
        curr_mask = torch.isin(labels, curr_set)
    except AttributeError:
        curr_mask = torch.zeros_like(labels, dtype=torch.bool)
        for v in current_task_labels:
            curr_mask |= (labels == v)

    aux_mask = ~curr_mask
    idx_curr = torch.nonzero(curr_mask, as_tuple=False).squeeze(1)
    idx_aux = torch.nonzero(aux_mask, as_tuple=False).squeeze(1)
    n_curr = int(idx_curr.numel()); n_aux = int(idx_aux.numel())
    meta['n_curr'] = n_curr; meta['n_aux'] = n_aux

    # If there are no AUX samples, the AUX constraint is not applicable.
    if n_aux == 0 or n_curr + n_aux == 0:
        meta['realized_aux_frac'] = 0.0
        # Expose targets for visibility but do NOT mark a violation or error
        f0 = float(getattr(args, 'aux_keep_frac', 0.5))
        meta['scdt_aux_target_frac'] = float(f0)
        meta['scdt_aux_target_mode'] = getattr(args, 'scdt_mass_mode', 'relative_to_aux')
        meta['scdt_aux_realized_frac_total'] = 0.0
        meta['scdt_aux_realized_frac_aux'] = 0.0
        # Mark as "not applicable"
        meta['scdt_aux_defined'] = 0
        return inputs, labels, not_aug_inputs, logits, meta

    # Target fraction and parameterization mode
    f = float(getattr(args, 'aux_keep_frac', 0.5))
    f = max(0.0, min(0.999, f))
    mass_mode = getattr(args, 'scdt_mass_mode', 'relative_to_aux')
    mode_str = 'relative_to_aux' if mass_mode == 'relative_to_aux' else 'relative_to_total'

    if mass_mode == 'relative_to_aux':
        n_aux_target = max(0, min(n_aux, int(round(f * n_aux))))
    else:
        # Choose n_aux_target so that AUX share in final batch ≈ f
        n_aux_target = _compute_aux_target(n_curr=n_curr, n_aux_avail=n_aux, target_frac=f)

    # Intended final-batch AUX fraction (for audits): n_aux_target / (n_curr + n_aux_target)
    denom_final = float(n_curr + n_aux_target) if (n_curr + n_aux_target) > 0 else 1.0
    mass_target_frac_final = float(n_aux_target) / denom_final

    if n_curr == 0 and n_aux_target == 0:
        n_aux_target = min(n_aux, 1)

    if n_aux_target >= n_aux:
        kept_idx = torch.cat([idx_curr, idx_aux], dim=0)
        q = None  # used only for FAIRNESS audit below if needed
        pi_nom = None
    else:
        scdt_on = int(getattr(args, 'scdt', 0)) == 1
        sample_eta = float(getattr(args, 'scdt_sample_eta', 0.0))

        if not scdt_on:
            perm = torch.randperm(n_aux, device=device)
            idx_aux_keep = idx_aux[perm[:n_aux_target]]
            kept_idx = torch.cat([idx_curr, idx_aux_keep], dim=0)
            q = None; pi_nom = None
        else:
            # 1) AUX class availability and precomputed masks (reused)
            aux_labels = labels.index_select(0, idx_aux)  # [n_aux]
            classes, counts = aux_labels.unique(return_counts=True)
            C = int(classes.numel())

            # Precompute masks for each present class once
            local_masks = [(aux_labels == classes[k]) for k in range(C)]
            per_class_indices = [
                idx_aux[torch.nonzero(local_masks[k], as_tuple=False).squeeze(1)]
                for k in range(C)
            ]

            # 2) Nominal π_nom
            nominal_mode = getattr(args, 'scdt_nominal', 'buffer')
            if nominal_mode == 'uniform':
                pi_nom = torch.full((C,), 1.0 / max(1, C), dtype=torch.float, device=device)
            elif nominal_mode == 'ema':
                ema_dict: Dict[int, float] = getattr(args, '_scdt_pi0_ema', None) or {}
                vec = torch.empty(C, dtype=torch.float, device=device)
                for k in range(C):
                    cls_id = int(classes[k].item())
                    val = ema_dict.get(cls_id, float(counts[k].item()) / max(1, n_aux))
                    vec[k] = val
                pi_nom = vec / vec.sum().clamp(min=1e-12)
                setattr(args, '_scdt_pi0_ema', ema_dict)
            else:
                pi_nom = counts.float() / max(1, n_aux)

            # 3) Preference u (class-level), compute CE once (if logits available)
            #    *** MODIFIED: add class-utility EMA so preference remains active when logits are absent ***
            u = torch.zeros(C, dtype=torch.float, device=device)
            ce_aux = None

            # Utility EMA state
            util_ema_dict: Dict[int, float] = getattr(args, '_scdt_util_ema', None) or {}
            util_mom = float(getattr(args, 'scdt_util_mom', 0.9))
            util_mom = max(0.0, min(0.9999, util_mom))  # clamp to [0, 1)

            if logits is not None:
                aux_logits = logits.index_select(0, idx_aux)
                ce_aux = F.cross_entropy(aux_logits, aux_labels, reduction='none')  # [n_aux]
                # class means via precomputed masks (no repeated equality)
                u_raw = torch.zeros(C, dtype=torch.float, device=device)
                for k in range(C):
                    if local_masks[k].any():
                        u_raw[k] = ce_aux[local_masks[k]].mean()
                        # Update class-utility EMA with raw (unscaled) utility
                        cls_id = int(classes[k].item())
                        old = util_ema_dict.get(cls_id, float(u_raw[k].item()))
                        util_ema_dict[cls_id] = util_mom * old + (1.0 - util_mom) * float(u_raw[k].item())
                # Use the raw utilities for current step preference (possibly scaled below)
                u = u_raw
                # Persist EMA
                setattr(args, '_scdt_util_ema', util_ema_dict)
            else:
                # No logits: pull utilities from EMA (if missing, default to 0.0)
                u_from_ema = torch.zeros(C, dtype=torch.float, device=device)
                for k in range(C):
                    cls_id = int(classes[k].item())
                    u_from_ema[k] = float(util_ema_dict.get(cls_id, 0.0))
                u = u_from_ema

            eta_cfg = float(getattr(args, 'scdt_pref_eta', 0.0))
            if eta_cfg != 0.0:
                u = u * eta_cfg

            # 4) Divergence/budget/caps
            div = getattr(args, 'scdt_divergence', 'KL').upper()
            eps = float(getattr(args, 'scdt_budget', 0.05))
            upper_cap = counts.clone().long()

            # Window state
            W = int(getattr(args, 'scdt_window', 0))
            hist_window: list = getattr(args, '_scdt_aux_hist_window', None)
            if hist_window is None:
                hist_window = []
                setattr(args, '_scdt_aux_hist_window', hist_window)

            # Fixed nominal over the window (dictionary of class_id -> prob)
            pi0_ref_dict: Optional[Dict[int, float]] = getattr(args, '_scdt_pi0_window_ref', None)

            def _ensure_window_p0_ref() -> Dict[int, float]:
                nonlocal pi0_ref_dict
                if W <= 0:
                    return {int(classes[k].item()): float(pi_nom[k].item()) for k in range(C)}
                if pi0_ref_dict is None or len(hist_window) == 0:
                    # Initialize with current nominal over present classes
                    pi0_ref_dict = {int(classes[k].item()): float(pi_nom[k].item()) for k in range(C)}
                    setattr(args, '_scdt_pi0_window_ref', pi0_ref_dict)
                return pi0_ref_dict

            def _window_divergence_with_candidate(q_cand: torch.LongTensor) -> float:
                """
                Compute divergence over the UNION of classes seen in the window and current step,
                comparing the aggregated window histogram (including q_cand) to the fixed p0_ref.
                """
                if W <= 0:
                    return 0.0
                p0_ref = _ensure_window_p0_ref()
                # Aggregate past counts into a dict
                agg_dict: Dict[int, int] = {}
                for step_counts in hist_window:
                    for cid, cnt in step_counts.items():
                        agg_dict[cid] = agg_dict.get(cid, 0) + int(cnt)
                # Add current candidate quotas
                for k in range(C):
                    need = int(q_cand[k].item())
                    if need <= 0:
                        continue
                    cid = int(classes[k].item())
                    agg_dict[cid] = agg_dict.get(cid, 0) + need
                total = sum(agg_dict.values())
                if total <= 0:
                    return 0.0
                union_keys = set(agg_dict.keys()) | set(p0_ref.keys())
                keys = sorted(list(union_keys))
                p_win = torch.tensor([agg_dict.get(cid, 0) / total for cid in keys], dtype=torch.float, device=device)
                p0_vec = torch.tensor([p0_ref.get(cid, 0.0) for cid in keys], dtype=torch.float, device=device)
                s0 = float(p0_vec.sum().item())
                if s0 <= 0.0:
                    p0_vec = torch.full_like(p_win, 1.0 / max(1, len(keys)))
                else:
                    p0_vec = p0_vec / max(1e-12, s0)
                return _tv_distance(p_win, p0_vec) if div == 'TV' else _kl_divergence(p_win, p0_vec)

            # Divergence ring for δ_t scheduler (store per-batch divergences w.r.t fixed p0_ref)
            div_ring: list = getattr(args, '_scdt_div_ring', None)
            if div_ring is None:
                div_ring = []
                setattr(args, '_scdt_div_ring', div_ring)

            # ---------------------------
            # Branch: FAIRNESS (legacy)
            # ---------------------------
            if div == 'FAIRNESS':
                q_lower = torch.clamp(((pi_nom - eps).clamp(min=0.0) * n_aux_target).floor().long(), min=0)
                q_upper = torch.clamp(((pi_nom + eps).clamp(max=1.0) * n_aux_target).ceil().long(), max=upper_cap)
                q_lower = torch.min(q_lower, upper_cap)
                q_upper = torch.max(q_upper, q_lower)

                q = q_lower.clone()
                R = int(n_aux_target - int(q.sum().item()))
                cap = (q_upper - q)
                order = torch.argsort(u, descending=False)
                if R > 0 and int(cap.sum().item()) > 0:
                    for idx in order.tolist():
                        if R <= 0:
                            break
                        add = int(min(int(cap[idx].item()), R))
                        if add > 0:
                            q[idx] += add
                            R -= add
                # selection (either constrained or random fallback)
                kept_aux_list = []
                if R != 0:
                    perm = torch.randperm(n_aux, device=device)
                    idx_aux_keep = idx_aux[perm[:n_aux_target]]
                else:
                    per_sample_util = ce_aux  # may be None
                    for k in range(C):
                        need = int(q[k].item())
                        pool_k = per_class_indices[k]
                        if need <= 0:
                            continue
                        util_local = (per_sample_util[local_masks[k]] if per_sample_util is not None else None)
                        sel = _select_indices_with_temp(pool_k, util_local, need, device, sample_eta)
                        kept_aux_list.append(sel)
                    idx_aux_keep = torch.cat(kept_aux_list, dim=0) if len(kept_aux_list) > 0 else torch.empty(0, dtype=idx_aux.dtype, device=device)
                kept_idx = torch.cat([idx_curr, idx_aux_keep], dim=0)

                # FAIRNESS audit using realized kept composition
                try:
                    kept_labels = labels.index_select(0, idx_aux_keep)
                    kept_counts = []
                    for k in range(C):
                        kept_counts.append(int((kept_labels == classes[k]).sum().item()))
                    kept_counts = torch.tensor(kept_counts, dtype=torch.float, device=device)
                    p_bar = kept_counts / max(1.0, float(kept_counts.sum().item()))
                    # Minimal audit + utility gain (centered)
                    u_center = (u - u.mean()).float() if _has_variance(u) else torch.zeros_like(u, dtype=torch.float)
                    delta_p = (p_bar - pi_nom).float()
                    meta.update({
                        'scdt_div': 'FAIRNESS_BAND',
                        'scdt_budget': eps,
                        'scdt_div_batch': _tv_distance(p_bar, pi_nom),
                        'scdt_u_gain': float((delta_p * u_center).sum().item())
                    })
                except Exception:
                    pass  # if something goes wrong, skip audit for fairness

            # ---------------------------
            # Branch: TV (exact) + window (with δ_t scheduler)
            # ---------------------------
            elif div == 'TV':
                # Compute δ_t for this step using the divergence ring
                eps_eff = _compute_delta_t(eps, div_ring, W)

                # Build fixed p0_ref and present-class p0 vector (normalized on present classes)
                p0_ref = _ensure_window_p0_ref()
                p0_present = torch.tensor(
                    [p0_ref.get(int(classes[k].item()), 0.0) for k in range(C)],
                    dtype=torch.float, device=device
                )
                s0p = float(p0_present.sum().item())
                if s0p <= 0.0:
                    p0_present = torch.full((C,), 1.0 / max(1, C), dtype=torch.float, device=device)
                else:
                    p0_present = p0_present / max(1e-12, s0p)

                p_star = _pscdt_tv_optimal(p0_present, u, eps_eff)

                # Helper: per-batch divergence vs p0_ref over UNION keys for a candidate q
                def _per_batch_div_vs_p0_ref(q_cand: torch.LongTensor) -> float:
                    batch_counts = {int(classes[k].item()): int(q_cand[k].item()) for k in range(C)}
                    union_keys = set(batch_counts.keys()) | set(p0_ref.keys())
                    keys = sorted(list(union_keys))
                    total_b = float(sum(batch_counts.get(cid, 0) for cid in keys))
                    if total_b <= 0.0:
                        return 0.0
                    p_b = torch.tensor([batch_counts.get(cid, 0) / total_b for cid in keys], dtype=torch.float, device=device)
                    p0_vec = torch.tensor([p0_ref.get(cid, 0.0) for cid in keys], dtype=torch.float, device=device)
                    s0 = float(p0_vec.sum().item())
                    p0_vec = (p0_vec if s0 <= 0.0 else (p0_vec / max(1e-12, s0)))
                    return _tv_distance(p_b, p0_vec)

                window_enforce = (W > 1 and len(hist_window) > 0)
                lam_lo, lam_hi = 0.0, 1.0
                best_q, best_gap, best_win_div, best_batch_ref = None, None, None, None
                best_p_use = None  # <-- track the continuous mix used for accurate rounding-gap accounting
                tried = 0
                while tried < 24:
                    lam = 1.0 if not window_enforce and tried == 0 else 0.5 * (lam_lo + lam_hi)
                    p_use = (lam * p_star + (1.0 - lam) * p0_present).clamp(min=0.0)
                    p_use = p_use / p_use.sum().clamp(min=1e-12)

                    q_try, gap_try = _integerize_with_caps(p_use, n_aux_target, upper_cap, u)
                    q_try = _post_audit_transfer(q_try, p0_present, eps_eff, upper_cap, u, div='TV')
                    win_D = _window_divergence_with_candidate(q_try)
                    batch_ref_D = _per_batch_div_vs_p0_ref(q_try)
                    feasible = (batch_ref_D <= eps_eff + 1e-12) and ((not window_enforce) or (win_D <= eps + 1e-12))
                    if feasible:
                        best_q, best_gap, best_win_div, best_batch_ref = q_try, gap_try, win_D, batch_ref_D
                        best_p_use = p_use.clone()
                        lam_lo = lam
                    else:
                        lam_hi = lam
                    tried += 1
                    if not window_enforce and feasible:
                        break

                # Hard enforce feasibility by taking the best found (if any), else fall back to p0_present quotas
                if best_q is None:
                    q, _ = _integerize_with_caps(p0_present, n_aux_target, upper_cap, u)
                    q = _post_audit_transfer(q, p0_present, eps_eff, upper_cap, u, div='TV')
                    win_final = _window_divergence_with_candidate(q)
                    batch_ref_final = _per_batch_div_vs_p0_ref(q)
                    p_use_final = p0_present
                    # Recompute rounding gap AFTER caps & audit, against the continuous mix actually used
                    rounding_l1_gap_final = float((q.float() / float(max(1, n_aux_target)) - p_use_final).abs().sum().item())
                else:
                    q, win_final, batch_ref_final = best_q, best_win_div, best_batch_ref
                    p_use_final = best_p_use if best_p_use is not None else p0_present
                    rounding_l1_gap_final = float((q.float() / float(max(1, n_aux_target)) - p_use_final).abs().sum().item())

                kept_aux_list = []
                per_sample_util = (ce_aux if logits is not None else None)
                for k in range(C):
                    need = int(q[k].item())
                    pool_k = per_class_indices[k]
                    if need <= 0:
                        continue
                    util_local = (per_sample_util[local_masks[k]] if per_sample_util is not None else None)
                    sel = _select_indices_with_temp(pool_k, util_local, need, device, sample_eta)
                    kept_aux_list.append(sel)
                idx_aux_keep = torch.cat(kept_aux_list, dim=0) if len(kept_aux_list) > 0 else torch.empty(0, dtype=idx_aux.dtype, device=device)
                kept_idx = torch.cat([idx_curr, idx_aux_keep], dim=0)

                if W > 0:
                    step_counts = {int(classes[k].item()): int(q[k].item()) for k in range(C)}
                    hist_window.append(step_counts)
                    if len(hist_window) > W:
                        hist_window.pop(0)

                p_bar = (q.float() / float(max(1, int(q.sum().item()))))
                delta_p = (p_bar - p0_present).float()
                u_center = (u - u.mean()).float() if _has_variance(u) else torch.zeros_like(u, dtype=torch.float)

                # Update divergence ring with per-batch divergence vs fixed p0_ref
                if W > 0:
                    div_ring.append(float(batch_ref_final))
                    if len(div_ring) > max(0, W - 1):
                        div_ring.pop(0)

                meta.update({
                    'scdt_div': 'TV',
                    'scdt_budget': eps,
                    'scdt_div_batch': _tv_distance(p_bar, p0_present),
                    'scdt_div_batch_ref': float(batch_ref_final),
                    'scdt_div_window': float(win_final),
                    'scdt_window_violation': int(W > 0 and win_final > eps + 1e-12),
                    'scdt_rounding_L1_gap': rounding_l1_gap_final,
                    'scdt_rounding_L1_bound': (float(C) / max(1, n_aux_target)),
                    'scdt_u_corr': _safe_pearson_corr(delta_p, u_center),
                    # ---- NEW: per-step utility gain (Budget Efficiency / Optimality) ----
                    'scdt_u_gain': float((delta_p * u_center).sum().item())
                })

            # ---------------------------
            # Branch: KL (single-tilt) + window (with δ_t scheduler)
            # ---------------------------
            elif div == 'KL':
                # Compute δ_t for this step using the divergence ring
                eps_eff = _compute_delta_t(eps, div_ring, W)

                # Build fixed p0_ref and present-class p0 vector (normalized on present classes)
                p0_ref = _ensure_window_p0_ref()
                p0_present = torch.tensor(
                    [p0_ref.get(int(classes[k].item()), 0.0) for k in range(C)],
                    dtype=torch.float, device=device
                )
                s0p = float(p0_present.sum().item())
                if s0p <= 0.0:
                    p0_present = torch.full((C,), 1.0 / max(1, C), dtype=torch.float, device=device)
                else:
                    p0_present = p0_present / max(1e-12, s0p)

                p_star, smoothing_eps = _pscdt_kl_optimal(p0_present, u, eps_eff)

                # Helper: per-batch divergence vs p0_ref over UNION keys for a candidate q
                def _per_batch_div_vs_p0_ref(q_cand: torch.LongTensor) -> float:
                    batch_counts = {int(classes[k].item()): int(q_cand[k].item()) for k in range(C)}
                    union_keys = set(batch_counts.keys()) | set(p0_ref.keys())
                    keys = sorted(list(union_keys))
                    total_b = float(sum(batch_counts.get(cid, 0) for cid in keys))
                    if total_b <= 0.0:
                        return 0.0
                    p_b = torch.tensor([batch_counts.get(cid, 0) / total_b for cid in keys], dtype=torch.float, device=device)
                    p0_vec = torch.tensor([p0_ref.get(cid, 0.0) for cid in keys], dtype=torch.float, device=device)
                    s0 = float(p0_vec.sum().item())
                    p0_vec = (p0_vec if s0 <= 0.0 else (p0_vec / max(1e-12, s0)))
                    return _kl_divergence(p_b, p0_vec)

                window_enforce = (W > 1 and len(hist_window) > 0)
                lam_lo, lam_hi = 0.0, 1.0
                best_q, best_gap, best_win_div, best_batch_ref = None, None, None, None
                best_p_use = None  # <-- track the continuous mix used for accurate rounding-gap accounting
                tried = 0
                while tried < 24:
                    lam = 1.0 if not window_enforce and tried == 0 else 0.5 * (lam_lo + lam_hi)
                    p_use = (lam * p_star + (1.0 - lam) * p0_present).clamp(min=0.0)
                    p_use = p_use / p_use.sum().clamp(min=1e-12)

                    q_try, gap_try = _integerize_with_caps(p_use, n_aux_target, upper_cap, u)
                    q_try = _post_audit_transfer(q_try, p0_present, eps_eff, upper_cap, u, div='KL')
                    win_D = _window_divergence_with_candidate(q_try)
                    batch_ref_D = _per_batch_div_vs_p0_ref(q_try)
                    feasible = (batch_ref_D <= eps_eff + 1e-12) and ((not window_enforce) or (win_D <= eps + 1e-12))
                    if feasible:
                        best_q, best_gap, best_win_div, best_batch_ref = q_try, gap_try, win_D, batch_ref_D
                        best_p_use = p_use.clone()
                        lam_lo = lam
                    else:
                        lam_hi = lam
                    tried += 1
                    if not window_enforce and feasible:
                        break

                # Hard enforce feasibility by taking the best found (if any), else fall back to p0_present quotas
                if best_q is None:
                    q, _ = _integerize_with_caps(p0_present, n_aux_target, upper_cap, u)
                    q = _post_audit_transfer(q, p0_present, eps_eff, upper_cap, u, div='KL')
                    win_final = _window_divergence_with_candidate(q)
                    batch_ref_final = _per_batch_div_vs_p0_ref(q)
                    p_use_final = p0_present
                    # Recompute rounding gap AFTER caps & audit, against the continuous mix actually used
                    rounding_l1_gap_final = float((q.float() / float(max(1, n_aux_target)) - p_use_final).abs().sum().item())
                else:
                    q, win_final, batch_ref_final = best_q, best_win_div, best_batch_ref
                    p_use_final = best_p_use if best_p_use is not None else p0_present
                    rounding_l1_gap_final = float((q.float() / float(max(1, n_aux_target)) - p_use_final).abs().sum().item())

                kept_aux_list = []
                per_sample_util = (ce_aux if logits is not None else None)
                for k in range(C):
                    need = int(q[k].item())
                    pool_k = per_class_indices[k]
                    if need <= 0:
                        continue
                    util_local = (per_sample_util[local_masks[k]] if per_sample_util is not None else None)
                    sel = _select_indices_with_temp(pool_k, util_local, need, device, sample_eta)
                    kept_aux_list.append(sel)
                idx_aux_keep = torch.cat(kept_aux_list, dim=0) if len(kept_aux_list) > 0 else torch.empty(0, dtype=idx_aux.dtype, device=device)
                kept_idx = torch.cat([idx_curr, idx_aux_keep], dim=0)

                if W > 0:
                    step_counts = {int(classes[k].item()): int(q[k].item()) for k in range(C)}
                    hist_window.append(step_counts)
                    if len(hist_window) > W:
                        hist_window.pop(0)

                p_bar = (q.float() / float(max(1, int(q.sum().item()))))
                delta_p = (p_bar - p0_present).float()
                u_center = (u - u.mean()).float() if _has_variance(u) else torch.zeros_like(u, dtype=torch.float)

                # Update divergence ring with per-batch divergence vs fixed p0_ref
                if W > 0:
                    div_ring.append(float(batch_ref_final))
                    if len(div_ring) > max(0, W - 1):
                        div_ring.pop(0)

                meta.update({
                    'scdt_div': 'KL',
                    'scdt_budget': eps,
                    'scdt_div_batch': _kl_divergence(p_bar, p0_present),
                    'scdt_div_batch_ref': float(batch_ref_final),
                    'scdt_div_window': float(win_final),
                    'scdt_window_violation': int(W > 0 and win_final > eps + 1e-12),
                    'scdt_rounding_L1_gap': rounding_l1_gap_final,
                    'scdt_rounding_L1_bound': (float(C) / max(1, n_aux_target)),
                    'scdt_u_corr': _safe_pearson_corr(delta_p, u_center),
                    # ---- NEW: per-step utility gain (Budget Efficiency / Optimality) ----
                    'scdt_u_gain': float((delta_p * u_center).sum().item()),
                    'scdt_kl_support_smoothing': int(smoothing_eps > 0.0),
                    'scdt_smoothing_eps': float(smoothing_eps)
                })
            else:
                perm = torch.randperm(n_aux, device=device)
                idx_aux_keep = idx_aux[perm[:n_aux_target]]
                kept_idx = torch.cat([idx_curr, idx_aux_keep], dim=0)

            # Update nominal EMA (if enabled)
            if scdt_on and nominal_mode == 'ema':
                ema_dict: Dict[int, float] = getattr(args, '_scdt_pi0_ema', None) or {}
                beta = float(getattr(args, 'scdt_ema_beta', 0.1))
                for k in range(C):
                    cls_id = int(classes[k].item())
                    curr = float(counts[k].item()) / max(1, n_aux)
                    old = ema_dict.get(cls_id, curr)
                    ema_dict[cls_id] = (1.0 - beta) * old + beta * curr
                setattr(args, '_scdt_pi0_ema', ema_dict)

    kept_idx, _ = torch.sort(kept_idx)

    inputs_kept = inputs.index_select(0, kept_idx)
    labels_kept = labels.index_select(0, kept_idx)
    not_aug_kept = not_aug_inputs.index_select(0, kept_idx)
    logits_kept = logits.index_select(0, kept_idx) if logits is not None else None

    n_after = int(kept_idx.numel())
    meta['n_after'] = n_after
    meta['n_aux_kept'] = max(0, n_after - int(idx_curr.numel()))

    # --- Realized AUX fractions (both definitions) ---
    # (1) share in final kept batch
    realized_frac_total = (meta['n_aux_kept'] / n_after) if n_after > 0 else 0.0
    # (2) kept over available AUX
    realized_frac_aux = (meta['n_aux_kept'] / n_aux) if n_aux > 0 else 0.0

    meta['realized_aux_frac'] = float(realized_frac_total)  # legacy
    meta['lr_scale'] = 1.0

    # ----- KD mass auditing (counts & fractions) -----
    meta['scdt_mass_target'] = int(n_aux_target)
    meta['scdt_mass_realized'] = int(meta['n_aux_kept'])
    meta['scdt_mass_target_frac'] = float(mass_target_frac_final)
    meta['scdt_mass_realized_frac'] = float(realized_frac_total)
    meta['scdt_mass_dev'] = abs(meta['scdt_mass_realized_frac'] - meta['scdt_mass_target_frac'])

    # ----- AUX keep-fraction budget (primary metric) -----
    meta['scdt_aux_target_frac'] = float(f)
    meta['scdt_aux_target_mode'] = mode_str
    meta['scdt_aux_realized_frac_total'] = float(realized_frac_total)
    meta['scdt_aux_realized_frac_aux'] = float(realized_frac_aux)

    # Mark as applicable for AUX budget accounting on this step
    meta['scdt_aux_defined'] = 1

    if mode_str == 'relative_to_aux':
        aux_error = abs(realized_frac_aux - f)
        # Integer resolution tolerance: 1 / n_aux (half-step margin optional; we use 1/n)
        tol = 1.0 / max(1, n_aux)
    else:
        aux_error = abs(realized_frac_total - f)
        # Integer resolution tolerance: 1 / n_after (AUX share in final batch)
        tol = 1.0 / max(1, n_after)

    meta['scdt_aux_frac_error'] = float(aux_error)
    meta['scdt_aux_violation'] = int(aux_error > tol + 1e-12)

    return inputs_kept, labels_kept, not_aug_kept, logits_kept, meta

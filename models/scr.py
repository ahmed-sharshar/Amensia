
from typing import Tuple, List, Optional, Dict

import torch
import torch.nn.functional as F
from datasets import get_dataset

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer

# Optional: reuse your replay-only SCDT/AUX trimmer to keep compatibility with ER/ER-ACE
from utils.attacks.KD_DRO import _maybe_trim_aux


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via Supervised Contrastive Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--supcon_weight', type=float, default=0.2,
                        help='Weight λ for the supervised contrastive term.')
    parser.add_argument('--supcon_temp', type=float, default=0.07,
                        help='Temperature τ for the supervised contrastive loss.')
    return parser


class Scr(ContinualModel):
    NAME = 'scr'
    COMPATIBILITY = ['class-il', 'task-il', 'domain-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Scr, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        # optional debug/audit like your ER/ER-ACE
        self._scdt_last_audit: Optional[Dict] = None

        # class/task bookkeeping (like ER-ACE)
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.num_classes = get_dataset(args).N_TASKS * get_dataset(args).N_CLASSES_PER_TASK
        self.task = 0

    def end_task(self, dataset):
        self.task += 1

    # -------------------------- helpers --------------------------
    @staticmethod
    def _split_curr_aux_masks(labels: torch.Tensor, current_task_labels: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return boolean masks for current‑task vs auxiliary labels."""
        if not current_task_labels:
            curr_mask = torch.ones_like(labels, dtype=torch.bool)
            aux_mask = ~curr_mask
            return curr_mask, aux_mask
        device = labels.device
        curr_set = torch.tensor(current_task_labels, device=device, dtype=labels.dtype)
        try:
            curr_mask = torch.isin(labels, curr_set)  # torch >= 1.10
        except AttributeError:
            curr_mask = torch.zeros_like(labels, dtype=torch.bool)
            for v in current_task_labels:
                curr_mask |= (labels == v)
        aux_mask = ~curr_mask
        return curr_mask, aux_mask

    def _trim_replay_batch_if_needed(
        self,
        buf_x: Optional[torch.Tensor],
        buf_y: Optional[torch.Tensor],
        buf_na: Optional[torch.Tensor],
        current_task_labels: List[int],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Dict]:
        """
        Attack **ONLY the replay minibatch** (same contract as ER/ER-ACE):
          • keep ALL replay items whose labels are in current_task_labels;
          • apply SCDT/AUX trimming to replay items not in current_task_labels;
          • return concatenated (current_replay ⊕ trimmed_aux_replay).
        """
        meta: Dict = {}
        if buf_x is None or buf_y is None or buf_y.numel() == 0:
            return buf_x, buf_y, buf_na, meta

        # --- EDIT 1: Fix the logic gate ---
        # Run if EITHER is enabled. Return only if BOTH are disabled.
        if int(getattr(self.args, 'aux_trim', 0)) == 0 and int(getattr(self.args, 'scdt', 0)) == 0:
            return buf_x, buf_y, buf_na, meta
        # ---------------------------------

        # masks inside REPLAY
        is_curr, is_aux = self._split_curr_aux_masks(buf_y, current_task_labels)
        curr_x = buf_x[is_curr]
        curr_y = buf_y[is_curr]
        curr_na = buf_na[is_curr] if (buf_na is not None) else None

        aux_x  = buf_x[is_aux]
        aux_y  = buf_y[is_aux]
        aux_na = buf_na[is_aux] if (buf_na is not None) else None

        # nothing to trim
        if aux_y.numel() == 0:
            return buf_x, buf_y, buf_na, meta

        # fallback not_aug for the trimmer (expects a tensor)
        aux_na_use = aux_na if isinstance(aux_na, torch.Tensor) else aux_x

        # --- EDIT 2: Add try...finally wrapper to force-pass internal gate ---
        # ⬇️ NOTE: _maybe_trim_aux returns FIVE values (x, y, not_aug, logits, meta)
        saved_aux_trim = getattr(self.args, 'aux_trim', 0)
        try:
            self.args.aux_trim = 1  # Force-enable to pass internal gate in KD_DRO
            aux_x_t, aux_y_t, aux_na_t, _logits_t, meta = _maybe_trim_aux(
                aux_x, aux_y, aux_na_use, current_task_labels, self.args, logits=None
            )
        finally:
            self.args.aux_trim = saved_aux_trim # Restore
        # -------------------------------------------------------------------

        # defensive pass-through if trimmer returns None
        if aux_x_t is None or aux_y_t is None:
            aux_x_t, aux_y_t = aux_x, aux_y
            aux_na_t = aux_na if aux_na is not None else None

        # reassemble replay = (all current replay) ⊕ (trimmed aux replay)
        if aux_na_t is None and aux_x_t is not None:
            aux_na_t = aux_x_t  # keep shapes consistent if caller expects it

        x_final = torch.cat([curr_x, aux_x_t], dim=0) if aux_x_t is not None else curr_x
        y_final = torch.cat([curr_y, aux_y_t], dim=0) if aux_y_t is not None else curr_y

        if (curr_na is not None) or (aux_na_t is not None):
            na_curr = curr_na if curr_na is not None else curr_x
            na_aux  = aux_na_t if aux_na_t is not None else aux_x_t
            na_final = torch.cat([na_curr, na_aux], dim=0) if na_aux is not None else na_curr
        else:
            na_final = None

        self._scdt_last_audit = meta  # expose audit for outer logger
        return x_final, y_final, na_final, meta

    @staticmethod
    def _supcon_loss(feats: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Supervised contrastive loss (Khosla et al. formulation) on a single view.
        feats: [B, D] penultimate features (will be L2-normalized)
        labels: [B]
        """
        device = feats.device
        feats = F.normalize(feats, p=2, dim=1)
        sim = torch.matmul(feats, feats.t()) / max(temperature, 1e-8)  # [B, B]
        # numerical stability: subtract row-wise max
        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()

        B = labels.shape[0]
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(device)  # [B, B]
        # remove self-contrast
        logits_mask = torch.ones_like(mask, dtype=torch.bool, device=device)
        logits_mask.fill_diagonal_(False)
        mask = mask * logits_mask.float()

        # log-softmax over all (excluding diagonal)
        exp_sim = torch.exp(sim) * logits_mask.float()
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        # mean over positives for each anchor
        pos_count = mask.sum(dim=1)  # [B]
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / pos_count.clamp_min(1.0)

        # only anchors that actually have positives contribute
        valid = (pos_count > 0).float()
        loss = - (valid * mean_log_prob_pos).sum() / valid.sum().clamp_min(1.0)
        return loss

    # ------------------------------ training step ------------------------------
    def observe(self, inputs, labels, not_aug_inputs, current_task_labels,
                task_number: int = -1, args=None, tb_logger=None, epoch: int = -1,
                update_buffer: bool = True):

        real_batch_size = inputs.shape[0]
        self.opt.zero_grad()

        # ---- Build concatenated minibatch: current ⊕ (replay or SCDT-trimmed replay) ----
        X, Y = inputs, labels
        buf_inputs = buf_labels = buf_not_aug = None
        if hasattr(self, "buffer") and not self.buffer.is_empty():
            try:
                got = self._get_replay_batch(self.args.minibatch_size, include_logits=False)
                if isinstance(got, (list, tuple)):
                    if len(got) >= 2:
                        buf_inputs, buf_labels = got[0], got[1]
                        buf_not_aug = got[2] if len(got) >= 3 else None
                else:
                    buf_inputs, buf_labels = got, None
            except Exception:
                # fallback to vanilla buffer API
                buf_inputs, buf_labels = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform
                )
                buf_not_aug = None

            if buf_inputs is not None and buf_labels is not None and buf_labels.numel() > 0:
                if buf_not_aug is None:
                    buf_not_aug = buf_inputs
                # Optionally trim AUX part of replay (consistent with ER/ER-ACE)
                buf_inputs_t, buf_labels_t, buf_not_aug_t, meta = self._trim_replay_batch_if_needed(
                    buf_inputs, buf_labels, buf_not_aug, current_task_labels
                )
                if isinstance(meta, dict) and len(meta) > 0:
                    self._scdt_last_audit = meta
                if buf_inputs_t is not None and buf_labels_t is not None and buf_labels_t.numel() > 0:
                    X = torch.cat((inputs, buf_inputs_t))
                    Y = torch.cat((labels, buf_labels_t))

        # ---- Forward: logits and features ----
        logits, feats = self.forward_all(X)  # base class hook returns (logits, penultimate features)
        loss_ce = self.loss(logits, Y)
        # P‑SCDT utility update (safe even if disabled in args)
        try:
            self._scdt_update_class_loss_ema(Y, F.cross_entropy(logits, Y, reduction='none'))
        except Exception:
            pass

        # ---- Supervised contrastive term ----
        loss_sup = self._supcon_loss(feats, Y, temperature=float(self.args.supcon_temp))
        loss = loss_ce + float(self.args.supcon_weight) * loss_sup

        # ---- Optimize ----
        loss.backward()
        self.opt.step()

        # ---- Buffer writes: ONLY current-batch samples (consistent with your code) ----
        not_aug_curr = not_aug_inputs if isinstance(not_aug_inputs, torch.Tensor) else inputs
        not_aug_curr = not_aug_curr[:real_batch_size]
        labels_curr = labels[:real_batch_size]

        if update_buffer and current_task_labels != []:
            mask_list = torch.stack([labels_curr == l for l in current_task_labels])
            mask = torch.any(mask_list, dim=0)
            not_aug_curr = not_aug_curr[mask]
            labels_curr = labels_curr[mask]

        if update_buffer and isinstance(not_aug_curr, torch.Tensor) and not_aug_curr.numel() > 0:
            self.buffer.add_data(examples=not_aug_curr, labels=labels_curr)

        return float(loss.item())
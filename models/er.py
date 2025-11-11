
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn.functional as F  # per-sample CE for SCDT audits

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer

# SCDT trimmer (same primitive used in training/ER-ACE)
from utils.attacks.KD_DRO import _maybe_trim_aux


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class Er(ContinualModel):
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Er, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        # expose audits like ER-ACE so training.py can log them
        self._scdt_last_audit: Optional[Dict] = None

    # -------------------------- helpers --------------------------
    @staticmethod
    def _split_curr_aux_masks(labels: torch.Tensor, current_task_labels: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return boolean masks for current-task vs auxiliary labels.
        If current_task_labels is empty, treat all as current (no aux).
        """
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
        Attack **ONLY the replay minibatch**:
          • keep ALL replay items whose labels are in current_task_labels;
          • apply SCDT/AUX trimming to replay items not in current_task_labels;
          • return concatenated (current_replay ⊕ trimmed_aux_replay).

        Returns (x_trim, y_trim, na_trim, meta). Any of the first three can be None if replay empty.
        """
        meta: Dict = {}
        if buf_x is None or buf_y is None or buf_y.numel() == 0:
            return buf_x, buf_y, buf_na, meta

        # --- EDIT 1: Fix the logic gate ---
        # Run if EITHER is enabled. Return only if BOTH are disabled.
        if int(getattr(self.args, 'aux_trim', 0)) == 0 and int(getattr(self.args, 'scdt', 0)) == 0:
            return buf_x, buf_y, buf_na, meta
        # ---------------------------------

        # split replay into current vs aux (relative to *current* task labels)
        is_curr, is_aux = self._split_curr_aux_masks(buf_y, current_task_labels)
        curr_x = buf_x[is_curr]
        curr_y = buf_y[is_curr]
        curr_na = buf_na[is_curr] if (buf_na is not None) else None

        aux_x  = buf_x[is_aux]
        aux_y  = buf_y[is_aux]
        aux_na = buf_na[is_aux] if (buf_na is not None) else None

        if aux_y.numel() == 0:
            # nothing to trim
            return buf_x, buf_y, buf_na, meta

        # the trimmer expects a not-aug tensor; fall back to the images if NA is None
        aux_na_use = aux_na if isinstance(aux_na, torch.Tensor) else aux_x

        # --- EDIT 2: Add try...finally wrapper to force-pass internal gate ---
        # ⬇️ _maybe_trim_aux returns FIVE values (x, y, not_aug, logits, meta)
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

        self._scdt_last_audit = meta  # expose audit for training logger
        return x_final, y_final, na_final, meta
    # -------------------------------------------------------------

    def observe(self, inputs, labels, not_aug_inputs, current_task_labels,
                update_buffer: bool = True, **kwargs):

        real_batch_size = inputs.shape[0]
        self.opt.zero_grad()

        # -------------------- build concatenated (current ⊕ replay_trimmed) minibatch --------------------
        X, Y = inputs, labels
        if not self.buffer.is_empty():
            # robust replay fetch: supports 2- or 3-tensor returns
            buf_inputs = buf_labels = buf_not_aug = None
            try:
                got = self._get_replay_batch(self.args.minibatch_size, include_logits=False)
                if isinstance(got, (list, tuple)):
                    if len(got) >= 2:
                        buf_inputs, buf_labels = got[0], got[1]
                        buf_not_aug = got[2] if len(got) >= 3 else None
                else:
                    buf_inputs, buf_labels = got, None  # very unlikely path; fallback below
            except Exception:
                # fallback to the vanilla Buffer API
                buf_inputs, buf_labels = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform
                )
                buf_not_aug = None

            if buf_inputs is not None and buf_labels is not None:
                if buf_not_aug is None:
                    buf_not_aug = buf_inputs

                # === ATTACK: trim REPLAY ONLY (keep current-task replay; trim aux replay) ===
                buf_inputs_t, buf_labels_t, buf_not_aug_t, meta = self._trim_replay_batch_if_needed(
                    buf_inputs, buf_labels, buf_not_aug, current_task_labels
                )
                # concatenate only if something remains after trimming
                if buf_inputs_t is not None and buf_labels_t is not None and buf_labels_t.numel() > 0:
                    X = torch.cat((inputs, buf_inputs_t))
                    Y = torch.cat((labels, buf_labels_t))
                    # make audit visible to training.py logger (optional)
                    self._scdt_last_audit = meta

        # forward + loss
        outputs = self.net(X)
        loss = self.loss(outputs, Y)

        # --- optional P‑SCDT hook: per-class loss EMA (safe even if method is missing) ---
        try:
            self._scdt_update_class_loss_ema(Y, F.cross_entropy(outputs, Y, reduction='none'))
        except Exception:
            pass

        loss.backward()
        self.opt.step()

        # -------------------- buffer writes: ONLY current-batch samples --------------------
        # Trim writes to current task (as your original code did)
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
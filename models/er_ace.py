

from typing import Tuple, List, Optional, Dict

import torch
import torch.nn.functional as F
from datasets import get_dataset

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer

# SCDT trimmer (same primitive your training.py uses)
from utils.attacks.KD_DRO import _maybe_trim_aux


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class ErACE(ContinualModel):
    NAME = 'er_ace'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(ErACE, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.num_classes = get_dataset(args).N_TASKS * get_dataset(args).N_CLASSES_PER_TASK
        self.task = 0

        # optional debug counters
        self._scdt_last_audit: Optional[Dict] = None

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

    # ---------------------- replay‑only SCDT trimming ----------------------
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
        # Temporarily set aux_trim=1 to pass the internal gate in _maybe_trim_aux
        saved_aux_trim = getattr(self.args, 'aux_trim', 0)
        try:
            self.args.aux_trim = 1
            aux_x_t, aux_y_t, aux_na_t, _logits_t, meta = _maybe_trim_aux(
                aux_x, aux_y, aux_na_use, current_task_labels, self.args, logits=None
            )
        finally:
            self.args.aux_trim = saved_aux_trim # Restore
        # -------------------------------------------------------------------

        # defensive pass‑through if trimmer returns None
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
    # ----------------------------------------------------------------------

    def observe(self, inputs, labels, not_aug_inputs, current_task_labels,
                task_number: int = -1, args=None, tb_logger=None, epoch: int = -1,
                update_buffer: bool = True):
        """
        One training step.

        Base ER‑ACE:
          • CE on current batch (with ACE masking),
          • CE on replay batch.

        Attack (this file):
          • If --aux_trim=1 and --scdt=1, we **trim the replay batch only**
            (AUX = non‑current classes inside replay) under the SCDT budget/window.
            Current‑task replay is preserved.
        """

        # ------------------------------ ACE masking: CURRENT ------------------------------
        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        logits_cur = self.net(inputs)
        mask = torch.zeros_like(logits_cur)
        mask[:, present] = 1

        self.opt.zero_grad()
        if self.seen_so_far.max() < (self.num_classes - 1):
            mask[:, self.seen_so_far.max():] = 1

        if self.task > 0:
            logits_cur = logits_cur.masked_fill(mask == 0, torch.finfo(logits_cur.dtype).min)

        loss_main = self.loss(logits_cur, labels)
        # -----------------------------------------------------------------------------------

        # ------------------------------- Replay path --------------------------------
        loss_re = torch.tensor(0., device=loss_main.device)

        def _buffer_ready(buf):
            try:
                if hasattr(buf, "is_empty") and buf.is_empty():
                    return False
                if getattr(buf, "num_seen_examples", 0) <= 0:
                    return False
                ex = getattr(buf, "examples", None)
                if ex is None:
                    return False
                return int(ex.shape[0]) > 0
            except Exception:
                return False

        if self.task > 0 and hasattr(self, "buffer") and _buffer_ready(self.buffer):
            # get replay minibatch as (x, y) or (x, y, not_aug)
            buf_inputs = buf_labels = buf_not_aug = None
            try:
                got = self._get_replay_batch(self.args.minibatch_size, include_logits=False)
                if isinstance(got, (list, tuple)):
                    if len(got) == 3:
                        buf_inputs, buf_labels, buf_not_aug = got
                    elif len(got) == 2:
                        buf_inputs, buf_labels = got
                        buf_not_aug = None
                    else:
                        buf_inputs, buf_labels = got[0], got[1]
                        buf_not_aug = None
                else:
                    buf_inputs, buf_labels = got
                    buf_not_aug = None
            except Exception:
                buf_inputs, buf_labels = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform
                )
                buf_not_aug = None

            if buf_inputs is not None and buf_labels is not None and buf_labels.numel() > 0:
                if buf_not_aug is None:
                    buf_not_aug = buf_inputs

                # === ATTACK: trim REPLAY ONLY (keep current‑task replay; trim aux replay) ===
                buf_inputs_t, buf_labels_t, buf_not_aug_t, meta = self._trim_replay_batch_if_needed(
                    buf_inputs, buf_labels, buf_not_aug, current_task_labels
                )
                if isinstance(meta, dict) and len(meta) > 0:
                    self._scdt_last_audit = meta

                # compute replay loss if anything remains
                if buf_inputs_t is not None and buf_labels_t is not None and buf_labels_t.numel() > 0:
                    loss_re = self.loss(self.net(buf_inputs_t), buf_labels_t)
        # -------------------------------------------------------------------------

        loss = loss_main + loss_re
        loss.backward()
        self.opt.step()

        # ------------------------------ buffer update (base) ------------------------------
        if update_buffer and current_task_labels != []:
            mask_list = torch.stack([labels == l for l in current_task_labels])
            mask = torch.any(mask_list, dim=0)
            na_write = not_aug_inputs[mask]
            y_write  = labels[mask]
        else:
            na_write = not_aug_inputs
            y_write  = labels

        if update_buffer:
            self.buffer.add_data(examples=na_write, labels=y_write)
        # ----------------------------------------------------------------------------------

        return float(loss.item())
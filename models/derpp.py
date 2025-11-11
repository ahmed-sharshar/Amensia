

import torch
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
# Trimmer (AUX sampler with audits)
from utils.attacks.KD_DRO import _maybe_trim_aux


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    # Keep sensible defaults; users can override on CLI
    parser.add_argument('--alpha', type=float, default=0.0,
                        help='Penalty weight for KD on replay logits.')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Penalty weight for CE on replay labels.')
    return parser


class Derpp(ContinualModel):
    NAME = 'derpp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Derpp, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def _is_trim_enabled(self) -> bool:
        """Trim replay if EITHER aux_trim OR scdt are enabled."""
        return int(getattr(self.args, 'aux_trim', 0)) == 1 or int(getattr(self.args, 'scdt', 0)) == 1

    def observe(self, inputs, labels, not_aug_inputs, current_task_labels,
                task_number: int = -1, args=None, tb_logger=None, epoch: int = -1,
                update_buffer: bool = True):

        # --- current (online) batch ---
        self.opt.zero_grad()
        outputs_curr = self.net(inputs)
        loss = self.loss(outputs_curr, labels)

        # Maintain a class-loss EMA for diagnostics (harmless in baseline)
        try:
            self._scdt_update_class_loss_ema(
                labels, F.cross_entropy(outputs_curr, labels, reduction='none')
            )
        except Exception:
            pass

        # Decide path
        do_trim = self._is_trim_enabled()

        # --- REPLAY PATH ---
        if not self.buffer.is_empty():

            if not do_trim:
                # ============================ BASELINE BEHAVIOR ============================
                # Exact original DER++: draw twice (KD then CE), apply transforms, no audits.

                # KD branch (uses stored logits if available); alpha may be 0.0 (no effect).
                buf_tuple = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
                if isinstance(buf_tuple, (tuple, list)) and len(buf_tuple) >= 1:
                    buf_inputs_kd = buf_tuple[0]
                    buf_logits = None
                    if len(buf_tuple) >= 3 and isinstance(buf_tuple[2], torch.Tensor) and buf_tuple[2].dim() >= 2:
                        buf_logits = buf_tuple[2]
                    if buf_inputs_kd is not None and buf_logits is not None:
                        buf_inputs_kd = buf_inputs_kd.to(self.device)
                        buf_logits = buf_logits.to(self.device)
                        buf_outputs_kd = self.net(buf_inputs_kd)
                        loss = loss + self.args.alpha * F.mse_loss(buf_outputs_kd, buf_logits)

                # CE branch (labels)
                buf_tuple = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
                if isinstance(buf_tuple, (tuple, list)) and len(buf_tuple) >= 2:
                    buf_inputs_ce, buf_labels_ce = buf_tuple[0], buf_tuple[1]
                    if buf_inputs_ce is not None and buf_labels_ce is not None:
                        buf_inputs_ce = buf_inputs_ce.to(self.device)
                        buf_labels_ce = buf_labels_ce.to(self.device)
                        buf_outputs_ce = self.net(buf_inputs_ce)
                        loss = loss + self.args.beta * self.loss(buf_outputs_ce, buf_labels_ce)

                # No SCDT audits in baseline
                self._scdt_last_audit = {}

            else:
                # ======================= SCDT‑TRIMMED REPLAY BEHAVIOR =======================
                # Draw ONCE (with transform), trim under relative_to_aux (buffer-only pool),
                # publish audits, reuse the SAME trimmed batch for CE (+KD if enabled).
                need_kd = bool(getattr(self.args, 'alpha', 0.0) > 0.0)

                buf_tuple = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
                buf_inputs_raw = buf_labels_raw = buf_logits_raw = None
                if isinstance(buf_tuple, (tuple, list)) and len(buf_tuple) >= 1:
                    buf_inputs_raw = buf_tuple[0]
                    buf_labels_raw = buf_tuple[1] if len(buf_tuple) >= 2 else None
                    if len(buf_tuple) >= 3 and isinstance(buf_tuple[2], torch.Tensor) and buf_tuple[2].dim() >= 2:
                        buf_logits_raw = buf_tuple[2]

                if buf_inputs_raw is not None and buf_labels_raw is not None:
                    buf_inputs_raw = buf_inputs_raw.to(self.device)
                    buf_labels_raw = buf_labels_raw.to(self.device)
                    if buf_logits_raw is not None:
                        buf_logits_raw = buf_logits_raw.to(self.device)

                    # Force relative_to_aux for buffer-only trimming to avoid degenerate targets
                    saved_mode = getattr(self.args, 'scdt_mass_mode', 'relative_to_aux')
                    saved_aux_trim = getattr(self.args, 'aux_trim', 0) # NEW
                    try:
                        self.args.scdt_mass_mode = 'relative_to_aux'
                        self.args.aux_trim = 1  # NEW: Force-enable to pass internal gate in KD_DRO
                        buf_inputs_t, buf_labels_t, _, buf_logits_t, meta = _maybe_trim_aux(
                            buf_inputs_raw, buf_labels_raw, buf_inputs_raw,
                            current_task_labels, self.args,
                            logits=(buf_logits_raw if need_kd and (buf_logits_raw is not None) else None)
                        )
                    finally:
                        self.args.scdt_mass_mode = saved_mode
                        self.args.aux_trim = saved_aux_trim # NEW: Restore original value

                    # Publish sampler audit so the training logger can aggregate window & budget metrics
                    try:
                        self._scdt_last_audit = {
                            k: v for k, v in meta.items()
                            if isinstance(k, str) and k.startswith('scd_')
                        }
                    except Exception:
                        self._scdt_last_audit = {}

                    # CE on the SAME trimmed replay set
                    if buf_labels_t is not None and buf_labels_t.numel() > 0:
                        out_ce = self.net(buf_inputs_t)
                        loss = loss + self.args.beta * self.loss(out_ce, buf_labels_t)

                    # KD on the SAME trimmed replay set (if logits available and alpha>0)
                    if need_kd and (buf_logits_t is not None) and (buf_logits_t.numel() > 0):
                        out_kd = self.net(buf_inputs_t)
                        loss = loss + self.args.alpha * F.mse_loss(out_kd, buf_logits_t)

        # --- optimize ---
        loss.backward()
        self.opt.step()

        # --- buffer write: only current-task items (if list provided) ---
        if update_buffer and current_task_labels != []:
            import torch as _torch
            mask_list = _torch.stack([labels == l for l in current_task_labels])
            mask = _torch.any(mask_list, dim=0)
            not_aug_inputs_to_write = not_aug_inputs[mask]
            labels_to_write = labels[mask]
            logits_to_write = outputs_curr.detach()[mask]
        else:
            not_aug_inputs_to_write = not_aug_inputs
            labels_to_write = labels
            logits_to_write = outputs_curr.detach()

        if update_buffer and not_aug_inputs_to_write.numel() > 0:
            self.buffer.add_data(examples=not_aug_inputs_to_write,
                                 labels=labels_to_write,
                                 logits=logits_to_write)

        return float(loss.item())
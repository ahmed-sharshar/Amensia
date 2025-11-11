

import math
import os
import csv
import sys
from argparse import Namespace
from typing import Tuple, Optional, Dict, Any, List

import torch
import numpy as np
from datasets import get_dataset, get_forward_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel

from utils.loggers import *
from utils.status import ProgressBar

from utils.layers_freezing import freeze_layers
import re
import statistics
from torch.utils.data import DataLoader
from utils.freezing_eval import Buffer_dataset
from utils.freezing_eval import model_eval
from copy import deepcopy
from utils.buffer import Buffer

# === KD-DRO helpers (batch trimming / LR scaling). Import is harmless; code is gated by flags. ===
from utils.attacks.KD_DRO import (
    _is_aux_trim_enabled,          # NOTE: we still import, but we ALSO hard-gate locally by args.aux_trim==1
    _maybe_trim_aux,
    _apply_lr_scale_to_optimizer,
    _restore_lr_of_optimizer,
    _aux_debug_log,
)
# CSV writer for SCDT audits (per-step), used to persist sampler telemetry.
from utils.continual_training import _maybe_log_scdt_csv
# ================================================================================================

try:
    import wandb
except ImportError:
    wandb = None


# --------------------------- Robust helpers (added) ---------------------------

def _safe_task_labels(dataset: ContinualDataset, k: int) -> List[int]:
    """
    Return class indices for task k, using dataset.get_task_labels if available,
    otherwise fallback to the conventional contiguous block per task.
    """
    if hasattr(dataset, 'get_task_labels'):
        return list(dataset.get_task_labels(k))
    start = k * dataset.N_CLASSES_PER_TASK
    end = (k + 1) * dataset.N_CLASSES_PER_TASK
    return list(range(start, end))


def _robust_unpack_batch(data):
    """
    Extract (inputs, labels, not_aug_inputs, logits_or_None) from batch tuples
    shaped as (x,y), (x,y,not_aug), or (x,y,not_aug,logits).
    """
    logits = None
    if isinstance(data, (list, tuple)):
        if len(data) >= 2:
            inputs, labels = data[0], data[1]
            not_aug = data[2] if len(data) >= 3 else data[0]
            if len(data) >= 4:
                logits = data[3]
        else:
            inputs, labels = data
            not_aug = inputs
    else:
        inputs, labels = data
        not_aug = inputs
    return inputs, labels, not_aug, logits


def _unpack_loaders(loaders_out, expect_validation: bool):
    """
    Accept get_data_loaders() that returns 2/3/4 items.
    Always return (train_loader, val_loader, buff_loader, extra) where some can be None.
    """
    train_loader = val_loader = buff_loader = extra = None
    if isinstance(loaders_out, (list, tuple)):
        n = len(loaders_out)
        if expect_validation:
            if n == 4:
                train_loader, val_loader, buff_loader, extra = loaders_out
            elif n == 3:
                train_loader, val_loader, buff_loader = loaders_out
            elif n == 2:
                train_loader, val_loader = loaders_out
            elif n == 1:
                train_loader = loaders_out[0]
            else:
                raise ValueError(f"Unexpected number of loaders: {n}")
        else:
            if n >= 1:
                train_loader = loaders_out[0]
            if n >= 2:
                val_loader = loaders_out[1]
            if n >= 3:
                buff_loader = loaders_out[2]
            if n >= 4:
                extra = loaders_out[3]
    else:
        train_loader = loaders_out
    return train_loader, val_loader, buff_loader, extra

# -----------------------------------------------------------------------------


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Masks logits outside the current task labels (task-IL eval in class-IL models).
    """
    current_task_labels = _safe_task_labels(dataset, k)
    mask = torch.full_like(outputs, 1, dtype=bool)
    mask[:, current_task_labels] = False
    outputs[mask] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates accuracy for each past task. Returns (class-il accs, task-il accs).
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0

        # Only assert if both sides are available
        if hasattr(dataset, 'get_task_labels') and hasattr(test_loader.dataset, 'targets'):
            assert set(test_loader.dataset.targets) == set(_safe_task_labels(dataset, k)), \
                "Something wrong in test dataset creation."

        for data in test_loader:
            with torch.no_grad():
                inputs, labels, _, _ = _robust_unpack_batch(data)
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    return accs, accs_mask_classes


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace, attack=None) -> None:
    """
    The training process, including evaluations and loggers.
    attack: optional attack handle (None for baseline). Hooks may run on WAKE/NREM.
    """
    print(args)
    freezed_layers = 0

    if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        # modern API
        args.wandb_url = wandb.run.url

    model.net.to(model.device)
    results, results_mask_classes = [], []

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    # ----------------------- HARD BASELINE GUARD (explicit & simple) -----------------------
    # Baseline mode means: no attack, no AUX trimming.
    BASELINE_MODE = (
        str(getattr(args, 'attack', 'none')) == 'none' and
        str(getattr(args, 'attack_phase', 'none')) == 'none' and
        int(getattr(args, 'aux_trim', 0)) == 0 and int(getattr(args, 'scdt', 0)) == 0
    )
    print(f"[training] BASELINE_MODE = {BASELINE_MODE}")
    # ---------------------------------------------------------------------------------------

    if not args.ignore_other_metrics:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            if 'dream-' in dataset_copy.NAME:
                dataset_copy.set_pos_new_tasks(tuple(range(
                    t * dataset_copy.N_CLASSES_PER_TASK, (t + 1) * dataset_copy.N_CLASSES_PER_TASK)))
            # Just call; ignore arity
            _ = dataset_copy.get_data_loaders()
        if model.NAME not in ('icarl', 'pnn'):
            random_results_class, random_results_task = evaluate(model, dataset_copy)

    buffers_list = []

    # OPTIONAL: provide visibility normalization to the attack (for SSIM/LPIPS)
    if attack is not None and getattr(args, 'vis_metrics', 0):
        try:
            norm = dataset.get_normalization_transform()
            if hasattr(norm, 'mean') and hasattr(norm, 'std'):
                attack.set_visibility_normalization(norm.mean, norm.std)
            elif hasattr(norm, 'transforms'):
                for tr in norm.transforms:
                    if hasattr(tr, 'mean') and hasattr(tr, 'std'):
                        attack.set_visibility_normalization(tr.mean, tr.std)
                        break
        except Exception:
            pass

    global_step = 0

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        model.net.train()

        # Map heads for Dream datasets
        if 'dream-' in dataset.NAME:
            if t == 0:
                forward_dataset = get_forward_dataset(args)
                _ = forward_dataset.get_data_loaders()
                dataset.set_pos_new_tasks(tuple(range(0, dataset.N_CLASSES_PER_TASK)))
            else:
                next_pos_classes = get_next_pos_classes(model, forward_dataset, dataset.get_free_pos())
                dataset.set_pos_new_tasks(next_pos_classes)

        buff_loader_this_task = None
        val_loader = None

        # ----------------- robust loader unpacking (replaces fixed arity) -----------------
        if args.validation:
            train_loader, val_loader, buff_loader, _ = _unpack_loaders(dataset.get_data_loaders(),
                                                                       expect_validation=True)
            if buff_loader is not None:
                buffers_list.append(buff_loader)
            buff_loader_this_task = buff_loader
        elif 'dream-' in dataset.NAME:
            train_loader, val_loader, buff_loader, _ = _unpack_loaders(dataset.get_data_loaders(),
                                                                       expect_validation=False)
            buff_loader_this_task = buff_loader
        else:
            train_loader, val_loader, buff_loader, _ = _unpack_loaders(dataset.get_data_loaders(),
                                                                       expect_validation=False)
            buff_loader_this_task = buff_loader
        # ----------------------------------------------------------------------------------

        current_task_labels = []
        if 'dream-' in dataset.NAME:
            current_task_labels = dataset.get_current_labels()
            print(getattr(train_loader.dataset, 'class_to_idx', {}))
        else:
            current_task_labels = _safe_task_labels(dataset, t)

        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t and not args.ignore_other_metrics:
            accs = evaluate(model, dataset, last=True)
            results[t - 1] = results[t - 1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t - 1] = results_mask_classes[t - 1] + accs[1]

        scheduler = dataset.get_scheduler(model, args)
        print(f"Task n: {t} involves {len(train_loader.dataset)} samples.")

        for epoch in range(model.args.n_epochs):
            if args.model == 'joint':
                continue

            # Freezing evaluation (feature of our repo; leave gated by explicit flag)
            if args.freezing_eval is not None and t > 0 and epoch == 0:
                models, modules_names_to_freeze = [deepcopy(model)], []
                last_frozen_module = None
                for module_name, module in model.net.named_modules():
                    if re.match(re.compile('layer..'), module_name) is not None and not any(
                        word in module_name for word in ["conv1", "bn1", "conv2", "bn2", "4", "shortcut"]
                    ):
                        for _, parameters in module.named_parameters():
                            if parameters.requires_grad != False:
                                modules_names_to_freeze.append(module_name.replace(".", "[") + "]")
                                break
                            else:
                                last_frozen_module = str(module_name.replace(".", "[") + "]")

                print(f"Last frozen layer: {last_frozen_module}")
                if last_frozen_module is not None:
                    freeze_layers(models[0].net, eval(f'models[0].net.{last_frozen_module}'),
                                  torch.Tensor().new_ones((1, 3, 32, 32)).to(model.device))
                print(f"Decide whether to freeze: " + ', '.join(str(m) for m in modules_names_to_freeze))
                for i, module_name in enumerate(modules_names_to_freeze):
                    models.append(deepcopy(model))
                    freeze_layers(models[i + 1].net, eval(f'models[{i + 1}].net.{module_name}'),
                                  torch.Tensor().new_ones((1, 3, 32, 32)).to(model.device))

                # Share the main buffer with all copies
                for mc in models:
                    if hasattr(model, 'buffer'):
                        mc.buffer = model.buffer

                correct, total, accs_val, losses = [], [], [], []
                for _ in models:
                    correct.append(0.0)
                    total.append(0.0)
                    accs_val.append(0.0)
                    losses.append([])

            for i, data in enumerate(train_loader):
                if args.debug_mode and i > 3:
                    break

                # Robustly unpack any batch shape
                inputs, labels, not_aug_inputs, logits = _robust_unpack_batch(data)
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                not_aug_inputs = not_aug_inputs.to(model.device)
                if logits is not None:
                    logits = logits.to(model.device)

                # --------------------------- KD/distillation path ---------------------------
                if logits is not None:
                    # *** EDIT 1: allow AUX-trim for any dataset (not only "dream-") ***
                    if (not BASELINE_MODE) and len(current_task_labels) > 0:
                        inputs_t, labels_t, not_aug_t, logits_t, meta = _maybe_trim_aux(
                            inputs, labels, not_aug_inputs, current_task_labels, args, logits=logits
                        )
                    

                        loss = model.meta_observe(inputs_t, labels_t, not_aug_t, logits_t)

                        # --- P‑SCDT audit logging (AUX trim path) ---
                        scdt_audit = {k: v for k, v in meta.items() if isinstance(k, str) and k.startswith('scdt_')}
                        if scdt_audit:
                            if not args.nowand and wandb is not None:
                                try:
                                    wandb.log(scdt_audit)
                                except Exception:
                                    pass
                            if not args.disable_log:
                                try:
                                    logger.add_scdt_audit(scdt_audit)
                                except Exception:
                                    pass
                            try:
                                _maybe_log_scdt_csv(args, scdt_audit, global_step)
                            except Exception:
                                pass

                        every = int(getattr(args, 'aux_debug_every', 0))
                        if (not BASELINE_MODE) and every > 0 and (global_step % every == 0):
                            _aux_debug_log(args, dict(
                                phase='train', task=t, epoch=epoch, step=global_step, batch=i,
                                n_before=meta['n_before'], n_after=meta['n_after'],
                                n_curr=meta['n_curr'], n_aux=meta['n_aux'], n_aux_kept=meta['n_aux_kept'],
                                realized_aux_frac=round(meta['realized_aux_frac'], 6),
                                lr_scale=round(meta['lr_scale'], 6),
                                kd_path=1
                            ))
                    else:
                        # Pure baseline semantics on KD path
                        loss = model.meta_observe(inputs, labels, not_aug_inputs, logits)
                # ----------------------------------------------------------------------------

                else:
                    # --------------------------- Standard (no-logits) path ---------------------------
                    # Attack hooks (only if an attack object is provided AND not baseline)
                    if (not BASELINE_MODE) and (attack is not None):
                        phase_arg = getattr(args, 'attack_phase', 'none')
                        if phase_arg in ['wake', 'both', 'all']:
                            _orig = inputs.detach().clone()
                            inputs, labels = attack.on_wake_batch(
                                inputs, labels, model=model, dataset=dataset,
                                task_id=t, epoch=epoch, batch_idx=i
                            )
                            if getattr(args, 'vis_metrics', 0):
                                attack.cache_visibility('wake', _orig, inputs)

                    # *** EDIT (new): feed model's per-class CE EMA into KD_DRO for non-KD runs ***
                    # This gives KD_DRO a meaningful preference vector 'u' so scdt_u_corr isn't pinned to 0.
                    if (not BASELINE_MODE) and (int(getattr(args, 'aux_trim', 0)) == 1):
                        try:
                            util_ma = getattr(model, '_scdt_class_loss_ma', None)
                            if util_ma is not None and hasattr(model, 'num_classes') \
                               and isinstance(util_ma, torch.Tensor) and util_ma.numel() == int(model.num_classes):
                                util_list = util_ma.detach().float().cpu().tolist()
                                setattr(args, '_scdt_util_ema', {i: float(v) for i, v in enumerate(util_list)})
                        except Exception:
                            pass

                    # *** EDIT 2: allow AUX-trim for any dataset (not only "dream-") ***
                    if (not BASELINE_MODE) and len(current_task_labels) > 0:
                        inputs_t, labels_t, not_aug_t, _, meta = _maybe_trim_aux(
                            inputs, labels, not_aug_inputs, current_task_labels, args, logits=None
                        )

                    else:
                        inputs_t, labels_t, not_aug_t = inputs, labels, not_aug_inputs
                        meta = {'lr_scale': 1.0, 'n_before': int(labels.shape[0]), 'n_after': int(labels.shape[0]),
                                'n_curr': 0, 'n_aux': 0, 'n_aux_kept': 0, 'realized_aux_frac': 0.0}

                    if args.freezing_eval is not None and t > 0 and epoch == 0:
                        for model_copy in models:
                            loss = model_copy.meta_observe(inputs_t, labels_t, not_aug_t, current_task_labels)
                            assert not math.isnan(loss)
                    else:
                        loss = model.meta_observe(inputs_t, labels_t, not_aug_t, current_task_labels)

                    # --- P‑SCDT audit logging (AUX trim path) ---
                    scdt_audit = {k: v for k, v in meta.items() if isinstance(k, str) and k.startswith('scdt_')}
                    if scdt_audit:
                        if not args.nowand and wandb is not None:
                            try:
                                wandb.log(scdt_audit)
                            except Exception:
                                pass
                        if not args.disable_log:
                            try:
                                logger.add_scdt_audit(scdt_audit)
                            except Exception:
                                pass
                        try:
                            _maybe_log_scdt_csv(args, scdt_audit, global_step)
                        except Exception:
                            pass

                    # *** EDIT 3: aux_debug_every should trigger when > 0 (not == 1) ***
                    every = int(getattr(args, 'aux_debug_every', 0))
                    if (not BASELINE_MODE) and every > 0 and (global_step % every == 0):
                        _aux_debug_log(args, dict(
                            phase='train', task=t, epoch=epoch, step=global_step, batch=i,
                            n_before=meta['n_before'], n_after=meta['n_after'],
                            n_curr=meta['n_curr'], n_aux=meta['n_aux'], n_aux_kept=meta['n_aux_kept'],
                            realized_aux_frac=round(meta['realized_aux_frac'], 6),
                            lr_scale=round(meta['lr_scale'], 6),
                            kd_path=0
                        ))
                    # -------------------------------------------------------------------------------

                # --- P‑SCDT audit logging (model-level sampler, if present) ---
                _model_audit = getattr(model, '_scdt_last_audit', None)
                if isinstance(_model_audit, dict) and len(_model_audit) > 0:
                    if not args.nowand and wandb is not None:
                        try:
                            wandb.log(_model_audit)
                        except Exception:
                            pass
                    if not args.disable_log:
                        try:
                            logger.add_scdt_audit(_model_audit)
                        except Exception:
                            pass

                assert not math.isnan(loss)
                progress_bar.prog(i, len(train_loader), epoch, t, loss)
                global_step += 1

            # =============================== Freezing evaluation ===============================
            if args.freezing_eval is not None and t > 0 and epoch == 0:

                use_buffer_eval = (args.freezing_eval in ["buffer", "training_and_buff"]) \
                                  and (t - 1) < len(buffers_list) \
                                  and (buffers_list[t - 1] is not None)

                if args.freezing_eval != "training" and use_buffer_eval:
                    correct_buff, total_buff, losses_buff = [], [], []
                    for _ in models:
                        correct_buff.append(0.0)
                        total_buff.append(0.0)
                        losses_buff.append([])

                    for i_b, data in enumerate(buffers_list[t - 1]):
                        inputs, labels, _, _ = _robust_unpack_batch(data)
                        inputs, labels = inputs.to(model.device), labels.to(model.device)
                        for m_i, model_copy in enumerate(models):
                            correct_batch, total_batch, loss = model_eval(
                                model_copy, inputs, labels, dataset.get_loss())
                            correct_buff[m_i] += correct_batch
                            total_buff[m_i] += total_batch
                            losses_buff[m_i].append(loss)

                    accs_buff = []
                    for m_i, loss_list in enumerate(losses_buff):
                        accs_buff.append(correct_buff[m_i] / total_buff[m_i] * 100 if total_buff[m_i] > 0 else 0.0)
                        losses_buff[m_i] = statistics.mean(loss_list) if len(loss_list) > 0 else float('inf')

                # Validation on val_loader (if present)
                correct, total, accs_val, losses = [], [], [], []
                for _ in models:
                    correct.append(0.0)
                    total.append(0.0)
                    accs_val.append(0.0)
                    losses.append([])

                if args.freezing_eval != "buffer" and val_loader is not None:
                    for i_v, data in enumerate(val_loader):
                        inputs, labels, _, _ = _robust_unpack_batch(data)
                        inputs, labels = inputs.to(model.device), labels.to(model.device)
                        for m_i, model_copy in enumerate(models):
                            correct_batch, total_batch, loss = model_eval(
                                model_copy, inputs, labels, dataset.get_loss())
                            correct[m_i] += correct_batch
                            total[m_i] += total_batch
                            losses[m_i].append(loss)

                    for m_i, loss_list in enumerate(losses):
                        accs_val[m_i] = correct[m_i] / total[m_i] * 100 if total_m[i] > 0 else 0.0
                        losses[m_i] = round(statistics.mean(loss_list), 2) if len(loss_list) > 0 else float('inf')

                if args.freezing_eval == "buffer" and use_buffer_eval:
                    print()
                    for m_i, _ in enumerate(accs_val):
                        print(f"accs_buff_{m_i} = {accs_buff[m_i]:.4f} e loss_buff_{m_i} = {losses_buff[m_i]:.4f}")
                        accs_val[m_i] = accs_buff[m_i]
                        losses[m_i] = losses_buff[m_i]

                if args.freezing_eval == "training_and_buff" and use_buffer_eval and val_loader is not None:
                    print()
                    for m_i, _ in enumerate(accs_val):
                        accs_val[m_i] = round(statistics.mean([accs_val[m_i], accs_buff[m_i]]), 2)
                        losses[m_i] = round(statistics.mean([losses[m_i], losses_buff[m_i]]), 2)
                        print(f"accs_{m_i} = {accs_val[m_i]:.4f} e loss_{m_i} = {losses[m_i]:.4f}")

                best_model = losses.index(min(losses)) if len(losses) > 0 else 0

                model.net.load_state_dict(models[best_model].net.state_dict())
                freezed_layers = freezed_layers + best_model

                if best_model != 0:
                    print(f"It's better freezing up to {modules_names_to_freeze[best_model-1]}")
                    freeze_layers(model.net, eval('model.net.' + modules_names_to_freeze[best_model - 1]),
                                  torch.Tensor().new_ones((1, 3, 32, 32)).to(model.device))
                else:
                    print(f"It's better not to freeze")
                    if last_frozen_module is not None:
                        freeze_layers(model.net, eval('model.net.' + last_frozen_module),
                                      torch.Tensor().new_ones((1, 3, 32, 32)).to(model.device))
            # ================================================================================

            if scheduler is not None:
                scheduler.step()

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        if not args.disable_log:
            logger.log(mean_acc)
            logger.log_fullacc(accs)

        if not args.nowand:
            d2 = {'RESULT_class_mean_accs': mean_acc[0], 'RESULT_task_mean_accs': mean_acc[1],
                  'STEP': t, 'num_freezed_layers': freezed_layers,
                  **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs[0])},
                  **{f'RESULT_task_acc_{i}': a for i, a in enumerate(accs[1])}}
            wandb.log(d2)

        # Per-task attack stats logging
        if attack is not None:
            stats = attack.summarize(reset=True, compute_visibility=bool(getattr(args, 'vis_metrics', 0)))
            if not args.nowand and wandb is not None:
                wandb.log(stats)
            else:
                print('ATTACK_STATS', stats)

    if not args.disable_log and not args.ignore_other_metrics:
        logger.add_bwt(results, results_mask_classes)
        logger.add_forgetting(results, results_mask_classes)
        if model.NAME not in ('icarl', 'pnn'):
            logger.add_fwt(results, random_results_class, results_mask_classes, random_results_task)

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.url
            wandb.log(d)

    if not args.nowand:
        wandb.finish()


def get_next_pos_classes(model: ContinualModel, forward_dataset: ContinualDataset, free_heads: list):
    status = model.net.training
    # robust open
    forward_train_loader, _, _, _ = _unpack_loaders(forward_dataset.get_data_loaders(), expect_validation=False)
    model.net.eval()

    list_labels, list_pred = [], []

    for data in forward_train_loader:
        inputs, labels, _, _ = _robust_unpack_batch(data)
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        with torch.no_grad():
            outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)

        list_pred.append(pred.cpu())
        list_labels.append(labels.cpu())

    pred = torch.cat(list_pred)
    labels = torch.cat(list_labels)

    predictions_matrix = torch.ones(
        (forward_dataset.N_CLASSES_PER_TASK,
         forward_dataset.N_CLASSES_PER_TASK * forward_dataset.N_TASKS),
        dtype=torch.int) * -1
    predictions_matrix[:, free_heads] = 0

    if hasattr(forward_train_loader.dataset, 'targets'):
        next_labels = list(set(forward_train_loader.dataset.targets))
    else:
        next_labels = sorted(list(set(labels.tolist())))
    print(f'{forward_dataset.NAME} next classes: {next_labels}')

    for i, label in enumerate(next_labels):
        l_mask = [labels == label]
        all_pred = pred[l_mask].unique().tolist()
        print(f'activated_heads: {all_pred}')
        possible_outputs = list(set(all_pred) & set(free_heads))

        for l in possible_outputs:
            predictions_matrix[i, l] = (pred[l_mask] == l).sum(dim=0).item()
    print(predictions_matrix)

    list_pos = [0] * len(next_labels)
    for row in range(predictions_matrix.shape[0]):
        index = (predictions_matrix == torch.max(predictions_matrix)).nonzero(as_tuple=False)[0]
        r = index[0].item()
        c = index[1].item()
        list_pos[r] = c
        predictions_matrix[r, :] = -1
        predictions_matrix[:, c] = -1

    print('next heads:', list_pos)

    model.net.train(status)
    return list_pos
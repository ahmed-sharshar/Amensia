
import os
import csv
from argparse import Namespace

import torch
from datasets import get_dataset
from models import get_model
from models.utils.continual_model import ContinualModel
from utils.loggers import Logger

from utils.status import progress_bar

try:
    import wandb
except ImportError:
    wandb = None


def _maybe_log_scdt_csv(args: Namespace, audit: dict, step_idx: int) -> None:
    """
    Optional per-step CSV logging for P‑SCDT audits.
    Writes only when args.scdt_debug_log is provided. If args.scdt_debug_every > 0,
    logs every k steps.
    """
    path = getattr(args, 'scdt_debug_log', '')
    if not path or not isinstance(audit, dict) or len(audit) == 0:
        return
    every = int(getattr(args, 'scdt_debug_every', 0))
    if every > 0 and (step_idx % every != 0):
        return

    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    row = {'step': step_idx}
    # only keep keys that start with 'scdt_'
    for k, v in audit.items():
        if isinstance(k, str) and k.startswith('scdt_'):
            row[k] = v
    file_exists = os.path.isfile(path)
    with open(path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate(model: ContinualModel, dataset) -> float:
    """
    Evaluates the final accuracy of the model.
    :param model: the model to be evaluated
    :param dataset: the GCL dataset at hand
    :return: a float value that indicates the accuracy
    """
    model.net.eval()
    correct, total = 0, 0
    while not dataset.test_over:
        inputs, labels = dataset.get_test_data()
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct += torch.sum(predicted == labels).item()
        total += labels.shape[0]

    acc = correct / total * 100
    return acc


def train(args: Namespace):
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """

    dataset = get_dataset(args)
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())
    model.net.to(model.device)

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

    if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        args.wandb_url = wandb.run.get_url()

    model.net.train()
    epoch, i = 0, 0
    while not dataset.train_over:
        inputs, labels, not_aug_inputs = dataset.get_train_data()
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        not_aug_inputs = not_aug_inputs.to(model.device)
        loss = model.observe(inputs, labels, not_aug_inputs)

        # ---- P‑SCDT audits: per-batch logging (wandb) + CSV + aggregate in Logger ----
        scdt_audit = getattr(model, '_scdt_last_audit', None)
        if isinstance(scdt_audit, dict) and len(scdt_audit) > 0:
            # wandb live logging (per batch)
            if not args.nowand and wandb is not None:
                try:
                    wandb.log(scdt_audit)
                except Exception:
                    pass
            # CSV (optional, if args.scdt_debug_log is provided)
            _maybe_log_scdt_csv(args, scdt_audit, step_idx=i)
            # aggregate in repo Logger (summary written at the end)
            if not args.disable_log:
                try:
                    logger.add_scdt_audit(scdt_audit)
                except Exception:
                    pass
        # -----------------------------------------------------------------------------

        progress_bar(i, dataset.LENGTH // args.batch_size, epoch, 'C', loss)
        i += 1

    if model.NAME == 'joint_gcl':
        model.end_task(dataset)

    acc = evaluate(model, dataset)
    print('Accuracy:', acc)

    if not args.disable_log:
        logger.log(acc)
        logger.write(vars(args))

    if not args.nowand:
        wandb.log({'Accuracy': acc})
        wandb.finish()

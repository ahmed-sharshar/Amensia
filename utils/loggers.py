
from contextlib import suppress
import os
import sys
from typing import Any, Dict, List

import numpy as np

from utils import create_if_not_exists
from utils.conf import base_path
from utils.metrics import backward_transfer, forward_transfer, forgetting

useless_args = ['dataset', 'tensorboard', 'validation', 'model',
                'csv_log', 'notes', 'load_best_args']


def print_mean_accuracy(mean_acc: np.ndarray, task_number: int,
                        setting: str) -> None:
    """
    Prints the mean accuracy on stderr.
    :param mean_acc: mean accuracy value
    :param task_number: task index
    :param setting: the setting of the benchmark
    """
    if setting == 'domain-il':
        mean_acc, _ = mean_acc
        print('\nAccuracy for {} task(s): {} %'.format(
            task_number, round(mean_acc, 2)), file=sys.stderr)
    else:
        mean_acc_class_il, mean_acc_task_il = mean_acc
        print('\nAccuracy for {} task(s): \t [Class-IL]: {} %'
              ' \t [Task-IL]: {} %\n'.format(task_number, round(
                  mean_acc_class_il, 2), round(mean_acc_task_il, 2)), file=sys.stderr)


class Logger:
    def __init__(self, setting_str: str, dataset_str: str,
                 model_str: str) -> None:
        self.accs = []
        self.fullaccs = []
        if setting_str == 'class-il':
            self.accs_mask_classes = []
            self.fullaccs_mask_classes = []
        self.setting = setting_str
        self.dataset = dataset_str
        self.model = model_str
        self.fwt = None
        self.fwt_mask_classes = None
        self.bwt = None
        self.bwt_mask_classes = None
        self.forgetting = None
        self.forgetting_mask_classes = None

        # --- P‑SCDT / KD‑DRO audit storage ---
        # Training loop may push per-step dicts here, e.g.:
        # {'scdt_div_batch', 'scdt_div_window', 'scdt_mass_dev',
        #  'scdt_rounding_L1_gap', 'scdt_u_corr', 'scdt_div', 'scdt_budget',
        #  'scdt_window_violation', ...}
        # For AUX budget we expect:
        #  'scdt_aux_target_frac', 'scdt_aux_target_mode',
        #  'scdt_aux_realized_frac_total', 'scdt_aux_realized_frac_aux',
        #  'scdt_aux_frac_error', 'scdt_aux_violation'
        self.scdt_audits: List[Dict[str, Any]] = []

    def add_scdt_audit(self, audit: Dict[str, Any]) -> None:
        """Append a per-step audit dict produced by sampler or KD-DRO trimming."""
        if isinstance(audit, dict):
            self.scdt_audits.append(audit)

    def _aggregate_scdt(self) -> Dict[str, Any]:
        """
        Aggregate stored audits into summary statistics (means and p95s).
        Nan-safe and robust to missing keys.
        Also reports AUX keep‑fraction error summaries when available.
        """
        out: Dict[str, Any] = {}
        audits = self.scdt_audits
        n = len(audits)
        out['scdt_steps'] = n
        if n == 0:
            return out

        def collect_numeric(key: str) -> np.ndarray:
            vals = []
            for a in audits:
                if key in a:
                    with suppress(Exception):
                        fv = float(a[key])
                        if not np.isnan(fv):
                            vals.append(fv)
            return np.array(vals, dtype=np.float64)

        def collect_last(key: str):
            vals = [a[key] for a in audits if key in a]
            return vals[-1] if len(vals) > 0 else None

        def agg_push(key: str, arr: np.ndarray):
            if arr.size == 0:
                return
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                return
            out[f'{key}_mean'] = float(np.mean(arr))
            out[f'{key}_p95'] = float(np.percentile(arr, 95))

        # --- Core sampler visibility stats ---
        agg_push('scdt_div_batch', collect_numeric('scdt_div_batch'))
        agg_push('scdt_div_window', collect_numeric('scdt_div_window'))
        agg_push('scdt_mass_dev', collect_numeric('scdt_mass_dev'))
        agg_push('scdt_rounding_L1_gap', collect_numeric('scdt_rounding_L1_gap'))

        # ---- Budget Efficiency (Optimality): mean utility gain over steps ----
        # Populated by sampler/trimmer as 'scdt_u_gain' per step.
        u_gain_arr = collect_numeric('scdt_u_gain')
        agg_push('scdt_u_gain', u_gain_arr)
        if u_gain_arr.size > 0:
            out['scdt_u_gain_defined_steps'] = int(u_gain_arr.size)

        # --- KD-only mass deviation (avoid constant zeros from sampler path) ---
        kd_mass_vals = []
        for a in audits:
            if ('scdt_mass_dev' in a) and ('scdt_mass_target' in a or 'scdt_mass_realized' in a):
                with suppress(Exception):
                    fv = float(a['scdt_mass_dev'])
                    if not np.isnan(fv):
                        kd_mass_vals.append(fv)
        kd_mass_vals = np.array(kd_mass_vals, dtype=np.float64)
        if kd_mass_vals.size > 0:
            kd_mass_vals = kd_mass_vals[~np.isnan(kd_mass_vals)]
            if kd_mass_vals.size > 0:
                out['scdt_mass_dev_kd_mean'] = float(np.mean(kd_mass_vals))
                out['scdt_mass_dev_kd_p95'] = float(np.percentile(kd_mass_vals, 95))
                out['scdt_mass_dev_kd_steps'] = int(kd_mass_vals.size)

        # --- Correlation coverage & stats (detectability proxy) ---
        u_corr = collect_numeric('scdt_u_corr')
        if u_corr.size > 0:
            u_corr = u_corr[~np.isnan(u_corr)]
            if u_corr.size > 0:
                out['scdt_u_corr_defined_steps'] = int(u_corr.size)
                out['scdt_u_corr_nonzero_rate'] = float(np.mean(np.abs(u_corr) > 0.0))
                out['scdt_u_corr_mean'] = float(np.mean(u_corr))
                out['scdt_u_corr_p95'] = float(np.percentile(u_corr, 95))

        # --- Window violation counts/rate ---
        viol = collect_numeric('scdt_window_violation')
        if viol.size > 0:
            cnt = int(np.sum(viol))
            out['scdt_window_violation_count'] = cnt
            out['scdt_window_violation_rate'] = float(cnt / max(1, n))

        # --- AUX keep‑fraction metrics (BUDGET) ---
        # Realized fractions (both definitions) + absolute error vs target f
        agg_push('scdt_aux_realized_frac_total', collect_numeric('scdt_aux_realized_frac_total'))
        agg_push('scdt_aux_realized_frac_aux',   collect_numeric('scdt_aux_realized_frac_aux'))
        agg_push('scdt_aux_frac_error',          collect_numeric('scdt_aux_frac_error'))

        # AUX budget violation rate (over applicable steps only, if flag present)
        aux_viol = collect_numeric('scdt_aux_violation')
        if aux_viol.size > 0:
            aux_def = collect_numeric('scdt_aux_defined')
            if aux_def.size == aux_viol.size:
                mask = aux_def > 0.5
                denom = int(np.sum(mask))
                cnt = int(np.sum(aux_viol[mask])) if denom > 0 else 0
            else:
                # Backward-compat: if defined-flag is missing, use number of entries
                denom = int(aux_viol.size)
                cnt = int(np.sum(aux_viol))
            out['scdt_aux_violation_count'] = cnt
            out['scdt_aux_defined_steps'] = denom
            out['scdt_aux_violation_rate'] = float(cnt / max(1, denom))

        # --- AUX Signed Bias (defined steps only) ---
        # bias_t = realized - target; use realized that matches the mode
        biases = []
        for a in audits:
            try:
                # Apply applicability mask if present
                if 'scdt_aux_defined' in a and float(a.get('scdt_aux_defined', 0.0)) < 0.5:
                    continue
                f = a.get('scdt_aux_target_frac', None)
                if f is None:
                    continue
                mode = a.get('scdt_aux_target_mode', 'relative_to_aux')
                if mode == 'relative_to_aux':
                    realized = a.get('scdt_aux_realized_frac_aux', None)
                else:  # 'relative_to_total' (i.e., final_batch_frac)
                    realized = a.get('scdt_aux_realized_frac_total', None)
                if realized is None:
                    continue
                f = float(f); realized = float(realized)
                if not (np.isnan(f) or np.isnan(realized)):
                    biases.append(realized - f)
            except Exception:
                pass
        if len(biases) > 0:
            biases = np.asarray(biases, dtype=np.float64)
            out['scdt_aux_bias_defined_steps'] = int(biases.size)
            out['scdt_aux_bias_mean'] = float(np.mean(biases))
            out['scdt_aux_bias_median'] = float(np.median(biases))

        # Last-seen target (f) and mode ('relative_to_total' | 'relative_to_aux')
        last_f = collect_last('scdt_aux_target_frac')
        last_mode = collect_last('scdt_aux_target_mode')
        if last_f is not None:
            with suppress(Exception):
                out['scdt_aux_target_frac_last'] = float(last_f)
        if last_mode is not None:
            out['scdt_aux_target_mode_last'] = last_mode

        # --- Context: last divergence type/budget ---
        last_div = collect_last('scdt_div')
        last_budget = collect_last('scdt_budget')
        if last_div is not None:
            out['scdt_div_last'] = last_div
        if last_budget is not None:
            with suppress(Exception):
                out['scdt_budget_last'] = float(last_budget)

        return out

    def dump(self):
        dic = {
            'accs': self.accs,
            'fullaccs': self.fullaccs,
            'fwt': self.fwt,
            'bwt': self.bwt,
            'forgetting': self.forgetting,
            'fwt_mask_classes': self.fwt_mask_classes,
            'bwt_mask_classes': self.bwt_mask_classes,
            'forgetting_mask_classes': self.forgetting_mask_classes,
        }
        if self.setting == 'class-il':
            dic['accs_mask_classes'] = self.accs_mask_classes
            dic['fullaccs_mask_classes'] = self.fullaccs_mask_classes

        # P‑SCDT aggregated metrics (includes AUX budget if present)
        scdt_summary = self._aggregate_scdt()
        if scdt_summary:
            dic.update(scdt_summary)

        return dic

    def load(self, dic):
        self.accs = dic['accs']
        self.fullaccs = dic['fullaccs']
        self.fwt = dic['fwt']
        self.bwt = dic['bwt']
        self.forgetting = dic['forgetting']
        self.fwt_mask_classes = dic['fwt_mask_classes']
        self.bwt_mask_classes = dic['bwt_mask_classes']
        self.forgetting_mask_classes = dic['forgetting_mask_classes']
        if self.setting == 'class-il':
            self.accs_mask_classes = dic['accs_mask_classes']
            self.fullaccs_mask_classes = dic['fullaccs_mask_classes']

    def rewind(self, num):
        self.accs = self.accs[:-num]
        self.fullaccs = self.fullaccs[:-num]
        with suppress(BaseException):
            self.fwt = self.fwt[:-num]
            self.bwt = self.bwt[:-num]
            self.forgetting = self.forgetting[:-num]
            self.fwt_mask_classes = self.fwt_mask_classes[:-num]
            self.bwt_mask_classes = self.bwt_mask_classes[:-num]
            self.forgetting_mask_classes = self.forgetting_mask_classes[:-num]

        if self.setting == 'class-il':
            self.accs_mask_classes = self.accs_mask_classes[:-num]
            self.fullaccs_mask_classes = self.fullaccs_mask_classes[:-num]

    def add_fwt(self, results, accs, results_mask_classes, accs_mask_classes):
        self.fwt = forward_transfer(results, accs)
        if self.setting == 'class-il':
            self.fwt_mask_classes = forward_transfer(results_mask_classes, accs_mask_classes)

    def add_bwt(self, results, results_mask_classes):
        self.bwt = backward_transfer(results)
        self.bwt_mask_classes = backward_transfer(results_mask_classes)

    def add_forgetting(self, results, results_mask_classes):
        self.forgetting = forgetting(results)
        self.forgetting_mask_classes = forgetting(results_mask_classes)

    def log(self, mean_acc: np.ndarray) -> None:
        """
        Logs a mean accuracy value.
        :param mean_acc: mean accuracy value
        """
        if self.setting == 'general-continual':
            self.accs.append(mean_acc)
        elif self.setting == 'domain-il':
            mean_acc, _ = mean_acc
            self.accs.append(mean_acc)
        else:
            mean_acc_class_il, mean_acc_task_il = mean_acc
            self.accs.append(mean_acc_class_il)
            self.accs_mask_classes.append(mean_acc_task_il)

    def log_fullacc(self, accs):
        if self.setting == 'class-il':
            acc_class_il, acc_task_il = accs
            self.fullaccs.append(acc_class_il)
            self.fullaccs_mask_classes.append(acc_task_il)

    def write(self, args: Dict[str, Any]) -> None:
        """
        writes out the logged value along with its arguments.
        :param args: the namespace of the current experiment
        """
        wrargs = args.copy()

        for i, acc in enumerate(self.accs):
            wrargs['accmean_task' + str(i + 1)] = acc

        for i, fa in enumerate(self.fullaccs):
            for j, acc in enumerate(fa):
                wrargs['accuracy_' + str(j + 1) + '_task' + str(i + 1)] = acc

        wrargs['forward_transfer'] = self.fwt
        wrargs['backward_transfer'] = self.bwt
        wrargs['forgetting'] = self.forgetting

        # Aggregated P‑SCDT/KD‑DRO metrics (includes AUX budget stats)
        wrargs.update(self._aggregate_scdt())

        target_folder = base_path() + "results/"

        create_if_not_exists(target_folder + self.setting)
        create_if_not_exists(target_folder + self.setting +
                             "/" + self.dataset)
        create_if_not_exists(target_folder + self.setting +
                             "/" + self.dataset + "/" + self.model)

        path = target_folder + self.setting + "/" + self.dataset\
            + "/" + self.model + "/logs.pyd"
        with open(path, 'a') as f:
            f.write(str(wrargs) + '\n')

        if self.setting == 'class-il':
            create_if_not_exists(os.path.join(*[target_folder, "task-il/", self.dataset]))
            create_if_not_exists(target_folder + "task-il/"
                                 + self.dataset + "/" + self.model)

            for i, acc in enumerate(self.accs_mask_classes):
                wrargs['accmean_task' + str(i + 1)] = acc

            for i, fa in enumerate(self.fullaccs_mask_classes):
                for j, acc in enumerate(fa):
                    wrargs['accuracy_' + str(j + 1) + '_task' + str(i + 1)] = acc

            wrargs['forward_transfer'] = self.fwt_mask_classes
            wrargs['backward_transfer'] = self.bwt_mask_classes
            wrargs['forgetting'] = self.forgetting_mask_classes

            # Keep the same aggregated metrics in the task‑IL log as well
            path = target_folder + "task-il" + "/" + self.dataset + "/"\
                + self.model + "/logs.pyd"
            with open(path, 'a') as f:
                f.write(str(wrargs) + '\n')

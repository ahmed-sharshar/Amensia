
import numpy as np


def backward_transfer(results):
    n_tasks = len(results)
    li = []
    for i in range(n_tasks - 1):
        li.append(results[-1][i] - results[i][i])

    return np.mean(li)


def forward_transfer(results, random_results):
    n_tasks = len(results)
    li = []
    for i in range(1, n_tasks):
        li.append(results[i - 1][i] - random_results[i])

    return np.mean(li)


def forgetting(results):
    n_tasks = len(results)
    li = []
    for i in range(n_tasks - 1):
        results[i] += [0.0] * (n_tasks - len(results[i]))
    np_res = np.array(results)
    maxx = np.max(np_res, axis=0)
    for i in range(n_tasks - 1):
        li.append(maxx[i] - results[-1][i])

    return np.mean(li)


# ---------------------------
# Convenience helpers (added)
# ---------------------------

def damage_per_budget(delta_bwt: float, f: float) -> float:
    """
    Compute Damage-per-Budget, as suggested in the evaluation:
        DPB = ( -ΔBWT ) / f
    where ΔBWT = (BWT_method - BWT_baseline) at the same (f, δ).
    Positive values mean larger damage per unit of kept-AUX budget.

    :param delta_bwt: Difference in BWT vs. baseline (can be negative).
    :param f: Keep fraction (budget) in KD–DRO, in (0, 1].
    :return: Scalar DPB. Uses a small epsilon to avoid division by zero.
    """
    eps = 1e-12
    return float((-delta_bwt) / max(f, eps))


def area_under_negative_bwt(bwt_list) -> float:
    """
    Area under the negative part of the BWT retention curve across tasks.
    We integrate only the negative portions (clipped at 0), using trapezoidal rule.

    NOTE: Returns a non-positive value (≤ 0). More negative ⇒ more forgetting area.
    If you prefer magnitude, wrap with `abs(...)` at call site.

    :param bwt_list: Iterable of BWT values over tasks (list or np.ndarray).
    :return: Trapezoidal integral of min(0, BWT_t) over task index.
    """
    bwt = np.asarray(bwt_list, dtype=np.float64)
    neg = np.minimum(bwt, 0.0)
    # dx=1 between consecutive tasks
    return float(np.trapz(neg))


def pearsonr(xs, ys) -> float:
    """
    Lightweight Pearson correlation coefficient between two sequences.

    :param xs: Iterable of numbers
    :param ys: Iterable of numbers
    :return: Pearson r in [-1, 1]; returns np.nan if not defined (len<2 or zero variance).
    """
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.size < 2 or y.size < 2 or x.size != y.size:
        return float('nan')
    x_mean = x.mean()
    y_mean = y.mean()
    x_std = x.std()
    y_std = y.std()
    if x_std <= 1e-12 or y_std <= 1e-12:
        return float('nan')
    r = np.mean((x - x_mean) * (y - y_mean)) / (x_std * y_std)
    # Numerical clipping
    return float(np.clip(r, -1.0, 1.0))

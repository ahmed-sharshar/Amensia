
import random
import torch
import numpy as np

def get_device() -> torch.device:
    """
    Returns the GPU device if available else CPU.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def base_path() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    return './data/'

def base_path_dataset() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    return './data/'


def set_random_seed(seed: int) -> None:
    """
    Make randomness as reproducible as practical across common libs and PyTorch backends.
    Call this early (ideally at the very start of your program, before creating models,
    CUDA tensors, or DataLoaders).

    Notes:
      • PYTHONHASHSEED only fully takes effect if set before the Python process starts,
        but setting it here is still harmless.
      • Some CUDA ops require CUBLAS_WORKSPACE_CONFIG for full determinism.
      • Determinism may reduce performance and can error if an op has no deterministic
        implementation; we use warn_only=True to warn instead of raising.
    """
    import os
    import random
    import numpy as np
    import torch

    # -------- Environment-level seeds / flags --------
    os.environ["PYTHONHASHSEED"] = str(seed)            # best set before interpreter start
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed for deterministic matmul on CUDA

    # -------- Python / NumPy --------
    random.seed(seed)
    np.random.seed(seed)

    # -------- PyTorch (CPU & CUDA) --------
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # all GPUs

    # cuDNN / CUDA determinism knobs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # must be False for determinism

    # Avoid TF32 variability on Ampere+ (not strictly determinism, but improves reproducibility)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass  # older PyTorch may not have these toggles

    # Enforce deterministic algorithms where available
    try:
        # warn_only=True -> log a warning if a non-deterministic op is hit instead of raising
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        # Older PyTorch fallback (deprecated in newer versions)
        try:
            torch.set_deterministic(True)  # type: ignore[attr-defined]
        except Exception:
            pass

    # Optional: seed OpenCV if present (often used in data pipelines)
    try:
        import cv2  # type: ignore
        cv2.setRNGSeed(seed)
    except Exception:
        pass

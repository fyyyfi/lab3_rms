import numpy as np


def mse(ref: np.ndarray, test: np.ndarray) -> float:
    d = ref.astype(np.float64) - np.clip(test, 0, 255)
    return float(np.mean(d * d))


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    """PSNR = 10·lg(255² / MSE), дБ."""
    e = mse(ref, test)
    return float("inf") if e == 0 else 10.0 * np.log10(255.0 ** 2 / e)

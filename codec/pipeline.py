"""Повний цикл: перетворення -> стиснення -> обернене перетворення -> метрики."""
from dataclasses import dataclass

import numpy as np

from . import compress, dct8, haar
from .metrics import psnr


@dataclass
class Result:
    method: str
    restored: np.ndarray
    coef: np.ndarray        # коефіцієнти після стиснення
    nonzero: int            # фактична к-сть ненульових (np.count_nonzero)
    total: int
    psnr: float
    bpp: float | None = None     # оцінка біт/піксель (лише для квантування)

    @property
    def share(self) -> float:
        return self.nonzero / self.total

    @property
    def cr_coef(self) -> float:
        return self.total / max(self.nonzero, 1)

    @property
    def cr_bits(self) -> float | None:
        return None if not self.bpp else 8.0 / self.bpp


def run(img: np.ndarray, method: str, mode: str, value: float, levels: int = 4) -> Result:
    """method: 'DCT' | 'DWT';  mode: 'threshold' | 'quant';  value: поріг t або крок q."""
    if method == "DCT":
        coef, inv, groups = dct8.forward(img), dct8.inverse, dct8.position_map(img.shape)
    else:
        coef = haar.forward(img, levels)
        inv = lambda c: haar.inverse(c, levels)
        groups = haar.subband_map(img.shape, levels)

    bpp = None
    if mode == "threshold":
        kept = compress.threshold(coef, value)
    else:
        symbols, kept = compress.quantize(coef, value)
        bpp = compress.entropy_bits(symbols, groups) / img.size

    restored = np.clip(inv(kept), 0, 255)
    return Result(method, restored, kept, int(np.count_nonzero(kept)), kept.size,
                  psnr(img, restored), bpp)

"""Двовимірне дискретне вейвлет-перетворення Хаара (пірамідальне).

Один крок для пари відліків (x0, x1):
    s = (x0 + x1) / √2   — апроксимація (низькі частоти)
    d = (x0 − x1) / √2   — деталь (високі частоти)
Обернене:  x0 = (s + d) / √2,  x1 = (s − d) / √2.
2D-рівень: крок по рядках, потім по стовпцях -> підсмуги
    LL | HL
    ---+---
    LH | HH
Далі рекурсивно розкладається лише LL.
"""
import numpy as np

R2 = np.sqrt(2.0)
SHIFT = 128.0


def _split(a: np.ndarray, axis: int) -> np.ndarray:
    even = np.take(a, np.arange(0, a.shape[axis], 2), axis=axis)
    odd = np.take(a, np.arange(1, a.shape[axis], 2), axis=axis)
    return np.concatenate(((even + odd) / R2, (even - odd) / R2), axis=axis)


def _merge(a: np.ndarray, axis: int) -> np.ndarray:
    half = a.shape[axis] // 2
    s = np.take(a, np.arange(half), axis=axis)
    d = np.take(a, np.arange(half, 2 * half), axis=axis)
    out = np.empty_like(a)
    idx_even = [slice(None)] * a.ndim
    idx_odd = [slice(None)] * a.ndim
    idx_even[axis] = slice(0, None, 2)
    idx_odd[axis] = slice(1, None, 2)
    out[tuple(idx_even)] = (s + d) / R2
    out[tuple(idx_odd)] = (s - d) / R2
    return out


def forward(img: np.ndarray, levels: int = 4) -> np.ndarray:
    out = img.astype(np.float64) - SHIFT
    h, w = out.shape
    for _ in range(levels):
        block = out[:h, :w]
        block = _split(block, axis=1)       # рядки
        block = _split(block, axis=0)       # стовпці
        out[:h, :w] = block
        h //= 2
        w //= 2
    return out


def inverse(coef: np.ndarray, levels: int = 4) -> np.ndarray:
    out = coef.astype(np.float64).copy()
    H, W = out.shape
    for lev in reversed(range(levels)):
        h, w = H >> lev, W >> lev
        block = out[:h, :w]
        block = _merge(block, axis=0)
        block = _merge(block, axis=1)
        out[:h, :w] = block
    return out + SHIFT


def subband_map(shape, levels: int = 4) -> np.ndarray:
    """Номер підсмуги кожного коефіцієнта: 0 — LL, далі HL/LH/HH від найгрубшого рівня до найдрібнішого."""
    H, W = shape
    m = np.zeros(shape, dtype=int)
    band = 1
    for lev in range(levels, 0, -1):
        h, w = H >> lev, W >> lev
        m[:h, w:2 * w] = band          # HL
        m[h:2 * h, :w] = band + 1      # LH
        m[h:2 * h, w:2 * w] = band + 2 # HH
        band += 3
    return m

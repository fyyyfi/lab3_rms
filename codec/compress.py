"""Механізми зменшення кількості значущих коефіцієнтів.

threshold  — жорсткий поріг: |c| < t  ->  0.
quantize   — рівномірне квантування з кроком q: c -> round(c / q) · q.
             Цілі індекси round(c / q) — це «символи», які пішли б на ентропійне кодування.
"""
import numpy as np


def threshold(coef: np.ndarray, t: float) -> np.ndarray:
    out = coef.copy()
    out[np.abs(out) < t] = 0.0
    return out


def quantize(coef: np.ndarray, q: float):
    """Повертає (символи, відновлені коефіцієнти)."""
    symbols = np.rint(coef / q).astype(np.int64)
    return symbols, symbols * float(q)


def entropy_bits(symbols: np.ndarray, groups: np.ndarray) -> float:
    """Оцінка кількості біт для ідеального ентропійного кодера нульового порядку,
    окремо для кожної групи коефіцієнтів (позиція в блоці DCT або підсмуга DWT):
        bits = Σ_g  n_g · H_g,   H_g = −Σ p·log2 p.
    Позиції ненульових коефіцієнтів тут уже враховані (нуль — теж символ)."""
    total = 0.0
    for g in np.unique(groups):
        s = symbols[groups == g]
        _, counts = np.unique(s, return_counts=True)
        p = counts / s.size
        total += -s.size * float(np.sum(p * np.log2(p)))
    return total

"""Дискретне косинусне перетворення блоками 8×8 у матричній формі.

Ортонормована матриця DCT-II T (8×8):
    T[u, x] = c(u) · cos(π · (2x + 1) · u / 16),   c(0) = 1/√8,  c(u>0) = 1/2.
Пряме:    D = T · B · Tᵀ
Обернене: B = Tᵀ · D · T            (бо T · Tᵀ = I)
"""
import math

import numpy as np

N = 8
SHIFT = 128.0   # центрування яскравості перед DCT (0..255 -> -128..127)


def build_matrix(n: int = N) -> np.ndarray:
    t = np.empty((n, n))
    for u in range(n):
        c = math.sqrt(1 / n) if u == 0 else math.sqrt(2 / n)
        for x in range(n):
            t[u, x] = c * math.cos(math.pi * (2 * x + 1) * u / (2 * n))
    return t


T = build_matrix()
assert np.allclose(T @ T.T, np.eye(N)), "матриця DCT має бути ортонормованою"


def forward(img: np.ndarray) -> np.ndarray:
    """Зображення H×W (кратні 8) -> масив коефіцієнтів того ж розміру,
    де кожен блок 8×8 замінено своїм DCT-спектром."""
    h, w = img.shape
    src = img.astype(np.float64) - SHIFT
    out = np.empty_like(src)
    for r in range(0, h, N):
        for c in range(0, w, N):
            out[r:r + N, c:c + N] = T @ src[r:r + N, c:c + N] @ T.T
    return out


def inverse(coef: np.ndarray) -> np.ndarray:
    h, w = coef.shape
    out = np.empty_like(coef, dtype=np.float64)
    for r in range(0, h, N):
        for c in range(0, w, N):
            out[r:r + N, c:c + N] = T.T @ coef[r:r + N, c:c + N] @ T
    return out + SHIFT


def position_map(shape) -> np.ndarray:
    """Номер позиції (u·8 + v) кожного коефіцієнта всередині блоку: 0 — DC, 63 — найвища частота.
    Використовується для оцінки ентропії окремо по кожній частоті."""
    h, w = shape
    u = np.arange(h) % N
    v = np.arange(w) % N
    return u[:, None] * N + v[None, :]

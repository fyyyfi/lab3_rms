import numpy as np
from PIL import Image


def load_gray(path: str, size: int = 512) -> np.ndarray:
    """Відкрити будь-яке зображення як відтінки сірого size×size."""
    im = Image.open(path).convert("L")
    if im.size != (size, size):
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((size, size), Image.LANCZOS)
    return np.asarray(im, dtype=np.float64)


def save_gray(arr: np.ndarray, path: str) -> None:
    Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8)).save(path)

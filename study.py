"""Серія експериментів для звіту.

Для кожного методу (DCT, DWT) і кожного режиму стиснення (поріг, квантування)
перебирається кілька рівнів і зберігаються: таблиця, графіки, відновлені
зображення, збільшені фрагменти та карти похибок -> out/.

Запуск:  python study.py [зображення]
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from codec import pipeline
from codec.imgio import load_gray, save_gray

ROOT = Path(__file__).parent
OUT = ROOT / "out"
LEVELS = 4                                  # рівні розкладу Хаара
THRESHOLDS = [10, 20, 40, 80, 160]
STEPS = [8, 16, 32, 64, 128]
CROP = (slice(140, 340), slice(200, 400))   # фрагмент із літаком для збільшення
METHODS = ("DCT", "DWT")
COLORS = {"DCT": "#c0392b", "DWT": "#2471a3"}


def experiment(img):
    rows, images = [], {}
    for mode, values in (("threshold", THRESHOLDS), ("quant", STEPS)):
        for v in values:
            for m in METHODS:
                r = pipeline.run(img, m, mode, v, LEVELS)
                images[(mode, v, m)] = r.restored
                rows.append(dict(mode=mode, value=v, method=m, nonzero=r.nonzero,
                                 share_pct=round(100 * r.share, 2), cr_coef=round(r.cr_coef, 1),
                                 bpp=None if r.bpp is None else round(r.bpp, 3),
                                 cr_bits=None if r.cr_bits is None else round(r.cr_bits, 1),
                                 psnr=round(r.psnr, 2)))
    return rows, images


def write_tables(rows):
    with open(OUT / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    lines = []
    for mode, title, par in (("threshold", "Жорсткий поріг", "t"), ("quant", "Рівномірне квантування", "q")):
        lines.append(f"\n**{title}**\n")
        if mode == "threshold":
            lines.append(f"| {par} | DCT: ненульових | DCT: % | DCT: PSNR | DWT: ненульових | DWT: % | DWT: PSNR |")
            lines.append("|---|---|---|---|---|---|---|")
        else:
            lines.append(f"| {par} | DCT: ненульових | DCT: біт/піксель | DCT: PSNR | DWT: ненульових | DWT: біт/піксель | DWT: PSNR |")
            lines.append("|---|---|---|---|---|---|---|")
        for v in (THRESHOLDS if mode == "threshold" else STEPS):
            d = next(r for r in rows if r["mode"] == mode and r["value"] == v and r["method"] == "DCT")
            w = next(r for r in rows if r["mode"] == mode and r["value"] == v and r["method"] == "DWT")
            mid = "share_pct" if mode == "threshold" else "bpp"
            lines.append(f"| {v} | {d['nonzero']} | {d[mid]} | {d['psnr']} | {w['nonzero']} | {w[mid]} | {w['psnr']} |")
    (OUT / "results.md").write_text("\n".join(lines) + "\n")


def plots(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for m in METHODS:
        for mode, ls, lbl in (("threshold", "-", "поріг"), ("quant", "--", "квантування")):
            rr = [r for r in rows if r["method"] == m and r["mode"] == mode]
            axes[0].plot([r["share_pct"] for r in rr], [r["psnr"] for r in rr], ls, marker="o",
                         color=COLORS[m], label=f"{m}, {lbl}")
        rr = [r for r in rows if r["method"] == m and r["mode"] == "quant"]
        axes[1].plot([r["bpp"] for r in rr], [r["psnr"] for r in rr], marker="s",
                     color=COLORS[m], label=m)
    axes[0].set_xscale("log"); axes[0].set_xlabel("Ненульових коефіцієнтів, % (лог.)")
    axes[1].set_xscale("log"); axes[1].set_xlabel("Оцінка ентропії, біт/піксель (лог.)")
    for ax, t in zip(axes, ("PSNR від частки коефіцієнтів", "PSNR від оцінки розміру (квантування)")):
        ax.set_ylabel("PSNR, дБ"); ax.set_title(t); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "psnr_curves.png", dpi=130); plt.close(fig)


def comparison(img, rows, images, mode, values, name):
    """Original | DCT | DWT для кількох рівнів + збільшений фрагмент + карта похибки."""
    fig, axes = plt.subplots(len(values), 5, figsize=(17, 3.6 * len(values)))
    for i, v in enumerate(values):
        d, w = images[(mode, v, "DCT")], images[(mode, v, "DWT")]
        pd = next(r for r in rows if r["mode"] == mode and r["value"] == v and r["method"] == "DCT")
        pw = next(r for r in rows if r["mode"] == mode and r["value"] == v and r["method"] == "DWT")
        par = "t" if mode == "threshold" else "q"
        panels = [
            (img, "Оригінал", "gray"),
            (d, f"DCT, {par}={v}: {pd['psnr']} дБ", "gray"),
            (w, f"DWT, {par}={v}: {pw['psnr']} дБ", "gray"),
            (np.hstack([d[CROP], np.full((d[CROP].shape[0], 4), 255), w[CROP]]), "фрагмент: DCT | DWT", "gray"),
            (np.hstack([np.abs(img - d)[CROP], np.zeros((d[CROP].shape[0], 4)), np.abs(img - w)[CROP]]),
             "|похибка| ×4: DCT | DWT", "magma"),
        ]
        for ax, (im, t, cmap) in zip(axes[i], panels):
            vmax = 64 if cmap == "magma" else 255
            ax.imshow(im, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_title(t, fontsize=10); ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / name, dpi=100); plt.close(fig)


def main(path):
    OUT.mkdir(exist_ok=True)
    img = load_gray(path)
    save_gray(img, OUT / "original.png")
    rows, images = experiment(img)
    for (mode, v, m), im in images.items():
        save_gray(im, OUT / f"{m.lower()}_{'t' if mode == 'threshold' else 'q'}{v}.png")
    write_tables(rows); plots(rows)
    comparison(img, rows, images, "threshold", [20, 80, 160], "compare_threshold.png")
    comparison(img, rows, images, "quant", [16, 64, 128], "compare_quant.png")
    print((OUT / "results.md").read_text())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "images" / "7.1.02.tiff"))

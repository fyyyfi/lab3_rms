"""Інтерактивне порівняння Оригінал | DCT | DWT.

Запуск:  python gui.py [зображення]
"""
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
    QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from codec import pipeline
from codec.imgio import load_gray, save_gray

ROOT = Path(__file__).parent
DEFAULT = ROOT / "images" / "7.1.02.tiff"
VIEW = 440
MODES = {"Жорсткий поріг t": "threshold", "Квантування з кроком q": "quant"}


def to_pixmap(arr: np.ndarray, error_map: bool = False) -> QPixmap:
    if error_map:   # |похибка| 0..64 -> 0..255, щоб дрібні відхилення було видно
        arr = np.clip(arr * 4, 0, 255)
    a = np.ascontiguousarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))
    img = QImage(a.data, a.shape[1], a.shape[0], a.strides[0], QImage.Format_Grayscale8).copy()
    return QPixmap.fromImage(img).scaled(VIEW, VIEW, Qt.KeepAspectRatio, Qt.FastTransformation)


class Panel(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.pic = QLabel(); self.pic.setFixedSize(VIEW, VIEW); self.pic.setAlignment(Qt.AlignCenter)
        self.info = QLabel("—"); self.info.setTextFormat(Qt.RichText)
        lay = QVBoxLayout(self); lay.addWidget(self.pic); lay.addWidget(self.info)


class Window(QMainWindow):
    def __init__(self, path: Path):
        super().__init__()
        self.setWindowTitle("Стиснення зображень: DCT 8×8 та вейвлет Хаара")
        self.img, self.res = None, {}

        # --- панель параметрів ------------------------------------------
        self.mode = QComboBox(); self.mode.addItems(MODES)
        self.value = QSlider(Qt.Horizontal); self.value.setRange(1, 255); self.value.setValue(32)
        self.value_box = QSpinBox(); self.value_box.setRange(1, 255); self.value_box.setValue(32)
        self.value.valueChanged.connect(self.value_box.setValue)
        self.value_box.valueChanged.connect(self.value.setValue)
        self.levels = QSpinBox(); self.levels.setRange(1, 6); self.levels.setValue(4)
        self.err = QCheckBox("Показати карту похибки |оригінал − результат| ×4")
        btn_open = QPushButton("Відкрити зображення…"); btn_open.clicked.connect(self.open)
        btn_save = QPushButton("Зберегти результати…"); btn_save.clicked.connect(self.save)

        val_row = QHBoxLayout(); val_row.addWidget(self.value, 1); val_row.addWidget(self.value_box)
        form = QFormLayout()
        form.addRow("Механізм стиснення:", self.mode)
        form.addRow("Параметр (t або q):", val_row)
        form.addRow("Рівнів розкладу Хаара:", self.levels)
        form.addRow("", self.err)
        btns = QHBoxLayout(); btns.addWidget(btn_open); btns.addWidget(btn_save); btns.addStretch()
        form.addRow(btns)

        self.panels = {k: Panel(t) for k, t in (("orig", "Оригінал"), ("DCT", "DCT, блоки 8×8"),
                                                 ("DWT", "DWT, вейвлет Хаара"))}
        grid = QGridLayout()
        for i, p in enumerate(self.panels.values()):
            grid.addWidget(p, 0, i)
        self.summary = QLabel(); self.summary.setWordWrap(True)

        root = QWidget(); lay = QVBoxLayout(root)
        lay.addLayout(form); lay.addLayout(grid); lay.addWidget(self.summary)
        self.setCentralWidget(root)

        # перерахунок із невеликою затримкою, щоб повзунок не «гальмував»
        self.timer = QTimer(self, singleShot=True, interval=120, timeout=self.recompute)
        for sig in (self.mode.currentIndexChanged, self.value.valueChanged, self.levels.valueChanged):
            sig.connect(lambda *_: self.timer.start())
        self.err.toggled.connect(self.redraw)
        self.load(path)

    # ----------------------------------------------------------------------
    def load(self, path: Path):
        try:
            self.img = load_gray(str(path))
        except Exception as e:   # noqa: BLE001 — показуємо будь-яку помилку читання користувачу
            QMessageBox.critical(self, "Помилка", f"Не вдалося відкрити {path}:\n{e}")
            return
        self.path = Path(path)
        self.panels["orig"].info.setText(f"{self.path.name}<br>{self.img.shape[1]}×{self.img.shape[0]}, "
                                         f"{self.img.size} коефіцієнтів")
        self.recompute()

    def open(self):
        p, _ = QFileDialog.getOpenFileName(self, "Зображення", str(ROOT / "images"),
                                           "Зображення (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if p:
            self.load(Path(p))

    def recompute(self):
        if self.img is None:
            return
        mode, v, lv = MODES[self.mode.currentText()], self.value.value(), self.levels.value()
        self.res = {m: pipeline.run(self.img, m, mode, v, lv) for m in ("DCT", "DWT")}
        for m, r in self.res.items():
            lines = [f"<b>PSNR = {r.psnr:.2f} дБ</b>",
                     f"ненульових: {r.nonzero} з {r.total} ({100 * r.share:.2f} %)",
                     f"CR за коефіцієнтами ≈ {r.cr_coef:.1f} : 1"]
            if r.bpp is not None:
                lines.append(f"ентропія ≈ {r.bpp:.3f} біт/піксель (CR ≈ {r.cr_bits:.1f} : 1)")
            self.panels[m].info.setText("<br>".join(lines))
        d, w = self.res["DCT"], self.res["DWT"]
        better = "DWT" if w.psnr > d.psnr else "DCT"
        self.summary.setText(
            f"Параметр {'t' if mode == 'threshold' else 'q'} = {v}. Кращий PSNR: {better} "
            f"(різниця {abs(w.psnr - d.psnr):.2f} дБ). DWT залишив {w.nonzero} коефіцієнтів, "
            f"DCT — {d.nonzero}.")
        self.redraw()

    def redraw(self):
        if self.img is None or not self.res:
            return
        e = self.err.isChecked()
        self.panels["orig"].pic.setPixmap(to_pixmap(self.img))
        for m, r in self.res.items():
            self.panels[m].pic.setPixmap(to_pixmap(np.abs(self.img - r.restored) if e else r.restored, e))

    def save(self):
        folder = QFileDialog.getExistingDirectory(self, "Куди зберегти", str(ROOT))
        if not folder or not self.res:
            return
        tag = f"{MODES[self.mode.currentText()]}_{self.value.value()}"
        for m, r in self.res.items():
            save_gray(r.restored, Path(folder) / f"{self.path.stem}_{m.lower()}_{tag}.png")
        QMessageBox.information(self, "Збережено", f"Файли збережено в {folder}")


def main():
    app = QApplication(sys.argv)
    w = Window(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

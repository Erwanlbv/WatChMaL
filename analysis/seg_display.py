"""
Segmentation event display: the ResNet's view (CNN image), whole event and rings side by side.

    python -m analysis.seg_display FILE.h5 --mapping MAPPING.npz [--event N] [--light]

One window (PyQt6 + pyqtgraph). Each panel is the CNN image of N. Prouse's event display
(`pmt_image_positions`: one pixel per PMT, row 0 at the bottom): the first panel is the
whole event, rebuilt as the sum of the rings, and the next panels are the rings. All
panels share one colour scale. The values are the raw charge / time of the file: no
dataset transform or normalisation is applied. The colour scale starts in log10, as the
LogNorm default of N. Prouse's display.

Keys:  <- / ->  previous / next event    C / T  charge / time    L  log scale
       D  dark / light theme             Esc    quit
Moving the mouse over a pixel shows the PMT id, charge and time in the status bar.

The theme is applied to every element (window, panels, titles, colour bar, empty and unhit
pixels), so both themes are readable. The default follows the system setting.
"""
import argparse
import sys

import numpy as np

from analysis.load_seg_h5_data import PDG_NAMES, LoadSegData

THEMES = {
    "dark": dict(bg="#0f1117", surface="#171a22", border="#2a2f3a", text="#d4d8e0", muted="#7c8494",
                 unhit=(52, 56, 66)),
    "light": dict(bg="#f7f8fa", surface="#ffffff", border="#d3d8e0", text="#1f2633", muted="#5f6877",
                  unhit=(212, 216, 224)),
}
PDG_SYMBOLS = {11: "e⁻", -11: "e⁺", 13: "μ⁻", -13: "μ⁺", 22: "γ", 111: "π⁰", 211: "π⁺", -211: "π⁻"}


# ── pure numpy part (tested without Qt) ──────────────────────────────────────────────────

def load_mapping(mapping_file_path):
    """(n_pmts, 2) int array of (row, column) per 0-based PMT id, and the image shape."""
    rc = np.load(mapping_file_path)["pmt_image_positions"].astype(int)
    return rc, (int(rc[:, 0].max()) + 1, int(rc[:, 1].max()) + 1)


def cnn_image(hit_pmt, hit_charge, hit_time, mapping_rc, shape):
    """
    Charge and time images (rows, columns), NaN where no PMT was hit.

    A PMT hit several times (several triggers) gets the sum of its charges and its
    earliest time.
    """
    q_img = np.full(shape, np.nan)
    t_img = np.full(shape, np.nan)
    if len(hit_pmt):
        pmt, inv = np.unique(hit_pmt, return_inverse=True)
        q = np.zeros(pmt.size)
        np.add.at(q, inv.reshape(-1), hit_charge)
        t = np.full(pmt.size, np.inf)
        np.minimum.at(t, inv.reshape(-1), hit_time)
        r, c = mapping_rc[pmt, 0], mapping_rc[pmt, 1]
        q_img[r, c], t_img[r, c] = q, t
    return q_img, t_img


def pmt_mask(mapping_rc, shape):
    """True on the pixels that hold a PMT."""
    m = np.zeros(shape, dtype=bool)
    m[mapping_rc[:, 0], mapping_rc[:, 1]] = True
    return m


def to_rgba(values, mask, levels, cmap_lut, theme):
    """RGBA image: colour map on hit pixels, theme colour for unhit PMTs, transparent elsewhere."""
    rgba = np.empty(values.shape + (4,), dtype=np.uint8)
    rgba[...] = 0                      # no PMT: transparent, the theme background shows through
    rgba[mask] = (*theme["unhit"], 255)
    hit = np.isfinite(values)
    lo, hi = levels
    x = np.clip((values[hit] - lo) / (hi - lo if hi > lo else 1.0), 0, 1)
    rgba[hit] = cmap_lut[(x * (len(cmap_lut) - 1)).astype(int)]
    return rgba


def ring_title(data, i, k):
    pdg = int(data.ring_labels[i][k])
    a = data.ring_attrs[i][k]
    title = f"ring {k} · {PDG_SYMBOLS.get(pdg, PDG_NAMES.get(pdg, pdg))} {data.ring_energies[i][k]:.0f} MeV"
    if "dominant_dice" in a:
        title += f" · dice {a['dominant_dice']:.2f}"
    return title


# ── Qt window ────────────────────────────────────────────────────────────────────────────

def run(data, mapping_file_path, event=0, theme=None):
    import pyqtgraph as pg
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QKeySequence, QShortcut
    from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow, QPushButton,
                                 QSpinBox, QVBoxLayout, QWidget)

    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
    app = QApplication.instance() or QApplication(sys.argv)
    if theme is None:
        scheme = app.styleHints().colorScheme()
        theme = "light" if scheme == Qt.ColorScheme.Light else "dark"
    mapping_rc, shape = load_mapping(mapping_file_path)
    mask = pmt_mask(mapping_rc, shape)
    cmap = pg.colormap.get("plasma")
    lut = cmap.getLookupTable(nPts=256, alpha=True)
    print(f"[SegDisplay] mapping {mapping_file_path}: image {shape[0]} x {shape[1]}, {len(mapping_rc)} PMTs")

    class Window(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"Segmentation event display — {data.h5path.rsplit('/', 1)[-1]}")
            self.i, self.metric, self.log, self.theme = event, "charge", True, theme
            self.images = {}

            # navigation bar
            self.btn_prev, self.btn_next = QPushButton("←  Prev"), QPushButton("Next  →")
            self.lbl_event = QLabel()
            self.spin = QSpinBox()
            self.spin.setRange(0, len(data) - 1)
            btn_go = QPushButton("Go")
            nav = QHBoxLayout()
            for w in (self.btn_prev, self.lbl_event, self.btn_next):
                nav.addWidget(w)
            nav.addStretch(1)
            nav.addWidget(QLabel("event"))
            nav.addWidget(self.spin)
            nav.addWidget(btn_go)

            # panels
            self.view = pg.GraphicsLayoutWidget()

            # toggle bar
            self.btn_q, self.btn_t = QPushButton("Charge  [C]"), QPushButton("Time  [T]")
            self.btn_log, self.btn_theme = QPushButton(), QPushButton()
            bar = QHBoxLayout()
            for w in (self.btn_q, self.btn_t, self.btn_log):
                bar.addWidget(w)
            bar.addStretch(1)
            bar.addWidget(self.btn_theme)

            lay = QVBoxLayout()
            lay.addLayout(nav)
            lay.addWidget(self.view, 1)
            lay.addLayout(bar)
            root = QWidget()
            root.setLayout(lay)
            self.setCentralWidget(root)
            self.status = self.statusBar()

            self.btn_prev.clicked.connect(lambda: self.go(self.i - 1))
            self.btn_next.clicked.connect(lambda: self.go(self.i + 1))
            btn_go.clicked.connect(lambda: self.go(self.spin.value()))
            self.btn_q.clicked.connect(lambda: self.set_metric("charge"))
            self.btn_t.clicked.connect(lambda: self.set_metric("time"))
            self.btn_log.clicked.connect(self.toggle_log)
            self.btn_theme.clicked.connect(self.toggle_theme)
            for key, fn in (("Left", lambda: self.go(self.i - 1)), ("Right", lambda: self.go(self.i + 1)),
                            ("P", lambda: self.go(self.i - 1)), ("N", lambda: self.go(self.i + 1)),
                            ("C", lambda: self.set_metric("charge")), ("T", lambda: self.set_metric("time")),
                            ("L", self.toggle_log), ("D", self.toggle_theme), ("Esc", self.close)):
                QShortcut(QKeySequence(key), self, activated=fn)
            self.view.scene().sigMouseMoved.connect(self.on_mouse)
            self.resize(1500, 820)
            self.apply_theme()
            self.go(event)

        # data
        def go(self, i):
            self.i = int(np.clip(i, 0, len(data) - 1))
            n = int(data.n_rings[self.i])
            panels = [(None, "whole event (sum of rings)")] + [(k, ring_title(data, self.i, k)) for k in range(n)]
            self.panels = []
            for k, title in panels:
                data.load_hit_info(self.i, ring=k)
                q, t = cnn_image(data.hit_pmt, data.hit_charge, data.hit_time, mapping_rc, shape)
                self.panels.append((title, q, t))
            self.spin.setValue(self.i)
            self.lbl_event.setText(f"  {data.event_names[self.i]}   ({self.i + 1} / {len(data)})   "
                                   f"{n} ring{'s' if n != 1 else ''}  ")
            print(f"[SegDisplay] {data.event_names[self.i]}: {n} rings, "
                  f"{int(np.isfinite(self.panels[0][1]).sum())} hit PMTs in the whole event")
            self.render()

        def values(self, panel):
            v = panel[1] if self.metric == "charge" else panel[2]
            if self.log:
                with np.errstate(divide="ignore", invalid="ignore"):
                    v = np.where(v > 0, np.log10(v), np.nan)
            return v

        def label(self):
            unit = "charge [p.e.]" if self.metric == "charge" else "time [ns]"
            return f"log10 {unit}" if self.log else unit

        # drawing
        def render(self):
            th = THEMES[self.theme]
            self.view.clear()
            self.images = {}
            vals = [self.values(p) for p in self.panels]
            finite = np.concatenate([v[np.isfinite(v)] for v in vals])
            levels = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
            for col, ((title, _, _), v) in enumerate(zip(self.panels, vals)):
                plot = self.view.addPlot(row=0, col=col)
                plot.setTitle(title, color=th["text"], size="10pt")
                plot.hideAxis("left")
                plot.hideAxis("bottom")
                plot.setAspectLocked(True)
                plot.setMenuEnabled(False)
                plot.hideButtons()
                img = pg.ImageItem(to_rgba(v, mask, levels, lut, th))
                plot.addItem(img)
                plot.setRange(xRange=(0, shape[1]), yRange=(0, shape[0]), padding=0.01)
                self.images[plot.vb] = col
                if col:
                    plot.setXLink(self.view.getItem(0, 0))
                    plot.setYLink(self.view.getItem(0, 0))
            bar = pg.ColorBarItem(values=levels, colorMap=cmap, interactive=False, label=self.label(),
                                  width=14)
            bar.axis.setPen(pg.mkPen(th["muted"]))
            bar.axis.setTextPen(pg.mkPen(th["text"]))
            bar.axis.label.setHtml(f"<span style='color:{th['text']}'>{self.label()}</span>")
            self.view.addItem(bar, row=0, col=len(self.panels))
            self.btn_q.setEnabled(self.metric != "charge")
            self.btn_t.setEnabled(self.metric != "time")
            self.btn_log.setText(f"Log scale: {'on' if self.log else 'off'}  [L]")

        def apply_theme(self):
            th = THEMES[self.theme]
            app.setStyleSheet(f"""
                QWidget {{ background:{th['bg']}; color:{th['text']}; font-size:12px; }}
                QPushButton {{ background:{th['surface']}; border:1px solid {th['border']};
                               border-radius:5px; padding:5px 12px; }}
                QPushButton:hover {{ border-color:{th['muted']}; }}
                QPushButton:disabled {{ color:{th['muted']}; background:{th['bg']}; }}
                QSpinBox {{ background:{th['surface']}; border:1px solid {th['border']}; padding:2px 4px; }}
                QStatusBar {{ background:{th['surface']}; color:{th['muted']};
                              border-top:1px solid {th['border']}; }}""")
            self.view.setBackground(th["bg"])
            other = "light" if self.theme == "dark" else "dark"
            self.btn_theme.setText(f"{other.capitalize()} theme  [D]")

        # actions
        def set_metric(self, m):
            self.metric = m
            self.render()

        def toggle_log(self):
            self.log = not self.log
            self.render()

        def toggle_theme(self):
            self.theme = "light" if self.theme == "dark" else "dark"
            self.apply_theme()
            self.render()

        def on_mouse(self, pos):
            for vb, col in self.images.items():
                if vb.sceneBoundingRect().contains(pos):
                    p = vb.mapSceneToView(pos)
                    r, c = int(np.floor(p.y())), int(np.floor(p.x()))
                    if 0 <= r < shape[0] and 0 <= c < shape[1] and mask[r, c]:
                        pmt = int(np.flatnonzero((mapping_rc[:, 0] == r) & (mapping_rc[:, 1] == c))[0])
                        _, q, t = self.panels[col]
                        hit = (f"q = {q[r, c]:.3g} p.e.   t = {t[r, c]:.1f} ns" if np.isfinite(q[r, c])
                               else "not hit")
                        self.status.showMessage(f"{self.panels[col][0]}   PMT {pmt} (row {r}, col {c})   {hit}")
                    else:
                        self.status.clearMessage()
                    return

    win = Window()
    win.show()
    return app, win


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h5file", help="ring-segmented file (event_XXXXXX/ring_K)")
    p.add_argument("--mapping", required=True, help="npz with pmt_image_positions (CNN image mapping)")
    p.add_argument("-e", "--event", type=int, default=0, help="first event shown (index in the file)")
    p.add_argument("--tube-id-base", type=int, choices=(0, 1), default=None, help="default: detect")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--light", action="store_const", const="light", dest="theme")
    g.add_argument("--dark", action="store_const", const="dark", dest="theme")
    a = p.parse_args(argv)
    data = LoadSegData(a.h5file, tube_id_base=a.tube_id_base)
    app, _win = run(data, a.mapping, a.event, a.theme)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

"""
Segmentation event display: the ResNet's view (CNN image), whole event and rings side by side.

    python -m analysis.seg_display FILE.h5 --mapping MAPPING.npz [--truth WCSIM.h5] [--event N]
                                   [--view all|true|reco|custom] [--light]

One window (PyQt6 + pyqtgraph). Each panel is the CNN image of N. Prouse's event display
(`pmt_image_positions`: one pixel per PMT, row 0 at the bottom). Without --truth the first
panel is the whole event rebuilt as the sum of the rings, and the next panels are the rings. All
panels share one colour scale. The values are the raw charge / time of the file: no
dataset transform or normalisation is applied. The colour scale starts in log10, as the
LogNorm default of N. Prouse's display.

With --truth (the WCSim file of the `src_file` attribute; the segmented file itself holds
only the predicted rings), the first panel is the real unsegmented event, the first row adds
one panel per true particle (its digitized hits, `primary_track_id`), and the second row
holds the reco rings, each under the true particle it is matched to.

Views:  All (everything)   True (event + true particles)   Reco (event + rings)
        Custom (one toggle per panel, to keep or hide it). Double-clicking a panel shows it
        alone (to zoom on it); double-clicking it again goes back.

Keys:  <- / ->  previous / next event    C / T  charge / time    L  log scale
       1 2 3 4  all / true / reco / custom    D  dark / light theme    Esc  quit
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
                 unhit=(52, 56, 66), accent="#2c3b5e", accent_border="#5b7bc4"),
    "light": dict(bg="#f7f8fa", surface="#ffffff", border="#d3d8e0", text="#1f2633", muted="#5f6877",
                  unhit=(212, 216, 224), accent="#dce6f8", accent_border="#5b7bc4"),
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
    kind = "reco ring" if a.get("source") == "pred" else "ring"
    title = f"{kind} {k} · {PDG_SYMBOLS.get(pdg, PDG_NAMES.get(pdg, pdg))} {data.ring_energies[i][k]:.0f} MeV"
    if "dominant_dice" in a:
        title += f"<br>dice {a['dominant_dice']:.2f}"
        if "matched_parent_track_id" in a:
            title += f" · ← track {int(a['matched_parent_track_id'])}"
    return title


def true_title(p):
    """Title of a true-particle panel; `p` is one entry of LoadSegData.true_parents."""
    sym = PDG_SYMBOLS.get(p["pdg"], PDG_NAMES.get(p["pdg"], p["pdg"]))
    ring = f"→ ring {p['ring']}" if p["ring"] is not None else "· no ring"
    return f"true · {sym} {p['energy']:.0f} MeV<br>track {p['track_id']} · {p['pe']:.0f} p.e. {ring}"


def short_name(key, title):
    """Label of a panel toggle in the custom view."""
    if key[0] == "event":
        return "Event"
    if key[0] == "true":
        return "True " + title.split("<br>")[0].split(" · ")[1].rsplit(" ", 1)[0]      # "True μ⁻ 1563"
    return f"Ring {key[1]}"


# ── panel layout (tested without Qt) ─────────────────────────────────────────────────────

MODES = ("all", "true", "reco", "custom")


def panel_grid(n_true, ring_match, n_rings):
    """
    Grid position of every panel in the "all" view, as {key: (row, col)}.

    Row 0: the whole event, then one panel per true particle. Row 1: the reco rings, each under
    the true particle it is matched to (`ring_match[k]` = index of that particle, or None); a
    ring with no match, or whose particle already has a ring below it, gets an extra column.
    Without truth (n_true = 0) the rings go on row 0, after the event.
    """
    pos = {("event",): (0, 0)}
    for j in range(n_true):
        pos[("true", j)] = (0, 1 + j)
    row = 1 if n_true else 0
    used, extra = set(), 1 + n_true
    for k in range(n_rings):
        j = ring_match[k] if k < len(ring_match) else None
        if n_true and j is not None and j not in used:
            used.add(j)
            pos[("reco", k)] = (row, 1 + j)
        else:
            pos[("reco", k)] = (row, extra if n_true else 1 + k)
            extra += 1 if n_true else 0
    return pos


def visible_keys(keys, mode, hidden=()):
    """Panels shown by a view mode: all / true (event + true particles) / reco (event + rings) / custom."""
    if mode == "all":
        return [k for k in keys]
    if mode == "true":
        return [k for k in keys if k[0] in ("event", "true")]
    if mode == "reco":
        return [k for k in keys if k[0] in ("event", "reco")]
    return [k for k in keys if k not in hidden]


def compact(pos, keys):
    """
    Positions of `keys` only, with the empty columns removed. The two rows are kept only when
    some column still holds two panels (a true particle above its ring); otherwise every panel
    goes on one row.
    """
    cols = sorted({pos[k][1] for k in keys})
    two_rows = len({pos[k][1] for k in keys}) < len(keys)
    rows = sorted({pos[k][0] for k in keys}) if two_rows else None
    return {k: (rows.index(pos[k][0]) if two_rows else 0, cols.index(pos[k][1])) for k in keys}


# ── Qt window ────────────────────────────────────────────────────────────────────────────

def run(data, mapping_file_path, event=0, theme=None, mode="all"):
    import pyqtgraph as pg
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QKeySequence, QShortcut
    from PyQt6.QtWidgets import (QApplication, QButtonGroup, QHBoxLayout, QLabel, QMainWindow,
                                 QPushButton, QSpinBox, QVBoxLayout, QWidget)

    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
    app = QApplication.instance() or QApplication(sys.argv)
    if theme is None:
        scheme = app.styleHints().colorScheme()
        theme = "light" if scheme == Qt.ColorScheme.Light else "dark"
    if mode == "true" and not data.has_truth:
        print("[SegDisplay] no truth file: 'true' view unavailable, starting on 'all'")
        mode = "all"
    mapping_rc, shape = load_mapping(mapping_file_path)
    mask = pmt_mask(mapping_rc, shape)
    cmap = pg.colormap.get("plasma")
    lut = cmap.getLookupTable(nPts=256, alpha=True)
    print(f"[SegDisplay] mapping {mapping_file_path}: image {shape[0]} x {shape[1]}, {len(mapping_rc)} PMTs")
    print(f"[SegDisplay] truth: {data.truth.path if data.has_truth else 'none (rings only)'}")

    def toggle_button(text, group=None):
        b = QPushButton(text)
        b.setCheckable(True)
        if group is not None:
            group.addButton(b)
        return b

    class Window(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"Segmentation event display — {data.h5path.rsplit('/', 1)[-1]}")
            self.i, self.metric, self.log, self.theme, self.mode = event, "charge", True, theme, mode
            self.hidden, self.prev_mode, self.images = set(), "all", {}

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
            nav.addWidget(QLabel("view"))
            self.mode_group = QButtonGroup(self)
            self.btn_mode = {}
            for k, (m, txt) in enumerate((("all", "All"), ("true", "True"), ("reco", "Reco"),
                                          ("custom", "Custom"))):
                b = toggle_button(f"{txt}  [{k + 1}]", self.mode_group)
                b.clicked.connect(lambda _=False, m=m: self.set_mode(m))
                nav.addWidget(b)
                self.btn_mode[m] = b
            if not data.has_truth:
                self.btn_mode["true"].setEnabled(False)
                self.btn_mode["true"].setToolTip("needs the WCSim truth file (--truth)")
            nav.addSpacing(16)
            nav.addWidget(QLabel("event"))
            nav.addWidget(self.spin)
            nav.addWidget(btn_go)

            # custom view: one toggle per panel (rebuilt for every event)
            self.custom_widget = QWidget()
            self.custom_bar = QHBoxLayout(self.custom_widget)
            self.custom_bar.setContentsMargins(0, 0, 0, 0)

            # panels
            self.view = pg.GraphicsLayoutWidget()

            # toggle bar
            metric_group = QButtonGroup(self)
            self.btn_q = toggle_button("Charge  [C]", metric_group)
            self.btn_t = toggle_button("Time  [T]", metric_group)
            self.btn_log = toggle_button("Log scale  [L]")
            self.btn_theme = QPushButton()
            bar = QHBoxLayout()
            for w in (self.btn_q, self.btn_t, self.btn_log):
                bar.addWidget(w)
            bar.addStretch(1)
            bar.addWidget(QLabel("double-click a panel: show it alone / back"))
            bar.addSpacing(12)
            bar.addWidget(self.btn_theme)

            lay = QVBoxLayout()
            lay.addLayout(nav)
            lay.addWidget(self.custom_widget)
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
            keys = [("Left", lambda: self.go(self.i - 1)), ("Right", lambda: self.go(self.i + 1)),
                    ("P", lambda: self.go(self.i - 1)), ("N", lambda: self.go(self.i + 1)),
                    ("C", lambda: self.set_metric("charge")), ("T", lambda: self.set_metric("time")),
                    ("L", self.toggle_log), ("D", self.toggle_theme), ("Esc", self.close)]
            keys += [(str(k + 1), lambda m=m: self.set_mode(m)) for k, m in enumerate(MODES)]
            for key, fn in keys:
                QShortcut(QKeySequence(key), self, activated=fn)
            self.view.scene().sigMouseMoved.connect(self.on_mouse)
            self.view.scene().sigMouseClicked.connect(self.on_click)
            self.resize(1500, 900)
            self.apply_theme()
            self.go(event)

        # data
        def go(self, i):
            self.i = int(np.clip(i, 0, len(data) - 1))
            n = int(data.n_rings[self.i])
            parents = data.true_parents(self.i)
            self.panels = {}
            if parents:
                data.load_true_hit_info(self.i)
                ev_title = (f"whole event (WCSim)<br>{data.hit_charge.sum():.0f} p.e. · "
                            f"dark noise {data.noise_pe(self.i):.0f}")
            else:
                data.load_hit_info(self.i)
                ev_title = "whole event<br>(sum of rings)"
            self.panels[("event",)] = (ev_title, *cnn_image(data.hit_pmt, data.hit_charge, data.hit_time,
                                                            mapping_rc, shape))
            for j, p in enumerate(parents):
                data.load_true_hit_info(self.i, p["track_id"])
                self.panels[("true", j)] = (true_title(p), *cnn_image(data.hit_pmt, data.hit_charge,
                                                                      data.hit_time, mapping_rc, shape))
            for k in range(n):
                data.load_hit_info(self.i, ring=k)
                self.panels[("reco", k)] = (ring_title(data, self.i, k), *cnn_image(
                    data.hit_pmt, data.hit_charge, data.hit_time, mapping_rc, shape))
            track_rank = {p["track_id"]: j for j, p in enumerate(parents)}
            match = [track_rank.get(int(a.get("matched_parent_track_id", -1)))
                     for a in data.ring_attrs[self.i]]
            self.grid = panel_grid(len(parents), match, n)
            self.spin.setValue(self.i)
            truth = f"{len(parents)} true particle{'s' if len(parents) != 1 else ''}   " if parents else ""
            self.lbl_event.setText(f"  {data.event_names[self.i]}   ({self.i + 1} / {len(data)})   "
                                   f"{truth}{n} ring{'s' if n != 1 else ''}  ")
            print(f"[SegDisplay] {data.event_names[self.i]}: {len(parents)} true particles, {n} rings, "
                  f"{int(np.isfinite(self.panels[('event',)][1]).sum())} hit PMTs in the whole event")
            self.build_custom_bar()
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
        def build_custom_bar(self):
            while self.custom_bar.count():
                w = self.custom_bar.takeAt(0).widget()
                if w is not None:
                    w.deleteLater()
            self.custom_bar.addWidget(QLabel("panels"))
            for key in self.panels:
                b = toggle_button(short_name(key, self.panels[key][0]))
                b.setChecked(key not in self.hidden)
                b.clicked.connect(lambda on, key=key: self.toggle_panel(key, on))
                self.custom_bar.addWidget(b)
            self.custom_bar.addStretch(1)

        def render(self):
            th = THEMES[self.theme]
            self.view.clear()
            self.images = {}
            keys = visible_keys(list(self.panels), self.mode, self.hidden)
            if not keys:                       # custom view with every panel hidden
                keys = [("event",)]
            pos = compact(self.grid, keys)
            vals = {k: self.values(self.panels[k]) for k in keys}
            finite = np.concatenate([v[np.isfinite(v)] for v in vals.values()])
            levels = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
            for key in keys:
                r, c = pos[key]
                plot = self.view.addPlot(row=r, col=c)
                plot.setTitle(self.panels[key][0], color=th["text"], size="9pt")
                plot.hideAxis("left")
                plot.hideAxis("bottom")
                plot.setAspectLocked(True)
                plot.setMenuEnabled(False)
                plot.hideButtons()
                plot.addItem(pg.ImageItem(to_rgba(vals[key], mask, levels, lut, th)))
                plot.setRange(xRange=(0, shape[1]), yRange=(0, shape[0]), padding=0.01)
                self.images[plot.vb] = key
                plot.vb.sigRangeChangedManually.connect(lambda _m, vb=plot.vb: self.sync_zoom(vb))
            n_rows = max(r for r, _ in pos.values()) + 1
            n_cols = max(c for _, c in pos.values()) + 1
            grid = self.view.ci.layout
            for c in range(n_cols):              # equal panel widths, whatever the title lengths
                grid.setColumnStretchFactor(c, 1)
                grid.setColumnPreferredWidth(c, 1)
            grid.setColumnStretchFactor(n_cols, 0)
            bar = pg.ColorBarItem(values=levels, colorMap=cmap, interactive=False, label=self.label(),
                                  width=14)
            bar.axis.setPen(pg.mkPen(th["muted"]))
            bar.axis.setTextPen(pg.mkPen(th["text"]))
            bar.axis.label.setHtml(f"<span style='color:{th['text']}'>{self.label()}</span>")
            self.view.addItem(bar, row=0, col=n_cols, rowspan=n_rows)
            self.btn_q.setChecked(self.metric == "charge")
            self.btn_t.setChecked(self.metric == "time")
            self.btn_log.setChecked(self.log)
            self.btn_mode[self.mode].setChecked(True)
            self.custom_widget.setVisible(self.mode == "custom")

        def sync_zoom(self, source):
            """Same zoom on every panel. Done on manual zooms only: linked axes plus locked aspect
            ratios on cells of slightly different widths feed back into each other."""
            (x0, x1), (y0, y1) = source.viewRange()
            for vb in self.images:
                if vb is not source:
                    vb.blockSignals(True)
                    vb.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)
                    vb.blockSignals(False)

        def apply_theme(self):
            th = THEMES[self.theme]
            app.setStyleSheet(f"""
                QWidget {{ background:{th['bg']}; color:{th['text']}; font-size:12px; }}
                QPushButton {{ background:{th['surface']}; border:1px solid {th['border']};
                               border-radius:5px; padding:5px 12px; }}
                QPushButton:hover {{ border-color:{th['muted']}; }}
                QPushButton:checked {{ background:{th['accent']}; border-color:{th['accent_border']}; }}
                QPushButton:disabled {{ color:{th['muted']}; background:{th['bg']}; }}
                QSpinBox {{ background:{th['surface']}; border:1px solid {th['border']}; padding:2px 4px; }}
                QStatusBar {{ background:{th['surface']}; color:{th['muted']};
                              border-top:1px solid {th['border']}; }}""")
            self.view.setBackground(th["bg"])
            other = "light" if self.theme == "dark" else "dark"
            self.btn_theme.setText(f"{other.capitalize()} theme  [D]")

        # actions
        def set_mode(self, m):
            if m == "true" and not data.has_truth:
                return
            self.mode = m
            self.render()

        def toggle_panel(self, key, on):
            (self.hidden.discard if on else self.hidden.add)(key)
            self.render()

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

        def panel_at(self, pos):
            for vb, key in self.images.items():
                if vb.sceneBoundingRect().contains(pos):
                    return vb, key
            return None, None

        def on_click(self, ev):
            if not ev.double():
                return
            _, key = self.panel_at(ev.scenePos())
            if key is None:
                return
            if len(self.images) == 1:          # already alone: back to the previous view
                self.hidden = set()
                self.mode = self.prev_mode
            else:
                self.prev_mode = self.mode
                self.hidden = set(self.panels) - {key}
                self.mode = "custom"
            self.build_custom_bar()
            self.render()

        def on_mouse(self, pos):
            vb, key = self.panel_at(pos)
            if vb is None:
                return
            p = vb.mapSceneToView(pos)
            r, c = int(np.floor(p.y())), int(np.floor(p.x()))
            if 0 <= r < shape[0] and 0 <= c < shape[1] and mask[r, c]:
                pmt = int(np.flatnonzero((mapping_rc[:, 0] == r) & (mapping_rc[:, 1] == c))[0])
                title, q, t = self.panels[key]
                hit = f"q = {q[r, c]:.3g} p.e.   t = {t[r, c]:.1f} ns" if np.isfinite(q[r, c]) else "not hit"
                self.status.showMessage(f"{title.replace('<br>', ' · ')}   PMT {pmt} (row {r}, col {c})   {hit}")
            else:
                self.status.clearMessage()

    win = Window()
    win.show()
    return app, win


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h5file", help="ring-segmented file (event_XXXXXX/ring_K)")
    p.add_argument("--mapping", required=True, help="npz with pmt_image_positions (CNN image mapping)")
    p.add_argument("-e", "--event", type=int, default=0, help="first event shown (index in the file)")
    p.add_argument("--tube-id-base", type=int, choices=(0, 1), default=None, help="default: detect")
    p.add_argument("--truth", default=None,
                   help="WCSim h5 the events come from (attribute src_file): real whole event and true "
                        "per-particle hits")
    p.add_argument("--view", choices=MODES, default="all", help="first view (default all)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--light", action="store_const", const="light", dest="theme")
    g.add_argument("--dark", action="store_const", const="dark", dest="theme")
    a = p.parse_args(argv)
    data = LoadSegData(a.h5file, tube_id_base=a.tube_id_base, truth_path=a.truth)
    app, _win = run(data, a.mapping, a.event, a.theme, a.view)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

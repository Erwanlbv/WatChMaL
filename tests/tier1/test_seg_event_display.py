"""Checks for the ring-segmented loader (analysis/load_seg_h5_data.py) and the segmentation
event display (analysis/seg_display.py), on a small synthetic event/ring file."""
import h5py
import numpy as np
import pytest

from analysis.load_seg_h5_data import LoadSegData

N_PMTS = 20


def _ring(g, name, tube_ids, q, t, pdg=13, direction=(1.0, 0.0, 0.0), pred=None, track=None):
    R = g.create_group(name)
    R["tube_ids"] = np.asarray(tube_ids, dtype=np.int32)
    R["pmt_charge"] = np.asarray(q, dtype=np.float32)
    R["pmt_time"] = np.asarray(t, dtype=np.float32)
    R["event_type"] = pdg
    R["energy"] = 500.0
    R["n_digi_hits"] = len(tube_ids)
    for c, v in zip("xyz", (0.0, 0.0, 0.0)):
        R[f"vertex_{c}"] = v
    for c, v in zip("xyz", direction):
        R[f"particle_dir_{c}"] = v
    if pred is not None:
        R["predicted_three_momenta"] = np.asarray(pred, dtype=np.float32)
    R.attrs["dominant_dice"] = 0.8
    if track is not None:
        R.attrs["matched_parent_track_id"] = track
        R.attrs["source"] = "pred"


@pytest.fixture
def seg_file(tmp_path):
    """Two events, 1-based ids (as in Mathias' files). Event 0: two rings sharing PMT 3."""
    path = tmp_path / "seg.h5"
    with h5py.File(path, "w") as f:
        e0 = f.create_group("event_000000")
        e0.attrs["src_event"] = 0
        _ring(e0, "ring_0", [1, 2, 3], [1.0, 2.0, 0.5], [10.0, 11.0, 12.0], pred=(0.0, 1.0, 0.0), track=2)
        _ring(e0, "ring_1", [3, 4, N_PMTS], [1.5, 3.0, 1.0], [12.0, 20.0, 21.0], pdg=11,
              direction=(0.0, 0.0, 1.0), pred=(0.0, 0.0, 2.0), track=1)
        e1 = f.create_group("event_000001")
        e1.attrs["src_event"] = 1
        # one ring, PMT 5 hit in two triggers (distinct times): must stay two hits
        _ring(e1, "ring_0", [5, 5, 6], [1.0, 2.0, 1.0], [10.0, 1200.0, 11.0])
    return str(path)


HIT_DT = np.dtype([("tube_id", "<i4"), ("time_ns", "<f4"), ("charge_pe", "<f4"), ("x", "<f4"), ("y", "<f4"),
                   ("z", "<f4"), ("parent_track_id", "<i4"), ("primary_track_id", "<i4")])
PART_DT = np.dtype([("track_id", "<i4"), ("parent_id", "<i4"), ("pdg", "<i4"), ("energy_mev", "<f4")])


@pytest.fixture
def wcsim_file(tmp_path):
    """The WCSim h5 behind seg_file (tube_id 1-based). Event 0: track 1 (e-) and 2 (mu-), one
    dark-noise hit and one hit in no ring; the hit on PMT 3 has 3 photons (track 1, 2, 2) but
    WCSim labels it with its first photon (track 1). Event 1: its single ring, track 1."""
    hits0 = [(1, 10, 1.0, 1), (2, 11, 2.0, 2), (3, 12, 3.0, 1), (4, 20, 3.0, 1), (N_PMTS, 21, 1.0, 1),
             (7, 30, 0.5, -1), (8, 40, 4.0, 2)]
    photons0 = [[1], [2], [1, 2, 2], [1], [1], [-1], [2]]
    hits1 = [(5, 10, 1.0, 1), (5, 1200, 2.0, 1), (6, 11, 1.0, 1)]
    photons1 = [[1], [1], [1]]
    rows = [(t, tm, q, 0, 0, 0, p, p) for t, tm, q, p in hits0 + hits1]
    ph_rows, ph_ids, ph_index, hit_index = [], [], [0], [0]
    for photons in (photons0, photons1):
        local = 0
        for prims in photons:
            for pr in prims:
                ph_rows.append((0, 0, 1, 0, 0, 0, pr, pr))
                ph_ids.append(local)
                local += 1
            ph_index.append(len(ph_ids))
        hit_index.append(len(ph_rows))
    parts = [(1, 0, 11, 300.0), (2, 0, 13, 500.0), (5, 1, 22, 1.0), (1, 0, 13, 400.0)]
    path = tmp_path / "wcsim.h5"
    with h5py.File(path, "w") as f:
        ev = f.create_group("events")
        ev["digi_hits"] = np.array(rows, dtype=HIT_DT)
        ev["digi_hit_index"] = np.array([0, len(hits0), len(hits0) + len(hits1)], dtype=np.int64)
        ev["hits"] = np.array(ph_rows, dtype=HIT_DT)
        ev["hit_index"] = np.array(hit_index, dtype=np.int64)
        ev["digi_photon_ids"] = np.array(ph_ids, dtype=np.int32)
        ev["digi_photon_index"] = np.array(ph_index, dtype=np.int64)
        ev["particles"] = np.array(parts, dtype=PART_DT)
        ev["particle_index"] = np.array([0, 3, 4], dtype=np.int64)
    return str(path)


@pytest.fixture
def geometry(tmp_path):
    """N_PMTS PMTs on a cylinder (R=10, half height 10) and an image mapping (row, col)."""
    phi = np.linspace(-np.pi, np.pi, N_PMTS, endpoint=False)
    pos = np.column_stack([10 * np.cos(phi), 10 * np.sin(phi), np.linspace(-10, 10, N_PMTS)])
    geo, mapping = tmp_path / "geo.npz", tmp_path / "map.npz"
    np.savez(geo, position=pos)
    np.savez(mapping, pmt_image_positions=np.column_stack([np.arange(N_PMTS) % 5, np.arange(N_PMTS) // 5]))
    return str(geo), str(mapping)


def test_base_detected_and_removed(seg_file):
    data = LoadSegData(seg_file, n_pmts=N_PMTS)
    assert data.tube_id_base == 1
    data.load_hit_info(0, ring=1)
    assert data.hit_pmt.tolist() == [2, 3, N_PMTS - 1]


def test_base_ambiguous_raises(tmp_path):
    path = tmp_path / "amb.h5"
    with h5py.File(path, "w") as f:
        _ring(f.create_group("event_000000"), "ring_0", [3, 4], [1, 1], [1, 2])
    with pytest.raises(ValueError, match="cannot tell"):
        LoadSegData(str(path), n_pmts=N_PMTS)
    assert LoadSegData(str(path), tube_id_base=0, n_pmts=N_PMTS).tube_id_base == 0


def test_union_sums_shared_charge(seg_file):
    data = LoadSegData(seg_file, n_pmts=N_PMTS)
    data.load_hit_info(0)
    by_pmt = dict(zip(data.hit_pmt.tolist(), data.hit_charge.tolist()))
    assert by_pmt == {0: 1.0, 1: 2.0, 2: 2.0, 3: 3.0, N_PMTS - 1: 1.0}
    shared = data.hit_pmt == 2
    assert data.hit_n_rings[shared].tolist() == [2]
    assert data.hit_dominant_ring[shared].tolist() == [1]       # 1.5 > 0.5
    assert np.isclose(data.hit_charge.sum(), 1 + 2 + 0.5 + 1.5 + 3 + 1)


def test_union_keeps_multi_trigger_hits(seg_file):
    data = LoadSegData(seg_file, n_pmts=N_PMTS)
    with pytest.warns(UserWarning, match="several hits"):
        data.load_hit_info(1)
    assert sorted(zip(data.hit_pmt.tolist(), data.hit_time.tolist())) == [(4, 10.0), (4, 1200.0), (5, 11.0)]


def test_truth_and_predictions(seg_file):
    data = LoadSegData(seg_file, n_pmts=N_PMTS)
    assert data.n_rings.tolist() == [2, 1]
    assert data.ring_labels[0].tolist() == [13, 11]
    np.testing.assert_allclose(data.ring_pred_directions[0][1], [0, 0, 1])
    assert data.ring_pred_directions[1] is None


def test_same_ring_duplicate_is_not_shared(tmp_path):
    """Audit 2026-11-20: a (PMT, time) pair repeated in ONE ring was counted as a 2-ring PMT."""
    path = tmp_path / "dup.h5"
    with h5py.File(path, "w") as f:
        _ring(f.create_group("event_000000"), "ring_0", [0, 3, 3], [1.0, 1.0, 1.0], [5.0, 7.0, 7.0])
    data = LoadSegData(str(path), n_pmts=N_PMTS)
    data.load_hit_info(0)
    assert data.hit_n_rings.max() == 1


def test_truth_per_particle(seg_file, wcsim_file):
    data = LoadSegData(seg_file, n_pmts=N_PMTS, truth_path=wcsim_file)
    tp = data.true_parents(0)
    # PMT 3 (3 p.e., photons 1, 2, 2) is split 1 / 2 p.e.; WCSim's first-photon label gives it all to track 1
    assert [(p["track_id"], p["pdg"], p["ring"]) for p in tp] == [(1, 11, 1), (2, 13, 0)]
    assert np.allclose([p["pe"] for p in tp], [6.0, 8.0])
    assert np.allclose([p["pe_first_photon"] for p in tp], [8.0, 6.0])
    assert data.noise_pe(0) == 0.5
    data.load_true_hit_info(0)                       # the real event: noise and the no-ring hit included
    assert data.hit_pmt.size == 7 and np.isclose(data.hit_charge.sum(), 14.5)
    data.load_true_hit_info(0, track_id=2)           # the shared PMT appears in both particles
    assert sorted(data.hit_pmt.tolist()) == [1, 2, 7] and np.isclose(data.hit_charge.sum(), 8.0)
    data.load_true_hit_info(0, track_id=2, split=False)
    assert sorted(data.hit_pmt.tolist()) == [1, 7]
    assert data.true_parents(1)[0]["pdg"] == 13


def test_truth_wrong_file_is_rejected(seg_file, wcsim_file, tmp_path):
    """A truth file whose event does not hold the ring hits disables the truth for that event."""
    with h5py.File(wcsim_file, "a") as f:
        f["events/digi_hits"][...] = np.array([(9, 99, 1, 0, 0, 0, 1, 1)] * 10, dtype=HIT_DT)
    data = LoadSegData(seg_file, n_pmts=N_PMTS, truth_path=wcsim_file)
    with pytest.warns(UserWarning, match="wrong truth file"):
        assert data.true_parents(0) == []


def test_panel_grid_and_views():
    from analysis.seg_display import compact, panel_grid, visible_keys
    # 2 true particles; ring 0 matched to particle 1, ring 1 to particle 0, ring 2 unmatched
    pos = panel_grid(2, [1, 0, None], 3)
    assert pos[("true", 0)] == (0, 1) and pos[("reco", 1)] == (1, 1)
    assert pos[("reco", 0)] == (1, 2) and pos[("reco", 2)] == (1, 3)
    # no truth: one row
    assert panel_grid(0, [], 2) == {("event",): (0, 0), ("reco", 0): (0, 1), ("reco", 1): (0, 2)}
    keys = list(pos)
    assert visible_keys(keys, "true") == [("event",), ("true", 0), ("true", 1)]
    assert ("true", 0) not in visible_keys(keys, "reco")
    solo = visible_keys(keys, "custom", hidden=set(keys) - {("reco", 2)})
    assert compact(pos, solo) == {("reco", 2): (0, 0)}
    assert compact(pos, keys)[("reco", 1)] == (1, 1)                # all: ring 1 under true particle 0
    assert compact(pos, visible_keys(keys, "reco")) == {("event",): (0, 0), ("reco", 1): (0, 1),
                                                        ("reco", 0): (0, 2), ("reco", 2): (0, 3)}


def test_cnn_image_places_and_aggregates(geometry):
    """Pixel = mapping[pmt]; a PMT hit twice gets the summed charge and the earliest time."""
    from analysis.seg_display import cnn_image, load_mapping, pmt_mask
    _, mapping = geometry
    rc, shape = load_mapping(mapping)
    assert shape == (5, 4) and pmt_mask(rc, shape).all()
    q, t = cnn_image(np.array([7, 7, 2]), np.array([1.0, 2.0, 5.0]), np.array([30.0, 10.0, 4.0]), rc, shape)
    assert q[rc[7, 0], rc[7, 1]] == 3.0 and t[rc[7, 0], rc[7, 1]] == 10.0
    assert q[rc[2, 0], rc[2, 1]] == 5.0
    assert np.isfinite(q).sum() == 2


def test_to_rgba_theme_colours():
    from analysis.seg_display import THEMES, to_rgba
    lut = np.stack([np.arange(256)] * 3 + [np.full(256, 255)], axis=1).astype(np.uint8)
    values = np.array([[np.nan, np.nan, 1.0, 3.0]])
    mask = np.array([[False, True, True, True]])
    for name, th in THEMES.items():
        rgba = to_rgba(values, mask, (1.0, 3.0), lut, th)
        assert rgba[0, 0, 3] == 0                                   # no PMT: transparent
        assert tuple(rgba[0, 1, :3]) == th["unhit"]                 # unhit PMT: theme colour
        assert rgba[0, 2, 0] == 0 and rgba[0, 3, 0] == 255          # ends of the colour map


def test_window_smoke(seg_file, geometry, monkeypatch):
    """The window builds, navigates and switches metric, log and theme (offscreen Qt)."""
    pytest.importorskip("pyqtgraph")
    pytest.importorskip("PyQt6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from analysis.seg_display import run
    _, mapping = geometry
    data = LoadSegData(seg_file, n_pmts=N_PMTS)
    app, win = run(data, mapping, event=0, theme="dark")
    assert len(win.panels) == 3                                     # whole event + 2 rings
    win.set_metric("time")
    win.toggle_log()
    win.toggle_theme()
    win.go(1)
    assert win.i == 1 and len(win.panels) == 2 and win.theme == "light"
    win.close()


def test_window_views_with_truth(seg_file, wcsim_file, geometry, monkeypatch):
    """All / true / reco / custom views and the custom toggles (offscreen Qt)."""
    pytest.importorskip("pyqtgraph")
    pytest.importorskip("PyQt6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from analysis.seg_display import run
    _, mapping = geometry
    data = LoadSegData(seg_file, n_pmts=N_PMTS, truth_path=wcsim_file)
    app, win = run(data, mapping, event=0, theme="dark")
    assert len(win.panels) == 5 and len(win.images) == 5             # event + 2 true + 2 rings
    win.set_mode("true")
    assert sorted(k[0] for k in win.images.values()) == ["event", "true", "true"]
    win.set_mode("reco")
    assert sorted(k[0] for k in win.images.values()) == ["event", "reco", "reco"]
    win.set_mode("custom")
    win.toggle_panel(("event",), False)
    win.toggle_panel(("reco", 0), False)
    assert len(win.images) == 3 and win.custom_widget.isVisibleTo(win)
    win.go(1)                                                         # hidden panels stay hidden
    assert ("event",) not in win.images.values()
    win.close()
    no_truth = LoadSegData(seg_file, n_pmts=N_PMTS)
    app, win = run(no_truth, mapping, event=0, theme="light", mode="true")
    assert win.mode == "all" and not win.btn_mode["true"].isEnabled()
    win.close()


def test_loaddata_accepts_pdg_labels():
    """LoadData used to crash (IndexError 13) on files storing PDG codes."""
    from analysis.load_h5_data import to_watchmal_labels
    assert to_watchmal_labels(np.array([0, 1, 2, 3])).tolist() == [0, 1, 2, 3]
    assert to_watchmal_labels(np.array([11, -13, 22, 111, 13])).tolist() == [1, 2, 0, 3, 2]
    with pytest.raises(ValueError, match="211"):
        to_watchmal_labels(np.array([11, 211]))

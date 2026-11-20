"""Checks for the ring-segmented loader (analysis/load_seg_h5_data.py) and the segmentation
event display (analysis/seg_display.py), on a small synthetic event/ring file."""
import h5py
import numpy as np
import pytest

from analysis.load_seg_h5_data import LoadSegData

N_PMTS = 20


def _ring(g, name, tube_ids, q, t, pdg=13, direction=(1.0, 0.0, 0.0), pred=None):
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


@pytest.fixture
def seg_file(tmp_path):
    """Two events, 1-based ids (as in Mathias' files). Event 0: two rings sharing PMT 3."""
    path = tmp_path / "seg.h5"
    with h5py.File(path, "w") as f:
        e0 = f.create_group("event_000000")
        _ring(e0, "ring_0", [1, 2, 3], [1.0, 2.0, 0.5], [10.0, 11.0, 12.0], pred=(0.0, 1.0, 0.0))
        _ring(e0, "ring_1", [3, 4, N_PMTS], [1.5, 3.0, 1.0], [12.0, 20.0, 21.0], pdg=11,
              direction=(0.0, 0.0, 1.0), pred=(0.0, 0.0, 2.0))
        e1 = f.create_group("event_000001")
        # one ring, PMT 5 hit in two triggers (distinct times): must stay two hits
        _ring(e1, "ring_0", [5, 5, 6], [1.0, 2.0, 1.0], [10.0, 1200.0, 11.0])
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

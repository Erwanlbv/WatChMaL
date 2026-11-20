"""Checks for the ring-segmented loader (analysis/load_seg_h5_data.py) and the 2D displays
(analysis/plotting_2d.py), on a small synthetic event/ring file."""
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


def test_wall_point():
    from analysis.plotting_2d import _wall_point
    np.testing.assert_allclose(_wall_point([0, 0, 0], [1, 0, 0], 10, 10), [10, 0, 0])
    np.testing.assert_allclose(_wall_point([0, 0, 5], [0, 0, 1], 10, 10), [0, 0, 10])
    np.testing.assert_allclose(_wall_point([0, 0, 0], [0.6, 0, 0.8], 10, 10), [7.5, 0, 10])


def test_log_values_handles_non_positive():
    from analysis.plotting_2d import _log_values
    v, label, shift = _log_values(np.array([-2.0, 0.0, 8.0]), "time [ns]")
    assert np.isfinite(v).all() and shift == 3.0 and "+ 3" in label


@pytest.mark.parametrize("log", [False, True])
def test_figures_build(seg_file, geometry, log):
    pytest.importorskip("plotly")
    from analysis import plotting_2d as p2
    geo, mapping = geometry
    data = LoadSegData(seg_file, n_pmts=N_PMTS)
    fig = p2.plot_2d_original_vs_segmented(data, 0, mapping, "time", log=log, geo_file_path=geo)
    assert len(fig.layout.annotations) == 3                      # whole event + 2 rings
    names = {t.name for t in fig.data if t.name}
    assert any(n.startswith("true dir. ring 1") for n in names)
    assert any(n.startswith("pred. dir. ring 0") for n in names)
    p2.plot_2d_event(data, 0, mapping, ring=1, log=log, geo_file_path=geo)
    p2.plot_2d_ring_ownership(data, 0, mapping, geo_file_path=geo)
    p2.plot_hit_histograms(data, 0, log_x=log, log_y=log)


def test_image_coordinates_follow_prouse_convention(geometry):
    """x = image column, y = image row, as CNNEventDisplay.plot_data_2d."""
    from analysis.plotting_2d import load_image_mapping
    _, mapping = geometry
    rc = np.load(mapping)["pmt_image_positions"]
    np.testing.assert_array_equal(load_image_mapping(mapping), rc[:, [1, 0]])

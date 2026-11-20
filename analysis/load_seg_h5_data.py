"""
Loader for ring-segmented HDF5 files (one group per event, one sub-group per ring).

This is the counterpart of `analysis.load_h5_data.LoadData` for the hierarchical
`event/ring` layout written by the multi-ring segmentation chain (Mathias' RingSegmentor,
`format='segmented_events_v2'`) and by Z. Hu's `Root2h5*.py` converters. It exposes the
same `load_hit_info` interface, so the 2D displays of `analysis.plotting_2d` accept
either loader.
"""
import warnings

import h5py
import numpy as np

# HK FD 20" PMT count: highest tube id of a 1-based file
DEFAULT_N_PMTS = 19746

# PDG code -> short name, for titles
PDG_NAMES = {11: "e-", -11: "e+", 13: "mu-", -13: "mu+", 22: "gamma", 111: "pi0", 211: "pi+", -211: "pi-"}


class LoadSegData():
    """
    Load the truth information of a ring-segmented h5 file into numpy arrays.

    File layout::

        /event_XXXXXX/ring_K/tube_ids, pmt_charge, pmt_time      (n_hits,) per ring
        /event_XXXXXX/ring_K/energy, event_type, vertex_[xyz], particle_dir_[xyz], dwall, towall, ...
        /event_XXXXXX/ring_K/predicted_pid, predicted_positions, predicted_three_momenta   (recon files only)
        /event_XXXXXX/true_parent_*                              one entry per true particle

    The rings of one event may share PMTs: the segmentation splits the charge of a PMT
    between rings and copies its time. The unsegmented event is NOT stored in the file;
    `load_hit_info(i)` (ring=None) rebuilds it as the union of the rings, with the charge
    of each PMT summed over rings. Hits assigned to no ring (rings below the segmentation
    threshold) are therefore missing from it.

    Parameters
    ----------
    h5path : str
        Path to the h5 file.
    tube_id_base : {0, 1, None}, optional
        Value of the first PMT id in the file's `tube_ids`. It differs between producers:
        Mathias' segmented files are 1-based ([1, 19746]); Z. Hu's Root2h5 files are
        0-based ([0, 19745], `GetTubeId() - 1`). None (default) reads the file attribute
        `tube_id_base` if present, otherwise detects it from the data (an id 0 means
        0-based, an id equal to `n_pmts` means 1-based) and raises if neither is seen.
    n_pmts : int, optional
        Number of PMTs of the detector, used by the detection above. Default 19746 (HK FD).

    Attributes
    ----------
    event_names : list of str
        Name of each event group, in file order. Every per-event attribute is in this order.
    n_rings : np.ndarray of int, shape (n_events,)
        Number of rings of each event.
    ring_names : list of list of str
        Ring group names of each event.
    positions : np.ndarray, shape (n_events, 3)
        True vertex [cm] (taken from ring 0; all rings of an event share it).
    ring_labels, ring_energies : list of np.ndarray, shape (n_rings,)
        PDG code and true total energy [MeV] of the particle matched to each ring.
    ring_directions : list of np.ndarray, shape (n_rings, 3)
        True unit direction of the particle matched to each ring.
    ring_pred_directions : list of np.ndarray or None
        Unit direction of `predicted_three_momenta`, per ring, when the file has it.
    ring_pred_pid : list of np.ndarray or None
        `predicted_pid` scores, shape (n_rings, n_classes), when the file has it.
    ring_attrs : list of list of dict
        HDF5 attributes of each ring group (dice, containment, matched track id, ...).
    file_attrs : dict
        Root attributes of the file.
    tube_id_base : int
        The base used (see parameter).
    hit_pmt, hit_time, hit_charge : np.ndarray, shape (n_hits,)
        Hits of the loaded event (after load_hit_info). `hit_pmt` is 0-BASED, i.e. it
        indexes the rows of the WatChMaL geometry / image-mapping files directly, as in
        `LoadData`.
    hit_n_rings : np.ndarray of int, shape (n_hits,)
        Number of rings containing each PMT (1 for a single ring).
    hit_dominant_ring : np.ndarray of int, shape (n_hits,)
        Index of the ring holding the largest share of each PMT's charge.
    loaded_event_idx, loaded_ring : int, int or None
        What load_hit_info last loaded.

    Example
    -------
    >>> from analysis.load_seg_h5_data import LoadSegData
    >>> data = LoadSegData('/path/to/seg_b0000.h5')
    >>> data.load_hit_info(3)            # union of all rings of event 3
    >>> data.load_hit_info(3, ring=1)    # ring 1 only
    """

    def __init__(self, h5path, tube_id_base=None, n_pmts=DEFAULT_N_PMTS):
        self.h5path = h5path
        self.n_pmts = n_pmts
        with h5py.File(h5path, "r") as f:
            self.file_attrs = dict(f.attrs)
            self.event_names = sorted(k for k in f if k.startswith("event_"))
            self.ring_names, self.ring_attrs = [], []
            self.ring_labels, self.ring_energies, self.ring_directions = [], [], []
            self.ring_pred_directions, self.ring_pred_pid = [], []
            positions = []
            seen_min, seen_max = np.iinfo(np.int64).max, -1
            for name in self.event_names:
                g = f[name]
                rings = sorted((k for k in g if k.startswith("ring_")), key=lambda k: int(k.split("_")[1]))
                self.ring_names.append(rings)
                self.ring_attrs.append([dict(g[r].attrs) for r in rings])
                labels, energies, dirs, pdirs, ppid = [], [], [], [], []
                for r in rings:
                    R = g[r]
                    labels.append(int(R["event_type"][()]))
                    energies.append(float(R["energy"][()]))
                    dirs.append([float(R[f"particle_dir_{c}"][()]) for c in "xyz"])
                    if "predicted_three_momenta" in R:
                        p = np.asarray(R["predicted_three_momenta"][()], dtype=float)
                        pdirs.append(p / np.linalg.norm(p))
                    if "predicted_pid" in R:
                        ppid.append(np.asarray(R["predicted_pid"][()], dtype=float))
                    if tube_id_base is None and "tube_id_base" not in self.file_attrs:
                        t = R["tube_ids"]
                        if t.size:
                            t = t[()]
                            seen_min, seen_max = min(seen_min, int(t.min())), max(seen_max, int(t.max()))
                if rings:
                    R0 = g[rings[0]]
                    positions.append([float(R0[f"vertex_{c}"][()]) for c in "xyz"])
                else:
                    positions.append([np.nan] * 3)
                self.ring_labels.append(np.array(labels, dtype=int))
                self.ring_energies.append(np.array(energies))
                self.ring_directions.append(np.array(dirs).reshape(-1, 3))
                self.ring_pred_directions.append(np.array(pdirs) if len(pdirs) == len(rings) and rings else None)
                self.ring_pred_pid.append(np.array(ppid) if len(ppid) == len(rings) and rings else None)
        self.n_rings = np.array([len(r) for r in self.ring_names], dtype=int)
        self.positions = np.array(positions).reshape(-1, 3)
        self.tube_id_base = self._resolve_base(tube_id_base, seen_min, seen_max)
        print(f"[LoadSegData] {h5path}: {len(self.event_names)} events, rings/event "
              f"{dict(zip(*np.unique(self.n_rings, return_counts=True)))}, tube_id_base={self.tube_id_base}")

    def _resolve_base(self, tube_id_base, seen_min, seen_max):
        if tube_id_base is not None:
            if tube_id_base not in (0, 1):
                raise ValueError(f"tube_id_base must be 0 or 1, got {tube_id_base}")
            return int(tube_id_base)
        if "tube_id_base" in self.file_attrs:
            return int(self.file_attrs["tube_id_base"])
        if seen_min == 0 and seen_max == self.n_pmts:
            raise ValueError(f"{self.h5path}: tube_ids span [0, {self.n_pmts}], i.e. both bases at once")
        if seen_min == 0:
            return 0
        if seen_max == self.n_pmts:
            return 1
        raise ValueError(f"{self.h5path}: cannot tell the tube_ids base from the data (range [{seen_min}, "
                         f"{seen_max}], n_pmts={self.n_pmts}); pass tube_id_base=0 or 1")

    def __len__(self):
        return len(self.event_names)

    def _read_ring(self, f, event_idx, ring):
        R = f[self.event_names[event_idx]][self.ring_names[event_idx][ring]]
        pmt = R["tube_ids"][()].astype(np.int64) - self.tube_id_base
        return pmt, R["pmt_time"][()].astype(float), R["pmt_charge"][()].astype(float)

    def load_hit_info(self, event_idx, ring=None):
        """
        Read the hits of one event, or of one of its rings.

        Nothing is returned; sets hit_pmt (0-based), hit_time, hit_charge, hit_n_rings,
        hit_dominant_ring, loaded_event_idx and loaded_ring.

        Parameters
        ----------
        event_idx : int
            Index of the event within this object (position in event_names).
        ring : int, optional
            Ring index within the event (position in ring_names[event_idx]). None (default)
            loads the union of all rings: one entry per PMT, charge summed over rings.
        """
        n = self.n_rings[event_idx]
        if ring is not None and not 0 <= ring < n:
            raise IndexError(f"event {event_idx} has {n} rings, asked ring {ring}")
        with h5py.File(self.h5path, "r") as f:
            if ring is not None:
                pmt, t, q = self._read_ring(f, event_idx, ring)
                self.hit_pmt, self.hit_time, self.hit_charge = pmt, t, q
                self.hit_n_rings = np.ones(pmt.size, dtype=int)
                self.hit_dominant_ring = np.full(pmt.size, ring, dtype=int)
            else:
                parts = [self._read_ring(f, event_idx, k) for k in range(n)]
                if not parts:
                    empty = np.array([], dtype=float)
                    self.hit_pmt, self.hit_time, self.hit_charge = np.array([], dtype=np.int64), empty, empty
                    self.hit_n_rings = self.hit_dominant_ring = np.array([], dtype=int)
                else:
                    self._union(parts, event_idx)
        if self.hit_pmt.size and (self.hit_pmt.min() < 0 or self.hit_pmt.max() >= self.n_pmts):
            raise ValueError(f"event {event_idx}: hit_pmt out of [0, {self.n_pmts}) after removing base "
                             f"{self.tube_id_base} - wrong tube_id_base?")
        self.loaded_event_idx, self.loaded_ring = event_idx, ring

    def _union(self, parts, event_idx):
        pmt_all = np.concatenate([p[0] for p in parts])
        t_all = np.concatenate([p[1] for p in parts])
        q_all = np.concatenate([p[2] for p in parts])
        ring_all = np.concatenate([np.full(p[0].size, k) for k, p in enumerate(parts)])
        # One physical hit = one (PMT, time) pair: the segmentation copies the time of a hit into
        # every ring sharing it, while a PMT hit in several triggers appears with distinct times
        # (Z. Hu's Root2h5 files loop over all triggers). Merge on the pair, sum the charge.
        key = np.stack([pmt_all, np.round(t_all * 1000).astype(np.int64)], axis=1)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        pmt, t = pmt_all[first], t_all[first]
        q = np.zeros(pmt.size)
        np.add.at(q, inv, q_all)
        n_multi = pmt.size - np.unique(pmt).size
        if n_multi:
            warnings.warn(f"[LoadSegData] event {event_idx}: {n_multi} PMTs carry several hits at different "
                          f"times (several triggers); they are kept as separate hits")
        n_rings = np.bincount(inv, minlength=pmt.size)
        # dominant ring: ring with the largest charge share for each PMT
        best_q = np.full(pmt.size, -np.inf)
        dom = np.full(pmt.size, -1)
        for k in range(len(parts)):
            sel = ring_all == k
            better = q_all[sel] > best_q[inv[sel]]
            idx = inv[sel][better]
            best_q[idx] = q_all[sel][better]
            dom[idx] = k
        self.hit_pmt, self.hit_time, self.hit_charge = pmt, t, q
        self.hit_n_rings, self.hit_dominant_ring = n_rings, dom

    def ring_title(self, event_idx, ring):
        """Short description of one ring, for panel titles."""
        pdg = self.ring_labels[event_idx][ring]
        txt = f"ring {ring}: {PDG_NAMES.get(pdg, pdg)}, E={self.ring_energies[event_idx][ring]:.0f} MeV"
        a = self.ring_attrs[event_idx][ring]
        if "dominant_dice" in a:
            txt += f", dice={a['dominant_dice']:.2f}"
        return txt

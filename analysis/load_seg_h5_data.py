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
    truth_path : str, optional
        The WCSim h5 the events come from (file attribute `src_file`, event attribute
        `src_event`). The segmented files hold only the predicted rings (`write_true=False`);
        this file holds the real unsegmented event and the true primary of every digitized hit,
        read by `true_parents` and `load_true_hit_info`.

    Attributes
    ----------
    event_names : list of str
        Name of each event group, in file order. Every per-event attribute is in this order.
    n_rings : np.ndarray of int, shape (n_events,)
        Number of rings of each event.
    ring_names : list of list of str
        Ring group names of each event.
    positions : np.ndarray, shape (n_events, 3)
        True vertex [cm] (taken from ring 0; all rings of an event share it). NaN for an event
        with no ring (none passed the segmentation threshold).
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
        What load_hit_info last loaded. The plotting functions call load_hit_info, so they
        change these and the hit_* attributes.

    Notes
    -----
    In the union, hits are merged on (PMT, time): a (PMT, time) pair repeated inside one ring
    (seen once in Root2h5 files) is merged too, so the union can hold one hit fewer than the
    ring alone.

    Example
    -------
    >>> from analysis.load_seg_h5_data import LoadSegData
    >>> data = LoadSegData('/path/to/seg_b0000.h5')
    >>> data.load_hit_info(3)            # union of all rings of event 3
    >>> data.load_hit_info(3, ring=1)    # ring 1 only
    """

    def __init__(self, h5path, tube_id_base=None, n_pmts=DEFAULT_N_PMTS, truth_path=None):
        self.h5path = h5path
        self.n_pmts = n_pmts
        self._warned = set()
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
            self.src_events = [int(f[n].attrs["src_event"]) if "src_event" in f[n].attrs else None
                               for n in self.event_names]
        self.n_rings = np.array([len(r) for r in self.ring_names], dtype=int)
        self.positions = np.array(positions).reshape(-1, 3)
        self.tube_id_base = self._resolve_base(tube_id_base, seen_min, seen_max)
        print(f"[LoadSegData] {h5path}: {len(self.event_names)} events, rings/event "
              f"{dict(zip(*np.unique(self.n_rings, return_counts=True)))}, tube_id_base={self.tube_id_base}")
        self.truth = WCSimTruth(truth_path, n_pmts) if truth_path else None
        self._truth_cache = {}

    # ── per-PMT truth, from the WCSim file the events come from ─────────────────────────

    @property
    def has_truth(self):
        return self.truth is not None

    def true_parents(self, event_idx):
        """
        True particles of one event, from the WCSim file: list of dicts with track_id, pdg,
        energy [MeV], pe (its share of the digitized charge, split by photons), pe_first_photon
        (the same with WCSim's first-photon label: this is the `true_parent_pe` of the segmented
        files) and ring (index of the ring matched
        to it by the segmentation, `matched_parent_track_id`, or None). Empty without truth, or
        when the WCSim event does not hold the hits of the rings (wrong truth file).
        """
        if not self.has_truth:
            return []
        if event_idx not in self._truth_cache:
            self._truth_cache[event_idx] = self._build_truth(event_idx)
        return self._truth_cache[event_idx]

    def _build_truth(self, event_idx):
        src = self.src_events[event_idx]
        if src is None:
            src = event_idx
        if not 0 <= src < len(self.truth):
            warnings.warn(f"[LoadSegData] event {event_idx}: src_event {src} not in {self.truth.path}")
            return []
        h = self.truth.digi_hits(src)
        # consistency: every ring hit (PMT, time) must be a digitized hit of the WCSim event
        if self.n_rings[event_idx]:
            self.load_hit_info(event_idx)
            ring_keys = {(int(p), int(round(t * 1000))) for p, t in zip(self.hit_pmt, self.hit_time)}
            wc_keys = {(int(p), int(round(t * 1000))) for p, t in zip(h["pmt"], h["time"])}
            found = len(ring_keys & wc_keys) / len(ring_keys)
            if found < 0.99:
                warnings.warn(f"[LoadSegData] event {event_idx}: only {found:.1%} of the ring hits are in WCSim "
                              f"event {src} of {self.truth.path} - wrong truth file? Truth disabled for it.")
                return []
        matched = {}
        for k, a in enumerate(self.ring_attrs[event_idx]):
            if "matched_parent_track_id" in a:
                matched.setdefault(int(a["matched_parent_track_id"]), k)
        part = self.truth.particles(src)
        srow, sprim, frac = self._split(event_idx, src)
        out = []
        for tid in np.unique(sprim[sprim > 0]):
            row = part[(part["track_id"] == tid) & (part["parent_id"] == 0)]
            out.append(dict(track_id=int(tid),
                            pdg=int(row["pdg"][0]) if row.size else 0,
                            energy=float(row["energy_mev"][0]) if row.size else float("nan"),
                            pe=float((h["charge"][srow] * frac)[sprim == tid].sum()),
                            pe_first_photon=float(h["charge"][h["primary"] == tid].sum()),
                            ring=matched.get(int(tid))))
        return out

    def _split(self, event_idx, src):
        key = ("split", event_idx)
        if key not in self._truth_cache:
            self._truth_cache[key] = self.truth.photon_split(src)
        return self._truth_cache[key]

    def load_true_hit_info(self, event_idx, track_id=None, split=True):
        """
        Digitized hits of the WCSim event behind `event_idx`: all of them (track_id None: the
        real unsegmented event, dark noise included) or the part of them due to one true primary.
        Sets the same hit_* attributes as load_hit_info.

        split=True (default): a hit is shared between the primaries of its photons, each getting
        the charge times its photon fraction, so a PMT on two overlapping rings appears in both.
        split=False: WCSim's per-hit label `digi_hits.primary_track_id` (first photon only), the
        whole hit to one primary - shared regions are then missing from all but one particle.
        """
        if not self.has_truth:
            raise RuntimeError("no truth file: pass truth_path= (the WCSim h5 of the src_file attribute)")
        src = self.src_events[event_idx] if self.src_events[event_idx] is not None else event_idx
        h = self.truth.digi_hits(src)
        if track_id is None:
            self.hit_pmt, self.hit_time, self.hit_charge = h["pmt"], h["time"], h["charge"]
        elif split:
            row, prim, frac = self._split(event_idx, src)
            sel = prim == track_id
            r = row[sel]
            self.hit_pmt, self.hit_time, self.hit_charge = h["pmt"][r], h["time"][r], h["charge"][r] * frac[sel]
        else:
            sel = h["primary"] == track_id
            self.hit_pmt, self.hit_time, self.hit_charge = h["pmt"][sel], h["time"][sel], h["charge"][sel]
        self.hit_n_rings = self.hit_dominant_ring = np.full(self.hit_pmt.size, -1, dtype=int)
        self.loaded_event_idx, self.loaded_ring = event_idx, None

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
        if n_multi and event_idx not in self._warned:
            self._warned.add(event_idx)
            warnings.warn(f"[LoadSegData] event {event_idx}: {n_multi} PMTs carry several hits at different "
                          f"times (several triggers); they are kept as separate hits")
        # number of DISTINCT rings holding each hit (a (PMT, time) pair repeated inside one ring counts once)
        hit_ring = np.unique(np.stack([inv, ring_all], axis=1), axis=0)
        n_rings = np.bincount(hit_ring[:, 0], minlength=pmt.size)
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

    def noise_pe(self, event_idx):
        """Charge of the dark-noise hits (primary_track_id = -1) of the WCSim event, or None."""
        if not self.has_truth:
            return None
        src = self.src_events[event_idx] if self.src_events[event_idx] is not None else event_idx
        h = self.truth.digi_hits(src)
        row, prim, frac = self._split(event_idx, src)
        return float((h["charge"][row] * frac)[prim < 0].sum())

    def ring_title(self, event_idx, ring):
        """Short description of one ring, for panel titles."""
        pdg = self.ring_labels[event_idx][ring]
        txt = f"ring {ring}: {PDG_NAMES.get(pdg, pdg)}, E={self.ring_energies[event_idx][ring]:.0f} MeV"
        a = self.ring_attrs[event_idx][ring]
        if "dominant_dice" in a:
            txt += f", dice={a['dominant_dice']:.2f}"
        return txt


class WCSimTruth():
    """
    Per-hit truth of a WCSim h5 file (`wcsim_root_to_h5.py`, the `src_file` of the segmented
    events): /events/digi_hits (tube_id 1-based, time_ns, charge_pe, parent_track_id,
    primary_track_id) indexed by /events/digi_hit_index, and /events/particles by
    /events/particle_index. `primary_track_id` is the true primary each hit's photons come
    from (-1: dark noise); it is the `true_parent_track_id` of the segmented file.
    """

    def __init__(self, path, n_pmts=DEFAULT_N_PMTS):
        self.path, self.n_pmts = path, n_pmts
        with h5py.File(path, "r") as f:
            ev = f["events"]
            self.hit_index = ev["digi_hit_index"][()]
            self.particle_index = ev["particle_index"][()]
            self.photon_hit_index = ev["hit_index"][()]                 # photons (true hits) per event
            self.digi_photon_index = ev["digi_photon_index"][()]       # photons of each digitized hit
        print(f"[WCSimTruth] {path}: {len(self)} events, {self.hit_index[-1]} digitized hits")

    def __len__(self):
        return len(self.hit_index) - 1

    def digi_hits(self, src_event):
        a, b = self.hit_index[src_event], self.hit_index[src_event + 1]
        with h5py.File(self.path, "r") as f:
            h = f["events/digi_hits"][a:b]
        pmt = h["tube_id"].astype(np.int64) - 1
        if pmt.size and (pmt.min() < 0 or pmt.max() >= self.n_pmts):
            raise ValueError(f"{self.path} event {src_event}: tube_id out of [1, {self.n_pmts}]")
        return dict(pmt=pmt, time=h["time_ns"].astype(float), charge=h["charge_pe"].astype(float),
                    primary=h["primary_track_id"].astype(int))

    def photon_split(self, src_event):
        """
        Share of each digitized hit coming from each true primary, from its photons
        (`digi_photon_ids` -> `hits.primary_track_id`). Returns (digi_row, primary, fraction):
        one entry per (digitized hit, primary) pair, digi_row local to the event, fractions
        summing to 1 per hit (photon counts; -1 = dark-noise photons).

        `digi_hits.primary_track_id` is the primary of the FIRST photon only, so a PMT hit by
        two particles goes entirely to one of them (measured on batch_0000: 14% of the digitized
        hits mix primaries, carrying 42% of the charge).
        """
        a, b = self.hit_index[src_event], self.hit_index[src_event + 1]
        pa, pb = self.digi_photon_index[a], self.digi_photon_index[b]
        with h5py.File(self.path, "r") as f:
            ev = f["events"]
            ids = ev["digi_photon_ids"][pa:pb]                              # local to the event's photons
            h0, h1 = self.photon_hit_index[src_event], self.photon_hit_index[src_event + 1]
            prim = ev["hits"].fields("primary_track_id")[h0:h1]
        n_ph = np.diff(self.digi_photon_index[a:b + 1])
        digi = np.repeat(np.arange(b - a), n_ph)
        pairs, counts = np.unique(np.stack([digi, prim[ids]], axis=1), axis=0, return_counts=True)
        return pairs[:, 0], pairs[:, 1], counts / n_ph[pairs[:, 0]]

    def particles(self, src_event):
        a, b = self.particle_index[src_event], self.particle_index[src_event + 1]
        with h5py.File(self.path, "r") as f:
            return f["events/particles"][a:b]

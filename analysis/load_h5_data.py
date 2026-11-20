from watchmal.utils.math import  direction_from_angles, momentum_from_energy
import numpy as np
import h5py as h5

class LoadData():
    """
    Load the truth information of a WatChMaL h5 dataset into numpy arrays.

    Reads the per-event datasets of the h5 file (optionally only the events of
    one index set, e.g. the test set) and stores them as attributes, one row
    per event. The hits themselves are not loaded here, as they are too large;
    use load_hit_info to read the hits of a single event.

    Parameters
    ----------
    h5path : str
        Path to the h5 file. It must contain the datasets 'energies',
        'labels', 'positions', 'event_ids', 'angles', 'veto', 'veto2',
        'fully_contained', 'root_files', 'event_hits_index' and 'hit_pmt'.
        Can also load neutrino information for NEUT -> WCSim events
    idxs_path : str, optional
        Path to the .npz file holding the index sets (the train/val/test split
        file used for training).
    idxs_set : str, optional
        Name of the index set to load from `idxs_path`, e.g. 'train_idxs',
        'val_idxs' or 'test_idxs'. Both `idxs_path` and `idxs_set` are needed
        to select a subset: if either is missing, every event in the file is
        loaded.
    zenith_axis : int, optional
        Axis the zenith angle is measured from, used to turn `angles` into
        `directions`. The default, 2, is the z-axis (the vertical axis of the
        HK far detector).

    Attributes
    ----------
    h5path, idxs_path : str
        The paths given above.
    idxs : np.ndarray of int, shape (n_events,)
        Row number in the h5 file of each loaded event. Every other attribute
        is in this order, so attribute[i] belongs to h5 row idxs[i].
    energies : np.ndarray, shape (n_events,)
        True total energy of the particle [MeV].
    labels : np.ndarray of int, shape (n_events,)
        Particle type label (0: gamma, 1: electron, 2: muon, 3: pi0).
    momenta : np.ndarray, shape (n_events,)
        True momentum [MeV], calculated from `energies` and the particle mass
        given by `labels`.
    positions : np.ndarray, shape (n_events, 3)
        True vertex (x, y, z) [cm].
    event_ids : np.ndarray of int, shape (n_events,)
        Event number within its original root file.
    angles : np.ndarray, shape (n_events, 2)
        True (zenith, azimuth) angles of the particle direction [rad].
    directions : np.ndarray, shape (n_events, 3)
        True unit direction vector (x, y, z), calculated from `angles`.
    veto, veto2 : np.ndarray of bool, shape (n_events,)
        The 'veto' and 'veto2' flags of the h5 file.
    FC : np.ndarray of bool, shape (n_events,)
        The 'fully_contained' flag of the h5 file .
    root_files : np.ndarray, shape (n_events,)
        Path of the root file each event came from.
    nhits : np.ndarray of int, shape (n_events,)
        Number of hits in each event.
    event_hits_index : np.ndarray of int, shape (n_events,)
        Position of the first hit of each event in the h5 file's hit datasets
        ('hit_pmt', 'hit_time', 'hit_charge').
    nu_energies : np.ndarray, shape (n_events,)
        True neutrino energy [MeV].
    nu_directions : np.ndarray, shape (n_events, 3)
        True neutrino unit direction vector (x, y, z).
    nu_pids : np.ndarray of int, shape (n_events,)
        Neutrino particle code.
    hit_time, hit_charge, hit_pmt : np.ndarray, shape (n_hits,)
        Hits of one event. These only exist after calling load_hit_info.
    loaded_event_idx : int
        The event whose hits are currently loaded. Only exists after calling
        load_hit_info.

    Notes
    -----
    The three nu_* attributes are only set for files that contain the
    'neutrino_energy', 'neutrino_direction' and 'neutrino_pid' datasets (beam
    MC). For other files (e.g. particle gun) they do not exist, so check with
    hasattr(data, 'nu_energies') before using them.

    Each per-event dataset is read in full before the index set is applied, so
    loading takes a while for the 10M+ event files.

    The indices of an index set do not need to be sorted.

    Example
    -------
    >>> from analysis.load_h5_data import LoadData
    >>> data = LoadData('/path/to/events.h5', idxs_path='/path/to/idxs.npz', idxs_set='test_idxs')
    >>> selection = data.FC & (data.nhits > 200)
    >>> data.momenta[selection]
    >>> data.load_hit_info(0)
    >>> data.hit_charge
    """

    def __init__(self, h5path, idxs_path=None, idxs_set=None, zenith_axis=2):
        self.h5path = h5path
        self.idxs_path = idxs_path
        self.idxs=None
        
        if idxs_path and idxs_set:
            self.idxs = np.load(idxs_path, allow_pickle=True)[idxs_set]
        else:
            with h5.File(h5path) as h5file:
                self.idxs = np.array(range(len(h5file['labels'])))
            
        with h5.File(h5path) as h5file:
            self.energies       = np.array(h5file['energies'])[self.idxs].squeeze()
            self.labels         = np.array(h5file['labels'])[self.idxs].squeeze()
            self.momenta        = momentum_from_energy(self.energies, self.labels)
            self.positions      = np.array(h5file['positions'])[self.idxs].squeeze()
            self.event_ids      = np.array(h5file['event_ids'])[self.idxs].squeeze()
            self.angles         = np.array(h5file['angles'])[self.idxs].squeeze()
            self.directions     = direction_from_angles(self.angles, zenith_axis=zenith_axis)
            self.veto           = np.array(h5file['veto'])[self.idxs].squeeze()
            self.veto2          = np.array(h5file['veto2'])[self.idxs].squeeze()
            self.FC             = np.array(h5file['fully_contained'])[self.idxs].squeeze()
            self.root_files     = np.array(h5file['root_files'])[self.idxs].squeeze()
            
            self.event_hits_index = np.append(h5file['event_hits_index'], h5file['hit_pmt'].shape[0])
            
            self.nhits = self.event_hits_index[1:] - self.event_hits_index[:-1]
            
            if idxs_path and idxs_set:
                self.nhits = self.nhits[self.idxs]
            
            self.event_hits_index = np.array(h5file['event_hits_index'])[self.idxs].squeeze()
            
            try:
                self.nu_energies      = np.array(h5file['neutrino_energy'])[self.idxs].squeeze()
                self.nu_directions  = np.array(h5file['neutrino_direction'])[self.idxs].squeeze()
                self.nu_pids         = np.array(h5file['neutrino_pid'])[self.idxs].squeeze()
            except KeyError:
                pass
                
            
    def load_hit_info(self, event_idx):
        """
        Read the hits of a single event from the h5 file.

        Nothing is returned. The hits are stored on the object as hit_time,
        hit_charge and hit_pmt, replacing those of any previously loaded
        event, and loaded_event_idx is set to `event_idx`.

        Parameters
        ----------
        event_idx : int
            Index of the event within this object, i.e. its position in
            self.idxs / self.positions, NOT the event's row number in the h5
            file. If the object was loaded with an index set (e.g.
            idxs_set='test_idxs'), event_idx=0 is the first event of that set.

        Notes
        -----
        After the call:
            hit_time   : time of each hit [ns]
            hit_charge : charge of each hit
            hit_pmt    : id of the PMT that was hit, which indexes the
                         'position' array of the geometry .npz file
        each of shape (n_hits,), where n_hits == self.nhits[event_idx].

        Example
        -------
        >>> data.load_hit_info(3)
        >>> data.hit_time.min(), data.hit_charge.sum()
        """
        with h5.File(self.h5path, "r") as h5file:
            idx   = self.idxs[event_idx]
            start = h5file['event_hits_index'][idx]
            # the last event in the file has no following event, so its hits run to the end of the hit datasets
            if idx + 1 < len(h5file['event_hits_index']):
                end = h5file['event_hits_index'][idx + 1]
            else:
                end = h5file['hit_pmt'].shape[0]

            self.hit_time   = h5file['hit_time'][start:end]
            self.hit_charge = h5file['hit_charge'][start:end]
            self.hit_pmt    = h5file['hit_pmt'][start:end]
            self.loaded_event_idx = event_idx
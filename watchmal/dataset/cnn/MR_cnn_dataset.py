"""
Class for loading data from hierarchical HDF5 files with ring-based structure
"""

# torch imports
from torch import from_numpy
from torch.utils.data import Dataset

# generic imports
import h5py
import numpy as np
from glob import glob
import os
from abc import ABC

# WatChMaL imports
import watchmal.dataset.data_utils as du
from watchmal.utils.math import momentum_from_energy

class MRSegDataset(Dataset, ABC):
    """
    Dataset class for loading ring-based data from hierarchical HDF5 files.
    Compatible with CNNDataset interface for training and evaluation.
    
    Each HDF5 file contains multiple events, and each event contains multiple rings.
    Each ring is treated as a separate sample.
    
    File structure:
    /event_XXXXXX/ring_Y/energy, tube_ids, pmt_charge, pmt_time, etc.
    
    Attributes loaded per ring:
    - energy: float (SCALAR)
    - labels: int (SCALAR), converted from event_type
    - particle_dir_x, particle_dir_y, particle_dir_z: float (SCALAR)
    - particle_start_x, particle_start_y, particle_start_z: float (SCALAR)
    - tube_ids: array of int (variable length)
    - pmt_charge: array of float (variable length)
    - pmt_time: array of float (variable length)
    
    Output format matches H5CommonDataset/H5Dataset for compatibility with CNNDataset.
    """
    
    def __init__(
        self, 
        file_pattern, 
        pmt_positions_file, 
        use_times=True,
        use_charges=True,
        use_padding=False,
        padding_to_fixed_dimension=[192, 192],
        transforms=None,
        one_indexed=True,        
        use_memmap=True, 
        mask_pmts=None,
        channel_scale_factor=None,
        channel_scale_offset=None,
        use_isHit=False,
        use_positions=False,
        use_orientations=False,
        geometry_file=None,
        use_invalid_value=False,
        use_median_unhit_times=False,
        use_log_charge=False,
        label_set=None,
    ):
        """
        Initialize the hierarchical HDF5 dataset.
        
        Parameters
        ----------
        file_pattern: str
            Glob pattern matching HDF5 files (e.g., "/path/batch_*/segmented_rings_260831.h5")
        pmt_positions_file: str
            Location of an npz file containing the mapping from PMT IDs to CNN image pixel locations
        use_times: bool
            Whether to use PMT hit times as one of the initial CNN image channels. True by default.
        use_charges: bool
            Whether to use PMT hit charges as one of the initial CNN image channels. True by default.
        use_padding: bool
            Whether to pad the data to a fixed dimension (default: False).
        padding_to_fixed_dimension: list of int
            If use_padding is True, this specifies the fixed dimension to which the data will be padded.
        transforms: list
            List of random transforms to apply to data for data augmentation.
        one_indexed: bool
            Whether the PMT IDs in the H5 file are indexed starting at 1 (like SK tube numbers) or 0 (like WCSim PMT
            indexes). By default, zero-indexing is assumed.
        use_memmap: bool
            Whether to use memory mapping (not applicable for hierarchical structure, kept for API compatibility)
        mask_pmts: list of int
            List of PMT IDs to mask out from all data (None by default)
        channel_scale_factor: dict of float
            Dictionary with keys corresponding to channels and values contain the factors to divide that channel.
        channel_scale_offset: dict of float
            Dictionary with keys corresponding to channels and values contain the offsets to subtract from that channel.
        use_isHit: bool
            Whether to use a channel to tag the PMT hit or not.
        use_positions: bool
            Whether to use three channels to add the real positions info of PMTs.
        use_orientations: bool
            Whether to use three channels to add the real orientations info of PMTs.
        geometry_file: string
            Location of an npz file containing the real positions and orientations info.
        use_invalid_value: bool
            Whether to set all the channel of unhit as an invalid value (like -100).
        use_median_unhit_times: bool
            Whether to set unhit times to the median value of the normalised hit times
        use_log_charge: bool
            Whether to logarithmically transform the charge.
        label_set: sequence of labels
            Raw labels to map to indices before loading targets. For momentum regression, follow the order in momentum_from_energy()
            gamma, electron, muon, and neutral pion (e.g. [22, 11, 13, 111]). 
        """

        # Capture total number of rings to be processed
        self.h5_files = None
        self._h5_handles = {}
        self._h5_pid = None

        self.label_set = None if label_set is None else tuple(dict.fromkeys(label_set))
        self.labels_key = None
        self.target_key = None
        self.targets = None
        self.unmapped_labels = None

        self.hit_pmt = None
        self.hit_time = None
        self.hit_charges = None

        # Channel configuration (for CNN processing compatibility)
        self.use_memmap = use_memmap
        self.one_indexed = one_indexed
        self.mask_pmts = mask_pmts
        self.use_times = use_times
        self.use_charges = use_charges
        self.use_isHit = use_isHit
        self.use_positions = use_positions
        self.use_orientations = use_orientations
        self.use_invalid_value = use_invalid_value
        self.use_median_unhit_times = use_median_unhit_times
        self.use_log_charge = use_log_charge

        # Initialize PMT position mapping
        self.pmt_positions = np.load(pmt_positions_file)["pmt_image_positions"].astype(int)
        self.data_size = np.max(self.pmt_positions, axis=0) + 1
        if use_padding and padding_to_fixed_dimension is not None:
            self.data_size = np.array(padding_to_fixed_dimension)

        self.image_height, self.image_width = self.data_size[0], self.data_size[1]
        # make some index expressions for different parts of the image, to use in transformations etc
        rows, row_counts = np.unique(self.pmt_positions[:, 0], return_counts=True)
        cols, col_counts = np.unique(self.pmt_positions[:, 1], return_counts=True)
        # barrel rows are those where the row appears in mpmt_positions as many times as the image width
        barrel_rows = rows[row_counts > 0.7 * self.image_width]
        # endcap_size is the number of rows before the first barrel row
        self.endcap_size = np.min(barrel_rows)
        self.barrel = np.s_[..., self.endcap_size:np.max(barrel_rows) + 1, :]
        # endcap columns are those where the column appears more than the number of barrel rows
        endcap_cols = cols[col_counts > len(barrel_rows)]
        self.endcap_left = np.min(endcap_cols)
        self.endcap_right = np.max(endcap_cols) + 1
        self.top_endcap = np.s_[..., :self.endcap_size, self.endcap_left:self.endcap_right]
        self.bottom_endcap = np.s_[..., -self.endcap_size:, self.endcap_left:self.endcap_right]

        self.transforms = du.get_transformations(self, transforms)
        if self.transforms is None:
            self.transforms = []

        if use_positions:
            self.real_3Dpositions = np.load(geometry_file)["position"]
        else:
            self.real_3Dpositions = None
        if use_orientations:
            self.real_3Dorientations = np.load(geometry_file)["orientation"]
        else:
            self.real_3Dorientations = None

        if channel_scale_offset is None:
            channel_scale_offset = {}
        self.scale_offset = channel_scale_offset
        if channel_scale_factor is None:
            channel_scale_factor = {}
        self.scale_factor = channel_scale_factor

        self.channel_map = {}
        current_channel = 0

        if use_times:
            self.channel_map["time"] = current_channel
            current_channel += 1

        if use_charges:
            self.channel_map["charge"] = current_channel
            current_channel += 1

        if use_isHit:
            self.channel_map["isHit"] = current_channel
            current_channel += 1

        if use_positions:
            self.channel_map["position_X"] = current_channel
            self.channel_map["position_Y"] = current_channel + 1
            self.channel_map["position_Z"] = current_channel + 2
            current_channel += 3

        if use_orientations:
            self.channel_map["orientation_X"] = current_channel
            self.channel_map["orientation_Y"] = current_channel + 1
            self.channel_map["orientation_Z"] = current_channel + 2
            current_channel += 3
        if "time" not in self.channel_map and "charge" not in self.channel_map:
            raise ValueError("No time or charge information loaded.")

        self.n_channels = current_channel
        self.data_size = np.insert(self.data_size, 0, self.n_channels)

        # Find all matching HDF5 files
        self.h5_files = sorted(glob(file_pattern))
        if not self.h5_files:
            raise FileNotFoundError(f"No HDF5 files found matching pattern: {file_pattern}")
        
        print(f"Found {len(self.h5_files)} HDF5 files")
        
        # Build index of (file_idx, event_id, ring_id) tuples
        self.ring_index = []
        self._build_ring_index()
        
        print(f"Total rings (samples): {len(self.ring_index)}")
        
        # Instance variables for hit data (set during __getitem__)
        self.event_hit_pmts = None
        self.event_hit_charges = None
        self.event_hit_times = None
        self.ring_data = {}
    
    def _build_ring_index(self):
        """
        Scan all HDF5 files and build an index of available rings.
        Each entry is (file_idx, event_id, ring_id).
        """
        for file_idx, h5_path in enumerate(self.h5_files):
            try:
                with h5py.File(h5_path, 'r') as h5_file:
                    # Iterate over all events in the file
                    for event_id in sorted(h5_file.keys()):
                        if not event_id.startswith('event_'):
                            continue
                        
                        event_group = h5_file[event_id]
                        
                        # Iterate over all rings in the event
                        for ring_id in sorted(event_group.keys()):
                            if not ring_id.startswith('ring_'):
                                continue
                            
                            self.ring_index.append((file_idx, event_id, ring_id))
            
            except Exception as e:
                print(f"Error reading {h5_path}: {e}")
                raise

    def _get_h5_file(self, file_idx):
        """Lazily open each file once in the process that reads the samples."""
        pid = os.getpid()
        if self._h5_pid != pid:
            # A forked worker must not read through handles inherited from its parent.
            self.close()
            self._h5_pid = pid
        if file_idx not in self._h5_handles:
            self._h5_handles[file_idx] = h5py.File(self.h5_files[file_idx], "r")
        return self._h5_handles[file_idx]

    def close(self):
        """Release this process's cached HDF5 handles."""
        for handle in self._h5_handles.values():
            handle.close()
        self._h5_handles = {}
        self._h5_pid = None

    def __getstate__(self):
        # Spawned DataLoader workers open their own handles on first access.
        state = self.__dict__.copy()
        state["_h5_handles"] = {}
        state["_h5_pid"] = None
        return state

    def __del__(self):
        if hasattr(self, "_h5_handles"):
            self.close()

    def _load_ring_data(self, file_idx, event_id, ring_id):
        """
        Load data for a specific ring from an HDF5 file.
        
        Parameters
        ----------
        file_idx: int
            Index of the HDF5 file
        event_id: str
            Event ID (e.g., 'event_000000')
        ring_id: str
            Ring ID (e.g., 'ring_0')
        
        Returns
        -------
        dict
            Dictionary containing ring data
        """
        h5_file = self._get_h5_file(file_idx)
        ring_group = h5_file[event_id][ring_id]

        # Load scalar attributes
        energy = float(ring_group['energy'][()])
        labels = int(ring_group['event_type'][()])
        particle_dir_x = float(ring_group['particle_dir_x'][()])
        particle_dir_y = float(ring_group['particle_dir_y'][()])
        particle_dir_z = float(ring_group['particle_dir_z'][()])

        particle_start_x = float(ring_group['particle_start_x'][()])
        particle_start_y = float(ring_group['particle_start_y'][()])
        particle_start_z = float(ring_group['particle_start_z'][()])
            
        # Load array attributes
        tube_ids = np.array(ring_group['tube_ids'], dtype=np.int32)
        pmt_charge = np.array(ring_group['pmt_charge'], dtype=np.float32)
        pmt_time = np.array(ring_group['pmt_time'], dtype=np.float32)
            
        ring_dict = {
            'energy': energy,
            'labels': labels,
            'particle_dir': np.array([particle_dir_x, particle_dir_y, particle_dir_z], dtype=np.float32),
            'particle_start': np.array([particle_start_x, particle_start_y, particle_start_z], dtype=np.float32),
            'tube_ids': tube_ids,
            'pmt_charge': pmt_charge,
            'pmt_time': pmt_time,
            'n_hits': len(tube_ids),
        }
            
        return ring_dict
    
    def process_data(self, hit_pmts, hit_times, hit_charges):
        if self.one_indexed:
            hit_pmts = hit_pmts - 1 # cable numbers start at 1

        hit_rows = self.pmt_positions[hit_pmts, 0]
        hit_cols = self.pmt_positions[hit_pmts, 1]

        invalid_value = 0.0
        if self.use_invalid_value:
            invalid_value = -100.0

        time_offset = self.scale_offset.get("time", 0.0)
        time_scale = self.scale_factor.get("time", 1.0)
        charge_offset = self.scale_offset.get("charge", 0.0)
        charge_scale = self.scale_factor.get("charge", 1.0)
        positions_offset = self.scale_offset.get("positions", 0.0)
        positions_scale = self.scale_factor.get("positions", 1.0)
        orientations_offset = self.scale_offset.get("orientations", 0.0)
        orientations_scale = self.scale_factor.get("orientations", 1.0)

        if self.use_log_charge:
            hit_charges = np.log10(hit_charges)
        data = np.full(self.data_size, invalid_value, dtype=np.float32)
        if self.use_positions:
            data[
                self.channel_map["position_X"],
                self.pmt_positions[:, 0],
                self.pmt_positions[:, 1],
            ] = (self.real_3Dpositions[:, 0] - positions_offset) / positions_scale
            data[
                self.channel_map["position_Y"],
                self.pmt_positions[:, 0],
                self.pmt_positions[:, 1],
            ] = (self.real_3Dpositions[:, 1] - positions_offset) / positions_scale
            data[
                self.channel_map["position_Z"],
                self.pmt_positions[:, 0],
                self.pmt_positions[:, 1],
            ] = (self.real_3Dpositions[:, 2] - positions_offset) / positions_scale
        if self.use_orientations:
            data[
                self.channel_map["orientation_X"],
                self.pmt_positions[:, 0],
                self.pmt_positions[:, 1],
            ] = (
                self.real_3Dorientations[:, 0] - orientations_offset
            ) / orientations_scale
            data[
                self.channel_map["orientation_Y"],
                self.pmt_positions[:, 0],
                self.pmt_positions[:, 1],
            ] = (
                self.real_3Dorientations[:, 1] - orientations_offset
            ) / orientations_scale
            data[
                self.channel_map["orientation_Z"],
                self.pmt_positions[:, 0],
                self.pmt_positions[:, 1],
            ] = (
                self.real_3Dorientations[:, 2] - orientations_offset
            ) / orientations_scale

        if "time" in self.channel_map:
            normalised_times = (hit_times - time_offset) / time_scale
            if self.use_median_unhit_times:
                median_time = np.median(normalised_times)
                data[self.channel_map["time"]] = median_time
            data[self.channel_map["time"], hit_rows, hit_cols] = normalised_times
        if "charge" in self.channel_map:
            data[self.channel_map["charge"], hit_rows, hit_cols] = (
                hit_charges - charge_offset
            ) / charge_scale
        if "isHit" in self.channel_map:
            data[self.channel_map["isHit"], hit_rows, hit_cols] = 1.0

        return data
    
    def __len__(self):
        """Return total number of rings (samples) in the dataset."""
        return len(self.ring_index)
    
    def __getitem__(self, idx):
        """
        Get a ring sample by index.
        
        Returns dict in same format as H5Dataset.__getitem__(), compatible with CNNDataset.
        
        Parameters
        ----------
        idx: int
            Index of the sample
        
        Returns
        -------
        dict
            Dictionary containing targets (energies, labels, positions, angles) and indices.
            Instance variables event_hit_pmts, event_hit_charges, event_hit_times are set
            for use by CNNDataset.process_data().
        """
        file_idx, event_id, ring_id = self.ring_index[idx]        
        self.ring_data = self._load_ring_data(file_idx, event_id, ring_id)

        # Convert raw labels into sequential labels before computing three_momenta
        self.unmapped_labels = np.asarray(self.ring_data['labels'])
        if self.label_set is not None:
            self.map_labels(self.label_set)
        else:
            self.set_target(self.target_key)

        data_dict = {k: t.copy() for k, t in self.targets.items()}
        data_dict["indices"] = idx

        self.event_hit_pmts = self.ring_data['tube_ids']
        self.event_hit_charges = self.ring_data['pmt_charge']
        self.event_hit_times = self.ring_data['pmt_time']

        if self.mask_pmts is not None:
            mask = np.isin(self.event_hit_pmts, self.mask_pmts, invert=True)
            self.event_hit_pmts = self.event_hit_pmts[mask]
            self.event_hit_charges = self.event_hit_charges[mask]
            self.event_hit_times = self.event_hit_times[mask]
        data_dict['n_hits'] = len(self.event_hit_charges)
        data_dict['total_charge'] = np.sum(self.event_hit_charges)

        processed_data = self.process_data(
            self.event_hit_pmts,
            self.event_hit_times,
            self.event_hit_charges
        )
        
        # Apply transformations
        data_dict["data"] = processed_data
        for t in self.transforms:
            data_dict = t(data_dict)
        data_dict["data"] = from_numpy(data_dict["data"].copy())
        
        return data_dict

    def set_target(self, target_key):
        self.target_key = target_key
        if self.target_key is None:
            self.targets = {}
            return
        try:
            if isinstance(self.target_key, str):
                self.targets = {target_key: self.load_target(target_key)}
            else:
                self.targets = {t: self.load_target(t) for t in self.target_key}
        except KeyError:
            # truth info don't exist, can only predict but not train or evaluate
            self.targets = {}

    def load_target(self, target_key):
        '''
        target_key: directions, 
                    three_momenta, 
                    log_momenta, 
                    positions,
                    labels,
                    energy,
        '''
        if target_key == "directions":
            return self.ring_data['particle_dir']
        elif target_key == "three_momenta":
            directions = self.ring_data['particle_dir']
            momenta = momentum_from_energy(self.ring_data['energy'], self.ring_data['labels'])[..., None]
            return directions*momenta
        elif target_key == "log_momenta":
            momenta = momentum_from_energy(self.ring_data['energy'], self.ring_data['labels'])[..., None]
            return np.log(momenta)
        elif target_key == "positions":
            return self.ring_data['particle_start']
        else:
            return np.array(self.ring_data[target_key]).squeeze()

    def map_labels(self, label_set):
        """
        Map raw ring labels before loading classification or regression targets.
        Labels are mapped to their positions in the configured sequence. For
        momentum regression, this order must match momentum_from_energy's mass
        table: gamma, electron, muon, neutral pion. The original raw label is kept
        so repeated calls cannot remap an already mapped value.
        """
        self.label_set = tuple(dict.fromkeys(label_set))
        if self.unmapped_labels is not None:
            if not np.all(np.isin(self.unmapped_labels, self.label_set)):
                raise ValueError(f"Label {self.unmapped_labels} is not in label_set {self.label_set}")
            labels = np.empty(self.unmapped_labels.shape, dtype=np.int64)
            for i, label in enumerate(self.label_set):
                labels[self.unmapped_labels == label] = i
            self.ring_data['labels'] = labels
            self.set_target(self.target_key)

    def double_cover(self, data_dict):
        """
        Takes CNN input data in event-display-like format and returns the data with all parts of the detector duplicated
        and rearranged to provide a double-cover of the image, providing two 'views' of the detector from a single image
        with less blank space and physically meaningful cyclic boundary conditions at the edges of the image.

        Since CNNDataset uses a simple PMT grid (1 PMT per pixel) instead of mPMT (19 PMTs per pixel), this version
        is simpler - no channel permutations are needed.

        The transformation looks something like the following, where PMTs on the end caps are numbered and PMTs on the
        barrel are letters:
        ```
                                         CBALKJIHGFED
                         01                01    32
                         23                23    10
                    ABCDEFGHIJKL   -->   DEFGHIJKLABC
                    MNOPQRSTUVWX         PQRSTUVWXMNO
                         45                45    76
                         67                67    54
                                         ONMXWVUSTRQP
        ```
        """
        # Make copies of the endcaps, flipped (180° rotated), to use later
        top_endcap_copy = np.flip(data_dict["data"][self.top_endcap], [1, 2])
        bottom_endcap_copy = np.flip(data_dict["data"][self.bottom_endcap], [1, 2])
        # Roll the tensor so that the first quarter is the last quarter
        quarter_barrel_width = self.image_width // 4
        data = np.roll(data_dict["data"], -quarter_barrel_width, 2)
        # Paste the copied flipped endcaps a quarter barrel-width past the original endcap position
        endcap_copy_columns = np.s_[quarter_barrel_width + self.endcap_left: quarter_barrel_width + self.endcap_right]
        data[..., :self.endcap_size, endcap_copy_columns] = top_endcap_copy
        data[..., -self.endcap_size:, endcap_copy_columns] = bottom_endcap_copy
        # Rotate the bottom and top halves of barrel and concatenate to the top and bottom of the image
        # If the endcaps are offset from the middle of the image, need to roll the flipped barrel to keep the same offset
        offset = (self.image_width - self.endcap_right) - self.endcap_left
        barrel_rolled = np.roll(data[self.barrel], offset, 2)
        barrel_bottom_flipped, barrel_top_flipped = np.array_split(np.flip(barrel_rolled, [1, 2]), 2, axis=1)
        data_dict["data"] = np.concatenate((barrel_top_flipped, data, barrel_bottom_flipped), axis=1)
        return data_dict
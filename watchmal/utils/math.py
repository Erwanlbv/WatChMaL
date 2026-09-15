"""
Utility functions for performing mathematical, physical, statistical, geometrical operations
"""

import logging
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np


log = logging.getLogger(__name__)

DEFAULT_TANK_AXIS = 1

# Rest masses in MeV, keyed by the absolute value of the PDG Monte Carlo particle code: an antiparticle carries the
# negative of its particle's code and has the same mass. Values from the Particle Data Group, S. Navas et al., "Review
# of Particle Physics", Phys. Rev. D 110, 030001 (2024). Read-only, since it is the default argument of
# momentum_from_energy and energy_from_momentum.
PDG_MASSES_MEV = MappingProxyType({
    11: 0.51099895,     # electron
    13: 105.6583755,    # muon
    22: 0.0,            # photon
    111: 134.9768,      # neutral pion
    211: 139.57039,     # charged pion
})

# Species names, used only to state the lookup convention in the log.
_PDG_SPECIES = {11: "electron", 13: "muon", 22: "photon", 111: "neutral pion", 211: "charged pion"}

# Whether the lookup convention has been logged in this process (see _masses_from_pdg_codes).
_mass_convention_logged = False


def towall(position, angle, tank_half_height, tank_radius, tank_axis=None):
    """
    Calculate towall: distance from position to detector wall, in particle direction

    Parameters
    ----------
    position : array_like
        vector of (x, y, z) position of a particle or (N,3) array of positions for N particles
    angle : array_like
        vector of (zenith, azimuth) direction of particle or (N, 2) array of directions for N particles
    tank_half_height : float
        half-height of the detector ID
    tank_radius : float
        Radius of the detector ID
    tank_axis : int, optional
        Axis along which the tank cylinder is oriented. By default, use the y-axis.

    Returns
    -------
    np.ndarray or scalar
        array of towall values for each position, or scalar if only one position
    """
    if tank_axis is None:
        tank_axis = DEFAULT_TANK_AXIS
    pos_trans = np.delete(position, tank_axis, axis=-1)
    pos_along = position[..., tank_axis]
    dir_along, dir_trans = polar_to_cartesian(angle)
    a = np.linalg.norm(dir_trans, axis=-1)**2
    b = np.sum(pos_trans*dir_trans, axis=-1)
    c = np.linalg.norm(pos_trans, axis=-1) ** 2 - tank_radius ** 2
    towall_barrel = (-b + np.sqrt(b**2-a*c)) / a
    towall_endcap = tank_half_height / abs(dir_along) - pos_along / dir_along
    return np.minimum(towall_barrel, towall_endcap)


def dwall(position, tank_half_height, tank_radius, tank_axis=None):
    """
    Calculate dwall: distance from position to the nearest detector wall

    Parameters
    ----------
    position : array_like
        vector of (x, y, z) position of an event or (N,3) array of (x, y, z) position coordinates for N events
    tank_half_height : float
        half-height of the detector ID
    tank_radius : float
        Radius of the detector ID
    tank_axis : int, optional
        Axis along which the tank cylinder is oriented. By default, use y-axis

    Returns
    -------
    np.ndarray or scalar
        array of dwall values for each position, or scalar if only one position
    """
    if tank_axis is None:
        tank_axis = DEFAULT_TANK_AXIS
    pos_along = position[..., tank_axis]
    pos_trans = np.delete(position, tank_axis, axis=-1)
    dwall_barrel = tank_radius - np.linalg.norm(pos_trans, axis=-1)
    dwall_endcap = tank_half_height - np.abs(pos_along)
    return np.minimum(dwall_barrel, dwall_endcap)


def _reset_mass_convention_warning():
    """Re-arm the once-per-process warning that states the mass lookup convention. For tests only."""
    global _mass_convention_logged
    _mass_convention_logged = False


def log_mass_convention(particle_masses=PDG_MASSES_MEV):
    """
    Log, once per process, a warning that states the particle mass lookup convention and the mass table

    The first call in a process logs the warning and sets a module-level flag; later calls do nothing. The
    momentum-energy conversions call this function before each lookup. `H5CommonDataset.set_target` also calls it when
    the 'three_momenta' target is requested, in the process that builds the data loaders. When the loaders have
    workers, the targets are loaded later in them; workers started by fork (`watchmal/dataset/data_utils.py`) inherit
    the flag, so the warning is logged once in that process instead of once per worker.

    Parameters
    ----------
    particle_masses : Mapping, default: PDG_MASSES_MEV
        mapping from absolute PDG code to particle mass in MeV, listed in the warning
    """
    global _mass_convention_logged
    if _mass_convention_logged:
        return
    _mass_convention_logged = True
    table = "; ".join(f"{c} {_PDG_SPECIES.get(c, 'unnamed')} {float(particle_masses[c])}"
                      for c in sorted(particle_masses))
    log.warning("Particle masses for the momentum-energy conversion are looked up by PDG code: labels are read "
                "as PDG Monte Carlo particle codes and matched by absolute value, so an antiparticle takes the "
                "mass of its particle. Table (|PDG code| species mass/MeV): %s. Any label whose absolute value is "
                "not in the table raises ValueError; a mapped class index passes only if it equals a listed "
                "code.", table)


def _masses_from_pdg_codes(pdg_code, particle_masses):
    """
    Look up particle masses by the absolute value of the PDG code

    Calls `log_mass_convention` with `particle_masses` before the lookup. A label whose absolute value has no entry in
    the table raises, rather than receiving some mass. The former lookup indexed a mass array with the label, which
    gave a wrong mass without any error to labels in another convention: muons stored as class index 0 of the
    {13: 0, 11: 1, 22: 2} mapping received the photon mass.

    Parameters
    ----------
    pdg_code : array_like or int
        PDG code of particle type or array of PDG codes of particles
    particle_masses : Mapping
        mapping from absolute PDG code to particle mass in MeV

    Returns
    -------
    np.ndarray or scalar
        float64 array of masses with the shape of `pdg_code`, or scalar if only one code

    Raises
    ------
    TypeError
        if `particle_masses` is not a mapping
    ValueError
        if the absolute value of any code has no entry in `particle_masses`
    """
    if not isinstance(particle_masses, Mapping):
        raise TypeError(f"particle_masses must be a mapping from absolute PDG code to mass in MeV, not "
                        f"{type(particle_masses).__name__}; masses are not indexed by class label")
    log_mass_convention(particle_masses)
    table_codes = sorted(particle_masses)
    table_masses = np.array([particle_masses[c] for c in table_codes], dtype=np.float64)
    table_codes = np.array(table_codes, dtype=np.int64)
    codes = np.abs(np.asarray(pdg_code))
    known = np.isin(codes, table_codes)
    if not np.all(known):
        unknown = np.unique(np.asarray(pdg_code)[~known]).tolist()
        shown = str(unknown) if len(unknown) <= 20 else f"{unknown[:20]} and {len(unknown) - 20} further values"
        raise ValueError(f"Particle labels {shown} have no entry in the mass table, which is keyed by |PDG code| "
                         f"{table_codes.tolist()}. Labels must be PDG Monte Carlo particle codes; mapped class "
                         f"indices, such as those of a {{13: 0, 11: 1, 22: 2}} mapping, must be converted back to PDG "
                         f"codes first. The dataset target 'three_momenta' reads the 'labels' array of the HDF5 file "
                         f"directly, and H5CommonDataset.map_labels does not apply to it, so a file that stores mapped "
                         f"class indices must be rewritten with PDG codes in 'labels'.")
    return table_masses[np.searchsorted(table_codes, codes)]


def momentum_from_energy(energy, pdg_code, particle_masses=PDG_MASSES_MEV):
    """
    Calculate momentum of particle from total energy and particle type, given as a PDG code

    The mass is looked up by the absolute value of the PDG code, so an antiparticle has the mass of its particle. A
    label whose absolute value has no entry in `particle_masses` raises ValueError. Class indices produced by a label
    mapping are therefore rejected unless they coincide with a listed code: with the default table, a class index of
    11, 13, 22, 111 or 211 would be taken as that species. Unless `log_mass_convention` has already run in the process,
    the call logs a warning stating this convention and the mass table.

    Parameters
    ----------
    energy : array_like or scalar
        total energy of particle in MeV, or vector of energies of particles
    pdg_code : array_like or int
        PDG code of particle type or vector of PDG codes of particles
    particle_masses : Mapping, default: PDG_MASSES_MEV
        mapping from absolute PDG code to particle mass in MeV

    Returns
    -------
    np.ndarray or scalar
        array of momentum values for each energy, or scalar if only one energy

    Raises
    ------
    ValueError
        if the absolute value of any code has no entry in `particle_masses`
    """
    energy = np.asarray(energy)
    mass = np.asarray(_masses_from_pdg_codes(pdg_code, particle_masses), dtype=energy.dtype)
    mass = mass[(...,) + (None,) * (energy.ndim - mass.ndim)] # add axes to broadcast to energy along first axis/axes
    return np.sqrt(energy**2 - mass**2)


def energy_from_momentum(momentum, pdg_code, particle_masses=PDG_MASSES_MEV):
    """
    Calculate total energy of particle from momentum and particle type, given as a PDG code

    The mass is looked up by the absolute value of the PDG code, so an antiparticle has the mass of its particle. A
    label whose absolute value has no entry in `particle_masses` raises ValueError. Class indices produced by a label
    mapping are therefore rejected unless they coincide with a listed code: with the default table, a class index of
    11, 13, 22, 111 or 211 would be taken as that species. Unless `log_mass_convention` has already run in the process,
    the call logs a warning stating this convention and the mass table.

    Parameters
    ----------
    momentum : array_like
        momentum of particle in MeV, or vector of momenta of particles
    pdg_code : array_like or int
        PDG code of particle type or vector of PDG codes of particles
    particle_masses : Mapping, default: PDG_MASSES_MEV
        mapping from absolute PDG code to particle mass in MeV

    Returns
    -------
    np.ndarray or scalar
        array of energy values for each momentum, or scalar if only one momentum

    Raises
    ------
    ValueError
        if the absolute value of any code has no entry in `particle_masses`
    """
    mass = _masses_from_pdg_codes(pdg_code, particle_masses)
    return np.sqrt(momentum**2 + mass**2)


def polar_to_cartesian(angles):
    """
    Calculate (x,y,z) unit vector from azimuth and zenith angles

    Parameters
    ----------
    angles : array_like
        vector of (zenith, azimuth) of a direction or (N,2) array of (zenith, azimuth) angles for N directions

    Returns
    -------
    dir_along: np.ndarray or scalar
        array of the component along zenith direction for unit vector of each direction, or scalar if only one direction
    dir_trans: np.ndarray
        array of the components transverse to zenith direction for unit vector of each direction
    """
    zenith = angles[..., 0]
    azimuth = angles[..., 1]
    dir_along = np.cos(zenith)
    dir_trans = np.column_stack((np.sin(zenith) * np.cos(azimuth), np.sin(zenith) * np.sin(azimuth)))
    return dir_along, dir_trans


def direction_from_angles(angles, zenith_axis=None):
    """
    Calculate unit vector from azimuth and zenith angles

    Parameters
    ----------
    angles : array_like
        vector of (zenith, azimuth) of a direction or (N,2) array of (zenith, azimuth) angles for N directions
    zenith_axis : int, optional
        Axis along which the zenith angle is relative to (i.e. the axis the tank is oriented). By default, use y-axis.

    Returns
    -------
    np.ndarray
        array of unit vectors of each direction
    """
    dir_along, dir_trans = polar_to_cartesian(angles)
    if zenith_axis is None:
        zenith_axis = DEFAULT_TANK_AXIS
    return np.insert(dir_trans, zenith_axis, dir_along, axis=1)


def angles_from_direction(direction, zenith_axis=None):
    """
    Calculate azimuth and zenith angles from unit vector

    Parameters
    ----------
    direction : array_like
        vector of (x,y,z) components of a unit vector of a direction, or (N,3) array of (x,y,z) unit vector directions
    zenith_axis : int, optional
        Axis along which the zenith angle is relative to (i.e. the axis the tank is oriented). By default, use y-axis.

    Returns
    -------
    np.ndarray
        array of (zenith, azimuth) angles of each direction

    """
    if zenith_axis is None:
        zenith_axis = DEFAULT_TANK_AXIS
    dir_along = direction[..., zenith_axis]
    dir_trans = np.delete(direction, zenith_axis, axis=-1)
    zenith = np.arccos(dir_along)
    azimuth = np.arctan2(dir_trans[..., 1], dir_trans[..., 0])
    return np.column_stack((zenith, azimuth))


def angle_between_directions(direction1, direction2, degrees=False):
    """
    Calculate angle between two directions

    Parameters
    ----------
    direction1 : array_like
        vector of (x,y,z) components of a unit vector of a direction, or (N,3) array of (x,y,z) unit vector directions
    direction2 : array_like
        vector of (x,y,z) components of a unit vector of a direction, or (N,3) array of (x,y,z) unit vector directions
    degrees : bool, default: False
        if True, return values in degrees (otherwise radians)

    Returns
    -------
    angle: np.ndarray or scalar
        array of angles between direction1 and direction2, or scalar if direction1 and direction2 are single directions
    """
    angle = np.arccos(np.clip(np.einsum('...i,...i', direction1, direction2), -1.0, 1.0))
    if degrees:
        angle *= 180/np.pi
    return angle


def decompose_along_direction(vector, direction):
    """
    Decompose vector into longitudinal and transverse components along some direction

    Parameters
    ----------
    vector: np.ndarray
        vector of (x,y,z) components or (N,3) array of N (x,y,z) vectors
    direction: np.ndarray
        vector of (x,y,z) components of a unit vector of a direction, or (N,3) array of (x,y,z) unit vector directions

    Returns
    -------
    total_magnitude: np.ndarray or scalar
        array of magnitudes of each vector, or scalar if only one vector
    longitudinal_component: np.ndarray or scalar
        array of component of each vector along direction, or scalar if only one vector
    transverse_component: np.ndarray or scalar
        array of component of each vector transverse to direction, or scalar if only one vector
    """
    total_magnitude = np.linalg.norm(vector, axis=-1)
    longitudinal_component = np.einsum('...i,...i', vector, direction)
    transverse_component = np.sqrt(np.maximum(total_magnitude**2-longitudinal_component**2, 0))
    return total_magnitude, longitudinal_component, transverse_component


def binomial_error(x):
    """
    Calculate binomial standard error of an array of booleans

    Parameters
    ----------
    x: array_like
        array of booleans corresponding to binomial trial results

    Returns
    -------
    scalar
        binomial standard error of x
    """
    x = np.array(x)
    trials = x.size
    if trials == 0:
        return 0
    p = np.count_nonzero(x)/trials
    return np.sqrt(p*(1-p)/trials)

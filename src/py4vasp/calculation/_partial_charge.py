# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
import warnings
from typing import Union

import numpy as np

from py4vasp import exception
from py4vasp._third_party.graph import Graph
from py4vasp._third_party.graph.contour import Contour
from py4vasp._util import import_, select
from py4vasp._util.slicing import plane
from py4vasp.calculation import _base, _structure

interpolate = import_.optional("scipy.interpolate")
ndimage = import_.optional("scipy.ndimage")

_STM_MODES = {
    "constant_height": ["constant_height", "ch", "height"],
    "constant_current": ["constant_current", "cc", "current"],
}
_SPINS = ("up", "down", "total")


@dataclasses.dataclass
class STM_settings:
    """Settings for the STM simulation.

    sigma_z : float
        The standard deviation of the Gaussian filter in the z-direction.
        The default is 4.0.
    sigma_xy : float
        The standard deviation of the Gaussian filter in the xy-plane.
        The default is 4.0.
    truncate : float
        The truncation of the Gaussian filter. The default is 3.0.
    enhancement_factor : float
        The enhancement factor for the output of the constant heigth
        STM image. The default is 1000.
    interpolation_factor : int
        The interpolation factor for the z-direction in case of
        constant current mode. The default is 10.
    """

    sigma_z: float = 4.0
    sigma_xy: float = 4.0
    truncate: float = 3.0
    enhancement_factor: float = 1000
    interpolation_factor: int = 10


class PartialCharge(_base.Refinery, _structure.Mixin):
    """Partial charges describe the fraction of the charge density in a certain energy,
    band, or k-point range.

    Partial charges are produced by a post-processing VASP run after self-consistent
    convergence is achieved. They are stored in an array of shape
    (ngxf, ngyf, ngzf, ispin, nbands, nkpts). The first three dimensions are the
    FFT grid dimensions, the fourth dimension is the spin index, the fifth dimension
    is the band index, and the sixth dimension is the k-point index. Both band and
    k-point arrays are also saved and accessible in the .bands() and kpoints() methods.
    If ispin=2, the second spin index is the magnetization density (up-down),
    not the down-spin density.
    Since this is postprocessing data for a fixed density, there are no ionic steps
    to separate the data.
    """

    @property
    def stm_settings(self):
        return STM_settings()

    @_base.data_access
    def __str__(self):
        """Return a string representation of the partial charge density."""
        return f"""
        {"spin polarized" if self._spin_polarized() else ""} partial charge density of {self._topology()}:
        on fine FFT grid: {self.grid()}
        {"summed over all contributing bands" if 0 in self.bands() else f" separated for bands: {self.bands()}"}
        {"summed over all contributing k-points" if 0 in self.kpoints() else f" separated for k-points: {self.kpoints()}"}
        """.strip()

    @_base.data_access
    def grid(self):
        return self._raw_data.grid[:]

    @_base.data_access
    def to_dict(self):
        """Store the partial charges in a dictionary.

        Returns
        -------
        dict
            The dictionary contains the partial charges as well as the structural
            information for reference.
        """

        parchg = np.squeeze(self._raw_data.partial_charge[:].T)
        return {
            "structure": self._structure.read(),
            "grid": self.grid(),
            "bands": self.bands(),
            "kpoints": self.kpoints(),
            "partial_charge": parchg,
        }

    @_base.data_access
    def to_stm(
        self,
        selection: str = "constant_height",
        *,
        tip_height: float = 2.0,
        current: float = 1.0,
        supercell: Union[int, np.array] = 2,
        stm_settings: STM_settings = STM_settings(),
    ) -> Graph:
        """Generate STM image data from the partial charge density.

        Parameters
        ----------
        selection : str
            The mode in which the STM is operated and the spin channel to be used.
            Possible modes are "constant_height"(default) and "constant_current".
            Possible spin selections are "total"(default), "up", and "down".
        tip_height : float
            The height of the STM tip above the surface in Angstrom.
            The default is 2.0 Angstrom. Only used in "constant_height" mode.
        current : float
            The tunneling current in nA. The default is 1.
            Only used in "constant_current" mode.
        supercell : int | np.array
            The supercell to be used for plotting the STM. The default is 2.
        stm_settings : STM_settings
            Settings for the STM simulation concerning smoothening parameters
            and interpolation. The default is STM_settings().

        Returns
        -------
        Graph
            The STM image as a graph object. The title is the label of the Contour
            object.
        """
        _raise_error_if_vacuum_too_small(self._estimate_vacuum())

        tree = select.Tree.from_selection(selection)
        for index, selection in enumerate(tree.selections()):
            if index > 0:
                message = "Selecting more than one STM is not implemented."
                raise exception.NotImplemented(message)
            contour = self._make_contour(selection, tip_height, current, stm_settings)
        contour.supercell = self._parse_supercell(supercell)
        contour.settings = stm_settings
        return Graph(series=contour, title=contour.label)

    def _parse_supercell(self, supercell):
        if isinstance(supercell, int):
            return np.asarray([supercell, supercell])
        if len(supercell) == 2:
            return np.asarray(supercell)
        message = """The supercell has to be a single number or a 2D array. \
        The supercell is used to multiply the x and y directions of the lattice."""
        raise exception.IncorrectUsage(message)

    def _make_contour(self, selection, tip_height, current, stm_settings):
        self._raise_error_if_tip_too_far_away(tip_height)
        mode = self._parse_mode(selection)
        spin = self._parse_spin(selection)
        self._raise_error_if_selection_not_understood(selection, mode, spin)
        smoothed_charge = self._get_stm_data(spin, stm_settings)
        if mode == "constant_height" or mode is None:
            return self._constant_height_stm(
                smoothed_charge, tip_height, spin, stm_settings
            )
        current = current * 1e-09  # convert nA to A
        return self._constant_current_stm(smoothed_charge, current, spin, stm_settings)

    def _parse_mode(self, selection):
        for mode, aliases in _STM_MODES.items():
            for alias in aliases:
                if select.contains(selection, alias, ignore_case=True):
                    return mode
        return None

    def _parse_spin(self, selection):
        for spin in _SPINS:
            if select.contains(selection, spin, ignore_case=True):
                return spin
        return None

    def _raise_error_if_selection_not_understood(self, selection, mode, spin):
        if len(selection) != int(mode is not None) + int(spin is not None):
            message = f"STM mode '{selection}' was parsed as mode='{mode}' and spin='{spin}' which could not be used. Please use 'constant_height' or 'constant_current' as mode and 'up', 'down', or 'total' as spin."
            raise exception.IncorrectUsage(message)

    def _constant_current_stm(self, smoothed_charge, current, spin, stm_settings):
        z_start = _min_of_z_charge(
            self._get_stm_data(spin, stm_settings),
            sigma=stm_settings.sigma_z,
            truncate=stm_settings.truncate,
        )
        grid = self.grid()
        z_step = 1 / stm_settings.interpolation_factor
        # roll smoothed charge so that we are not bothered by the boundary of the
        # unit cell if the slab is not centered. z_start is now the first index
        smoothed_charge = np.roll(smoothed_charge, -z_start, axis=2)
        z_grid = np.arange(grid[2], 0, -z_step)
        splines = interpolate.CubicSpline(range(grid[2]), smoothed_charge, axis=-1)
        scan = z_grid[np.argmax(splines(z_grid) >= current, axis=-1)]
        scan = z_step * (scan - scan.min())
        spin_label = "both spin channels" if spin == "total" else f"spin {spin}"
        topology = self._topology()
        label = f"STM of {topology} for {spin_label} at constant current={current*1e9:.2f} nA"
        return Contour(data=scan, lattice=self._get_stm_plane(), label=label)

    def _constant_height_stm(self, smoothed_charge, tip_height, spin, stm_settings):
        zz = self._z_index_for_height(tip_height + self._get_highest_z_coord())
        height_scan = smoothed_charge[:, :, zz] * stm_settings.enhancement_factor
        spin_label = "both spin channels" if spin == "total" else f"spin {spin}"
        topology = self._topology()
        label = f"STM of {topology} for {spin_label} at constant height={float(tip_height):.2f} Angstrom"
        return Contour(data=height_scan, lattice=self._get_stm_plane(), label=label)

    def _z_index_for_height(self, tip_height):
        """Return the z-index of the tip height in the charge density grid."""
        # In case the surface is very up in the unit cell, we have to wrap around
        return round(
            np.mod(
                tip_height / self._out_of_plane_vector() * self.grid()[2],
                self.grid()[2],
            )
        )

    def _height_from_z_index(self, z_index):
        """Return the height of the z-index in the charge density grid."""
        return z_index * self._out_of_plane_vector() / self.grid()[2]

    def _get_highest_z_coord(self):
        cart_coords = _get_sanitized_cartesian_positions(self._structure)
        return np.max(cart_coords[:, 2])

    def _get_lowest_z_coord(self):
        cart_coords = _get_sanitized_cartesian_positions(self._structure)
        return np.min(cart_coords[:, 2])

    def _topology(self):
        return str(self._structure._topology())

    def _estimate_vacuum(self):
        _raise_error_if_vacuum_not_along_z(self._structure)
        slab_thickness = self._get_highest_z_coord() - self._get_lowest_z_coord()
        return self._out_of_plane_vector() - slab_thickness

    def _raise_error_if_tip_too_far_away(self, tip_height):
        if tip_height > self._estimate_vacuum() / 2:
            message = f"""The tip position at {tip_height:.2f} is above half of the
             estimated vacuum thickness {self._estimate_vacuum():.2f} Angstrom.
            You would be sampling the bottom of your slab, which is not supported."""
            raise exception.IncorrectUsage(message)

    def _get_stm_data(self, spin, stm_settings):
        if 0 not in self.bands() or 0 not in self.kpoints():
            massage = """Simulated STM images are only supported for non-separated bands and k-points.
            Please set LSEPK and LSEPB to .FALSE. in the INCAR file."""
            raise exception.NotImplemented(massage)
        chg = self._correct_units(self.to_numpy(spin, band=0, kpoint=0))
        return self._smooth_stm_data(chg, stm_settings)

    def _correct_units(self, charge_data):
        grid_volume = np.prod(self.grid())
        cell_volume = self._structure.volume()
        return charge_data / (grid_volume * cell_volume)

    def _smooth_stm_data(self, data, stm_settings):
        sigma = (
            stm_settings.sigma_xy,
            stm_settings.sigma_xy,
            stm_settings.sigma_z,
        )
        return ndimage.gaussian_filter(
            data, sigma=sigma, truncate=stm_settings.truncate, mode="wrap"
        )

    def _get_stm_plane(self):
        """Return lattice plane spanned by a and b vectors"""
        return plane(
            cell=self._structure.lattice_vectors(),
            cut="c",
            normal="z",
        )

    def _out_of_plane_vector(self):
        """Return out-of-plane component of lattice vectors."""
        lattice_vectors = self._structure.lattice_vectors()
        _raise_error_if_vacuum_not_along_z(self._structure)
        return lattice_vectors[2, 2]

    def _spin_polarized(self):
        return self._raw_data.partial_charge.shape[2] == 2

    @_base.data_access
    def to_numpy(self, selection="total", band=0, kpoint=0):
        """Return the partial charge density as a 3D array.

        Parameters
        ----------
        selection : str
            The spin channel to be used. The default is "total".
            The other options are "up" and "down".
        band : int
            The band index. The default is 0, which means that all bands are summed.
        kpoint : int
            The k-point index. The default is 0, which means that all k-points are summed.

        Returns
        -------
        np.array
            The partial charge density as a 3D array.
        """

        band = self._check_band_index(band)
        kpoint = self._check_kpoint_index(kpoint)

        parchg = self._raw_data.partial_charge[:].T
        if not self._spin_polarized() or selection == "total":
            return parchg[:, :, :, 0, band, kpoint]
        if selection == "up":
            return parchg[:, :, :, :, band, kpoint] @ np.array([0.5, 0.5])
        if selection == "down":
            return parchg[:, :, :, :, band, kpoint] @ np.array([0.5, -0.5])

        message = f"Spin '{selection}' not understood. Use 'up', 'down' or 'total'."
        raise exception.IncorrectUsage(message)

    @_base.data_access
    def bands(self):
        """Return the band array listing the contributing bands.

        [2,4,5] means that the 2nd, 4th, and 5th bands are contributing while
        [0] means that all bands are contributing.
        """

        return self._raw_data.bands[:]

    def _check_band_index(self, band):
        bands = self.bands()
        if band in bands:
            return np.where(bands == band)[0][0]
        elif 0 in bands:
            message = f"""The band index {band} is not available.
            The summed partial charge density is returned instead."""
            warnings.warn(message, UserWarning)
            return 0
        else:
            message = f"""Band {band} not found in the bands array.
            Make sure to set IBAND, EINT, and LSEPB correctly in the INCAR file."""
            raise exception.NoData(message)

    @_base.data_access
    def kpoints(self):
        """Return the k-points array listing the contributing k-points.

        [2,4,5] means that the 2nd, 4th, and 5th k-points are contributing with
        all weights = 1. [0] means that all k-points are contributing.
        """
        return self._raw_data.kpoints[:]

    def _check_kpoint_index(self, kpoint):
        kpoints = self.kpoints()
        if kpoint in kpoints:
            return np.where(kpoints == kpoint)[0][0]
        elif 0 in kpoints:
            message = f"""The k-point index {kpoint} is not available.
            The summed partial charge density is returned instead."""
            warnings.warn(message, UserWarning)
            return 0
        else:
            message = f"""K-point {kpoint} not found in the kpoints array.
            Make sure to set KPUSE and LSEPK correctly in the INCAR file."""
            raise exception.NoData(message)


def _raise_error_if_vacuum_too_small(vacuum_thickness, min_vacuum=5.0):
    """Raise an error if the vacuum region is too small."""

    if vacuum_thickness < min_vacuum:
        message = f"""The vacuum region in your cell is too small for STM simulations.
        The minimum vacuum thickness for STM simulations is {min_vacuum} Angstrom."""
        raise exception.IncorrectUsage(message)


def _raise_error_if_vacuum_not_along_z(structure):
    """Raise an error if the vacuum region is not along the z-direction."""
    frac_pos = _get_sanitized_fractional_positions(structure)
    delta_x = np.max(frac_pos[:, 0]) - np.min(frac_pos[:, 0])
    delta_y = np.max(frac_pos[:, 1]) - np.min(frac_pos[:, 1])
    delta_z = np.max(frac_pos[:, 2]) - np.min(frac_pos[:, 2])

    if delta_z > delta_x or delta_z > delta_y:
        message = """The vacuum region in your cell is not located along
        the third lattice vector.
        STM simulations for such cells are not implemented.
        Please make sure that your vacuum is along the z-direction
        and the surface you want to sample is facing 'upwards'."""
        raise exception.NotImplemented(message)


def _get_sanitized_fractional_positions(structure):
    """Return the fractional positions of the atoms in the structure."""
    frac_pos = structure.positions()
    # Make sure that all fractional positions are between 0 and 1.
    # Otherwise, add 1 or subtract 1 to get the correct fractional position.
    while np.any(frac_pos < 0.0) or np.any(frac_pos >= 1.0):
        frac_pos = np.where(frac_pos < 0.0, frac_pos + 1.0, frac_pos)
        frac_pos = np.where(frac_pos >= 1.0, frac_pos - 1.0, frac_pos)
    return frac_pos


def _get_sanitized_cartesian_positions(structure):
    """Return the cartesian positions of the atoms in the structure."""
    frac_pos = _get_sanitized_fractional_positions(structure)
    return np.dot(frac_pos, structure.lattice_vectors())


def _min_of_z_charge(charge, sigma=4, truncate=3.0):
    """Returns the z-coordinate of the minimum of the charge density in the z-direction"""
    # average over the x and y axis
    z_charge = np.mean(charge, axis=(0, 1))
    # smooth the data using a gaussian filter
    z_charge = ndimage.gaussian_filter1d(
        z_charge, sigma=sigma, truncate=truncate, mode="wrap"
    )
    # return the z-coordinate of the minimum
    return np.argmin(z_charge)

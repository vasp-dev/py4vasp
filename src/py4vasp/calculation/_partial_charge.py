import dataclasses
import warnings
from typing import Union

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from py4vasp._third_party.graph import Graph
from py4vasp._third_party.graph.contour import Contour
from py4vasp.calculation import _base, _structure
from py4vasp.exception import IncorrectUsage, NoData, NotImplemented

_STM_MODES = {
    "constant_height": ["constant_height", "ch", "height"],
    "constant_current": ["constant_current", "cc", "current"]
}


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

    def to_dict(self, squeeze=True):
        """Store the partial charges in a dictionary.

        Returns
        -------
        dict
            The dictionary contains the partial charges as well as the structural
            information for reference.
        """

        return {
            **self._read_structure(),
            **self._read_grid(),
            **self._read_bands(),
            **self._read_kpoints(),
            **self._read_partial_charge(squeeze=squeeze),
        }

    def to_stm(
        self,
        selection: str = "constant_height",
        tip_height: float = 2.0,
        current: float = 1.0,
        supercell: Union[int, np.array] = 2,
        spin="both",
    ) -> Graph:
        """Generate stm image data from the partial charge density.

        Parameters
        ----------
        selection : str
            The mode in which the stm is operated. The default is "constant_height".
            The other option is "constant_current".
        tip_height : float
            The height of the stm tip above the surface in Angstrom.
            The default is 2.0 Angstrom. Only used in "constant_height" mode.
        current : float
            The tunneling current in nA. The default is 1.
            Only used in "constant_current" mode.
        supercell : int | np.array
            The supercell to be used for plotting the STM. The default is 2.
        spin : str
            The spin channel to be used. The default is "both".
            The other options are "up" and "down".

        Returns
        -------
        Graph
            The STM image as a graph object. The title is the
            label of the Contour object.
        """

        if isinstance(supercell, int):
            supercell = np.asarray([supercell, supercell])
        elif len(supercell) == 2:
            supercell = np.asarray(supercell)
        else:
            message = """The supercell has to be a single number or a 2D array.
            The supercell is used to multiply the x and y directions of the lattice."""
            raise IncorrectUsage(message)

        self._check_z_orth()
        if selection.lower() in _STM_MODES["constant_height"]:
            self._check_tip_height(tip_height)

        self.smoothed_charge = self._get_stm_data(spin)

        if selection.lower() in _STM_MODES["constant_height"]:
            stm = self._constant_height_stm(tip_height, spin)
        elif selection.lower() in _STM_MODES["constant_current"]:
            current = current * 1e-09  # convert nA to A
            stm = self._constant_current_stm(current, spin)
        else:
            raise IncorrectUsage(
                f"STM mode '{selection}' not understood. Use 'constant_height' or 'constant_current'."
            )
        stm.supercell = supercell
        return Graph(
            series=stm,
            title=stm.label,
        )

    def _constant_current_stm(self, current, spin):
        z_start = min_of_z_charge(
            self._get_stm_data(spin),
            sigma=self.STM_settings.sigma_z,
            truncate=self.STM_settings.truncate,
        )
        grid = self.grid()
        cc_scan = np.zeros((grid[0], grid[1]))
        # scan over the x and y grid
        for i in range(grid[0]):
            for j in range(grid[1]):
                # for more accuracy, interpolate each z-line of data with cubic splines
                spl = CubicSpline(range(grid[2]), self.smoothed_charge[i][j])

                for k in np.arange(
                    z_start, 0, -1 / self.STM_settings.interpolation_factor
                ):
                    if spl(k) >= current:
                        break
                cc_scan[i][j] = k
        # normalize the data
        # cc_scan = cc_scan - np.min(cc_scan.flatten())
        # return the tip height over the surface
        cc_scan = (
            cc_scan / self.STM_settings.interpolation_factor
            - self._get_highest_z_coord()
        )
        spin_label = "both spin channels" if spin == "both" else f"spin {spin}"
        topology = self._topology()
        label = f"STM of {topology} for {spin_label} at constant current={current*1e9:.1e} nA"
        return Contour(
            data=cc_scan, lattice=self.lattice_vectors()[:2, :2], label=label
        )

    def _constant_height_stm(self, tip_height, spin):
        grid = self.grid()
        z_index = self._z_index_for_height(tip_height + self._get_highest_z_coord())
        ch_scan = np.zeros((grid[0], grid[1]))
        for i in range(grid[0]):
            for j in range(grid[1]):
                ch_scan[i][j] = (
                    self.smoothed_charge[i][j][z_index]
                    * self.STM_settings.enhancement_factor
                )
        spin_label = "both spin channels" if spin == "both" else f"spin {spin}"
        topology = self._topology()
        label = f"STM of {topology} for {spin_label} at constant height={float(tip_height):.2f} Angstrom"
        return Contour(
            data=ch_scan,
            lattice=self.lattice_vectors()[:2, :2],
            label=label,
        )

    def _z_index_for_height(self, tip_height):
        return int(tip_height / self.lattice_vectors()[2][2] * self.grid()[2])

    @_base.data_access
    def _get_highest_z_coord(self):
        return np.max(self._structure.cartesian_positions()[:, 2])

    @_base.data_access
    def _get_lowest_z_coord(self):
        return np.min(self._structure.cartesian_positions()[:, 2])

    @_base.data_access
    def _topology(self):
        return str(self._structure._topology())

    def _estimate_vacuum(self):
        slab_thickness = self._get_highest_z_coord() - self._get_lowest_z_coord()
        z_vector = self.lattice_vectors()[2, 2]
        return z_vector - slab_thickness

    def _check_tip_height(self, tip_height):
        if tip_height > self._estimate_vacuum() / 2:
            message = f"""The tip position at {tip_height:.2f} is above half of the
             estimated vacuum thickness {self._estimate_vacuum():.2f} Angstrom.
            You would be sampling the bottom of your slab, which is not supported."""
            raise IncorrectUsage(message)

    def _check_z_orth(self):
        lv = self.lattice_vectors()
        if lv[0][2] != 0 or lv[1][2] != 0 or lv[2][0] != 0 or lv[2][1] != 0:
            message = """The third lattice vector is not in cartesian z-direction.
            or the first two lattice vectors are not in the xy-plane.
            STM simulations for such cells are not implemented."""
            raise NotImplemented(message)

    def _get_stm_data(self, spin):
        if 0 not in self.bands() or 0 not in self.kpoints():
            massage = """Simulated STM images are only supported for non-separated bands and k-points.
            Please set LSEPK and LSEPB to .FALSE. in the INCAR file."""
            raise NotImplemented(massage)
        chg = self._correct_units(self.to_array(band=0, kpoint=0, spin=spin))
        return self._smooth_stm_data(chg)

    @_base.data_access
    def _correct_units(self, charge_data):
        grid_volume = np.prod(self.grid())
        cell_volume = self._structure.volume()
        return charge_data / (grid_volume * cell_volume)

    def _smooth_stm_data(self, data):
        smoothed_charge = gaussian_filter(
            data,
            sigma=(
                self.STM_settings.sigma_xy,
                self.STM_settings.sigma_xy,
                self.STM_settings.sigma_z,
            ),
            truncate=self.STM_settings.truncate,
            mode="wrap",
        )
        return smoothed_charge

    @_base.data_access
    def lattice_vectors(self):
        """Return the lattice vectors of the input structure."""
        return self._structure._lattice_vectors()

    def _spin_polarized(self):
        return self._raw_data.partial_charge.shape[2] == 2

    def _read_grid(self):
        return {"grid": self.grid()}

    def _read_bands(self):
        return {"bands": self.bands()}

    def _read_kpoints(self):
        return {"kpoints": self.kpoints()}

    @_base.data_access
    def _read_structure(self):
        return {"structure": self._structure.read()}

    @_base.data_access
    def _read_partial_charge(self, squeeze=True):
        if squeeze:
            return {"partial_charge": np.squeeze(self._raw_data.partial_charge[:].T)}
        else:
            return {"partial_charge": self._raw_data.partial_charge[:].T}

    @_base.data_access
    def to_array(self, band=0, kpoint=0, spin="both"):
        """Return the partial charge density as a 3D array.

        Parameters
        ----------
        band : int
            The band index. The default is 0, which means that all bands are summed.
        kpoint : int
            The k-point index. The default is 0, which means that all k-points are summed.
        spin : str
            The spin channel to be used. The default is "both".
            The other options are "up" and "down".

        Returns
        -------
        np.array
            The partial charge density as a 3D array.
        """

        parchg = self._raw_data.partial_charge[:].T

        band = self._check_band_index(band)
        kpoint = self._check_kpoint_index(kpoint)

        if self._spin_polarized():
            if spin == "both":
                parchg = parchg[:, :, :, 0, band, kpoint]
            elif spin == "up":
                parchg = (
                    parchg[:, :, :, 0, band, kpoint] + parchg[:, :, :, 1, band, kpoint]
                ) / 2
            elif spin == "down":
                parchg = (
                    parchg[:, :, :, 0, band, kpoint] - parchg[:, :, :, 1, band, kpoint]
                ) / 2
            else:
                message = (
                    f"""Spin '{spin}' not understood. Use 'up', 'down' or 'both'."""
                )
                raise IncorrectUsage(message)
        else:
            parchg = parchg[:, :, :, 0, band, kpoint]

        return parchg

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
            raise NoData(message)

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
            raise NoData(message)


def min_of_z_charge(
    charge,
    sigma=4,
    truncate=3.0,
):
    """Returns the z-coordinate of the minimum of the charge density in the z-direction"""
    # average over the x and y axis
    z_charge = np.mean(charge, axis=(0, 1))
    # smooth the data using a gaussian filter
    z_charge = gaussian_filter1d(z_charge, sigma=sigma, truncate=truncate, mode="wrap")
    # return the z-coordinate of the minimum
    return np.argmin(z_charge)

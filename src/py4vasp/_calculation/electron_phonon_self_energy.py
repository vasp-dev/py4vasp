# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from collections import abc

from py4vasp import exception
from py4vasp._calculation import base
from py4vasp._calculation.electron_phonon_accumulator import ElectronPhononAccumulator
from py4vasp._calculation.electron_phonon_instance import ElectronPhononInstance
from py4vasp._util import convert


class ElectronPhononSelfEnergyInstance(ElectronPhononInstance):
    """
    Represents a single instance of electron-phonon self-energy calculations.
    This class provides access to the electron-phonon self-energy data for a specific
    self-energy accumulator. It allows retrieval of various components of the
    self-energy, such as Debye-Waller and Fan terms.

    Examples
    --------
    >>> instance = ElectronPhononSelfEnergyInstance(parent, index=0)
    >>> print(instance)
    electron phonon self energy 0
    >>> data = instance.to_dict()
    >>> fan_value = instance.get_fan((iband, ikpt, isp))
    """

    def __str__(self):
        """
        Return a string representation of the self energy instance.

        Returns
        -------
        str
            String representation including chemical potential and number of bands
            included in the sum.
        """
        return "\n".join(self._generate_lines())

    def _generate_lines(self):
        yield f"Electron self-energy instance {self.index + 1}:"
        indent = 4 * " "
        # Information about the chemical potential
        mu_tag, mu_val = self.parent.chemical_potential_mu_tag()
        yield f"{indent}{mu_tag}: {mu_val[self._get_data('id_index')[2] - 1]}"
        # Information about the number of bands summed over
        nbands_sum = self._get_data("nbands_sum")
        yield f"{indent}nbands_sum: {nbands_sum}"
        # Information about the broadening parameter
        delta = self._get_data("delta")
        yield f"{indent}selfen_delta: {delta}"
        # Information about the scattering approximation
        scattering_approx = self._get_data("scattering_approximation")
        yield f"{indent}scattering_approx: {scattering_approx}"

    def to_dict(self):
        """
        Convert the electron-phonon self-energy instance to a dictionary.

        Returns
        -------
        dict
            Dictionary containing metadata, eigenvalues, fan terms, debye_waller terms,
            and energies for this self-energy instance.
        """
        return {
            "metadata": self._read_metadata(),
            "eigenvalues": self.parent.eigenvalues(),
            "fan": self._get_fan(),
            "debye_waller": self._get_data("debye_waller"),
            "energies": self._get_data("energies"),
        }

    def _get_fan(self):
        return convert.to_complex(self._get_data("fan"))

    def fan(self):
        """
        Return the Fan contribution to the electron-phonon self-energy as a sparse tensor.

        Returns
        -------
        SparseTensor
            Sparse tensor containing Fan self-energy contributions for specific
            band, k-point, and spin combinations.
        """
        return self._make_sparse_tensor(self._get_fan())

    def debye_waller(self):
        """
        Return the Debye-Waller contribution to the electron-phonon self-energy as a sparse tensor.

        Returns
        -------
        SparseTensor
            Sparse tensor containing Debye-Waller self-energy contributions for specific
            band, k-point, and spin combinations.
        """
        return self._make_sparse_tensor(self._get_data("debye_waller"))

    def self_energy(self):
        """
        Return the total electron-phonon self-energy as a sparse tensor.

        Returns
        -------
        SparseTensor
            Sparse tensor containing the sum of Fan and Debye-Waller contributions
            for specific band, k-point, and spin combinations.
        """
        self_energy = self._get_fan() + self._get_data("debye_waller")
        return self._make_sparse_tensor(self_energy)

    def energies(self):
        """
        Return the energies at which the self energy was evaluated as a sparse tensor.

        Returns
        -------
        SparseTensor
            Sparse tensor containing energy values for specific band, k-point,
            and spin combinations.
        """
        return self._make_sparse_tensor(self._get_data("energies"))

    def _make_sparse_tensor(self, tensor_data):
        band_kpoint_spin_index = self._get_data("band_kpoint_spin_index").T - 1
        band_start = self._get_data("band_start")
        return SparseTensor(band_kpoint_spin_index, band_start, tensor_data)


class ElectronPhononSelfEnergy(base.Refinery, abc.Sequence):
    """Access and analyze electron-phonon self-energy data.

    This class provides methods to access, select, and analyze the electron-phonon
    self-energy. It allows you to retrieve various quantities
    such as eigenvalues, Debye-Waller and Fan self-energies, and scattering
    approximations for different selections of bands, k-points, and spin channel.

    Main features:
        - Retrieve self-energy data for specific bands, k-points and spin channels.
        - Convert self-energy data to dictionaries for further analysis.
        - Iterate over all available self-energy instances.

    Examples
    --------

        >>> elph_selfen = ElectronPhononSelfEnergy(raw_data)
        >>> print(elph_selfen)
        electron phonon self energy
        >>> instance = elph_selfen[0]
        >>> data = instance.to_dict()
    """

    def _accumulator(self):
        return ElectronPhononAccumulator(self, self._raw_data)

    @base.data_access
    def __str__(self):
        return str(self._accumulator())

    @base.data_access
    def to_dict(self):
        return self._accumulator().to_dict()

    @base.data_access
    def selections(self):
        """Return a dictionary describing what options are available to read the electron self-energies.

        Returns
        -------
        dict
            Dictionary containing available selection options with their possible values.
            Keys include selection criteria like "nbands_sum", "selfen_approx", "selfen_delta".
        """
        base_selections = super().selections()
        return self._accumulator().selections(base_selections)

    @base.data_access
    def chemical_potential_mu_tag(self):
        """Return the chemical potential tag and values.

        Returns
        -------
        tuple
            A tuple containing (tag_name, values_array) where tag_name is a string
            describing the chemical potential parameter and values_array contains
            the numerical values.
        """
        return self._accumulator().chemical_potential_mu_tag()

    @base.data_access
    def select(self, selection):
        """Return a list of ElectronPhononSelfEnergyInstance objects matching the selection.

        Parameters
        ----------
        selection : str
            String representing the selection criteria (e.g., "nbands_sum", "selfen_approx",
            "selfen_delta") and the desired values for those properties

        Returns
        -------
        list of ElectronPhononSelfEnergyInstance
            Instances that match the selection criteria.
        """
        indices = self._accumulator().select_indices(selection)
        return [ElectronPhononSelfEnergyInstance(self, index) for index in indices]

    @base.data_access
    def _get_data(self, name, index):
        return self._accumulator().get_data(name, index)

    @base.data_access
    def eigenvalues(self):
        """Return the eigenvalues from the raw data.

        Returns
        -------
        numpy.ndarray
            Array containing eigenvalues for all k-points, bands, and spin channels.
        """
        return self._raw_data.eigenvalues[:]

    @base.data_access
    def __getitem__(self, key):
        if 0 <= key < len(self._raw_data.valid_indices):
            return ElectronPhononSelfEnergyInstance(self, key)
        raise IndexError("Index out of range for electron phonon self energy instance.")

    @base.data_access
    def __len__(self):
        return len(self._raw_data.valid_indices)


class SparseTensor:
    """
    A sparse tensor implementation for electron-phonon data.

    This class provides efficient storage and access to tensor data that is only
    defined for specific combinations of band, k-point, and spin indices.

    Parameters
    ----------
    band_kpoint_spin_index : array_like
        Index mapping for (band, kpoint, spin) combinations.
    band_start : int
        Starting band index for valid data.
    tensor : array_like
        The actual tensor data values in compressed form.
    """

    def __init__(self, band_kpoint_spin_index, band_start, tensor):
        self._band_kpoint_spin_index = band_kpoint_spin_index
        self._band_start = band_start
        self._tensor = tensor

    def _get_band_kpoint_spin_index(self, spin, kpoint, band):
        if 0 <= band < self._band_start:
            raise exception.IncorrectUsage(
                f"Band index {band} is not in valid range {self._band_range_string()}."
            )
        try:
            if band > 0:
                band -= self._band_start
            return self._band_kpoint_spin_index[band, kpoint, spin]
        except IndexError:
            raise exception.IncorrectUsage(
                f"Invalid indices: {spin=}, {kpoint=}, {band=}. "
                f"Valid ranges are: 0 <= spin < {self._band_kpoint_spin_index.shape[2]}."
                f", 0 <= kpoint < {self._band_kpoint_spin_index.shape[1]}, "
                f", {self._band_range_string()}."
            )

    def _band_range_string(self):
        range_ = self.valid_bands
        return f"{range_.start} <= band < {range_.stop}"

    @property
    def valid_bands(self):
        """
        Return the range of valid band indices.

        Returns
        -------
        range
            Range object representing the valid band indices.
        """
        return range(
            self._band_start, self._band_start + self._band_kpoint_spin_index.shape[0]
        )

    def __getitem__(self, spin_kpoint_band_tuple):
        """
        Access tensor data for specific spin, k-point, and band indices.

        Parameters
        ----------
        spin, kpoint, band : int
            Spin, k-point, and band indices.

        Returns
        -------
        array_like
            The tensor data at the specified indices.

        Raises
        ------
        IncorrectUsage
            If the tuple doesn't contain exactly three indices, if band index
            is outside valid range, or if indices are invalid.
        DataMismatch
            If the calculation for the specified indices was not performed.
        """
        if len(spin_kpoint_band_tuple) != 3:
            raise exception.IncorrectUsage(
                "Please provide exactly three indices for spin, kpoint and band."
            )
        spin, kpoint, band = spin_kpoint_band_tuple
        index_ = self._get_band_kpoint_spin_index(spin, kpoint, band)
        if index_ < 0:
            raise exception.DataMismatch(
                f"The calculation for {band=} {kpoint=} {spin=} was not performed."
            )
        return self._tensor[index_]

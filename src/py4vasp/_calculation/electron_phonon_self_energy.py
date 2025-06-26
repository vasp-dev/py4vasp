# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import exception
from py4vasp._calculation import base, slice_
from py4vasp._calculation.electron_phonon_chemical_potential import (
    ElectronPhononChemicalPotential,
)
from py4vasp._util import select, suggest

ALIAS = {
    "selfen_delta": "delta",
    "scattering_approx": "scattering_approximation",
}


class ElectronPhononSelfEnergyInstance:
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

    def __init__(self, parent, index):
        self.parent = parent
        self.index = index

    def __str__(self):
        """
        Returns a string representation of the self energy instance, including chemical
        potential and number of bands included in the sum.
        """
        lines = []
        lines.append(f"Electron self-energy accumulator N =  {self.index + 1}")
        lines.append("----------------------------------")
        # Information about the chemical potential
        mu_tag, mu_val = self.get_mu_tag()
        lines.append(f"{mu_tag}: {mu_val}")
        # Information about the scattering approximation
        scattering_approx = self._get_data("scattering_approximation")
        lines.append(f"scattering_approximation: {scattering_approx}")
        # Information about the broadening parameter
        delta = self._get_scalar("delta")
        lines.append(f"delta: {delta}")
        # Information about the number of bands summed over
        nbands_sum = self._get_scalar("nbands_sum")
        lines.append(f"nbands_sum: {nbands_sum}")
        return "\n".join(lines) + "\n"

    def get_mu_tag(self):
        """Get choosen tag to select the chemical potential as well as its value"""
        mu_tag, mu_val = self.parent.chemical_potential_mu_tag()
        mu_idx = self.id_index[2] - 1
        return mu_tag, mu_val[mu_idx]

    def _get_data(self, name):
        return self.parent._get_data(name, self.index)

    def _get_scalar(self, name):
        return self.parent._get_scalar(name, self.index)

    def read(self):
        "Convenient wrapper around to_dict. Check that function for examples and optional arguments."
        return self.to_dict()

    def to_dict(self):
        mu_tag, mu_val = self.parent.chemical_potential_mu_tag()
        return {
            "metadata": {
                "nbands_sum": self._get_scalar("nbands_sum"),
                "selfen_delta": self._get_scalar("delta"),
                "scattering_approx": self._get_data("scattering_approximation"),
                mu_tag: mu_val[self._get_data("id_index")[2] - 1],
            },
            "eigenvalues": self.parent.eigenvalues(),
            "fan": self._get_data("fan"),
            "debye_waller": self._get_data("debye_waller"),
        }

    @property
    def id_index(self):
        return self._get_data("id_index")

    @property
    def id_name(self):
        return self.parent.id_name

    def get_fan(self, arg):
        iband, ikpt, isp, *more = arg
        st = SparseTensor(
            self._get_data("band_kpoint_spin_index") - 1,
            self._get_scalar("band_start"),
            self._get_data("fan"),
        )
        return st[arg]

    def get_debye_waller(self, arg):
        iband, ikpt, isp, *more = arg
        st = SparseTensor(
            self._get_data("band_kpoint_spin_index") - 1,
            self._get_scalar("band_start"),
            self._get_data("debye_waller"),
        )
        return st[arg]

    def get_self_energy(self, arg):
        import numpy as np

        iband, ikpt, isp, *more = arg
        fan = self._get_data("fan")
        dw = self._get_data("debye_waller")[:, np.newaxis, :]
        st = SparseTensor(
            self._get_data("band_kpoint_spin_index") - 1,
            self._get_scalar("band_start"),
            fan + dw,
        )
        return st[arg]


class ElectronPhononSelfEnergy(base.Refinery):
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

    @base.data_access
    def __str__(self):
        return "electron phonon self energy"

    @base.data_access
    def to_dict(self):
        return {"naccumulators": len(self._raw_data.valid_indices)}

    @base.data_access
    def eigenvalues(self):
        return self._raw_data.eigenvalues[:]

    @base.data_access
    def id_name(self):
        return self._raw_data.id_name[:]

    @base.data_access
    def id_size(self):
        return self._raw_data.id_size[:]

    @base.data_access
    def __getitem__(self, key):
        # TODO add logic to select instances
        return ElectronPhononSelfEnergyInstance(self, key)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @base.data_access
    def valid_delta(self):
        return {self._get_scalar("delta", index) for index in range(len(self))}

    @base.data_access
    def valid_nbands_sum(self):
        return {self._get_scalar("nbands_sum", index) for index in range(len(self))}

    @base.data_access
    def valid_scattering_approximation(self):
        return {
            self._get_data("scattering_approximation", index)
            for index in range(len(self))
        }

    @base.data_access
    def selections(self):
        """Return a dictionary describing what options are available
        to read the electron self-energies."""
        mu_tag, mu_val = self.chemical_potential_mu_tag()
        return {
            **super().selections(),
            mu_tag: np.unique(mu_val),
            "nbands_sum": np.unique(self._raw_data.nbands_sum),
            "selfen_delta": np.unique(self._raw_data.delta),
            "scattering_approx": np.unique(self._raw_data.scattering_approximation),
        }

    def _generate_selections(self, selection):
        tree = select.Tree.from_selection(selection)
        for selection in tree.selections():
            yield selection

    @base.data_access
    def chemical_potential_mu_tag(self):
        chemical_potential = ElectronPhononChemicalPotential.from_data(
            self._raw_data.chemical_potential
        )
        return chemical_potential.mu_tag()

    @base.data_access
    def select(self, selection):
        """Return a list of ElectronPhononSelfEnergyInstance objects matching the selection.

        Parameters
        ----------
        selection : dict
            Dictionary with keys as selection names (e.g., "nbands_sum", "selfen_approx", "selfen_delta")
            and values as the desired values for those properties.

        Returns
        -------
        list of ElectronPhononSelfEnergyInstance
            Instances that match the selection criteria.
        """
        indices = self._select_indices(selection)
        return [ElectronPhononSelfEnergyInstance(self, index) for index in indices]

    def _select_indices(self, selection):
        tree = select.Tree.from_selection(selection)
        return {
            index_
            for selection in tree.selections()
            for index_ in self._filter_indices(selection)
        }

    def _filter_indices(self, selection):
        remaining_indices = range(len(self._raw_data.valid_indices))
        for group in selection:
            self._raise_error_if_group_format_incorrect(group)
            assert len(group.group) == 2
            remaining_indices = self._filter_group(remaining_indices, *group.group)
            remaining_indices = list(remaining_indices)
        yield from remaining_indices

    def _raise_error_if_group_format_incorrect(self, group):
        if not isinstance(group, select.Group) or group.separator != "=":
            message = f'\
The selection {group} is not formatted correctly. It should be formatted like \
"key=value". Please check the "selections" method for available options.'
            raise exception.IncorrectUsage(message)

    def _filter_group(self, remaining_indices, key, value):
        for index_ in remaining_indices:
            if self._match_key_value(index_, key, str(value)):
                yield index_

    def _match_key_value(self, index_, key, value):
        instance_value = self._get_data(key, index_)
        try:
            value = float(value)
        except ValueError:
            return instance_value == value
        return np.isclose(instance_value, float(value), rtol=1e-8, atol=0)

    @base.data_access
    def _get_data(self, name, index):
        name = ALIAS.get(name, name)
        dataset = getattr(self._raw_data, name, None)
        if dataset is not None:
            return np.array(dataset[index])
        mu_tag, mu_val = self.chemical_potential_mu_tag()
        self._raise_error_if_not_present(name, expected_name=mu_tag)
        return mu_val[self._raw_data.id_index[index, 2] - 1]

    def _raise_error_if_not_present(self, name, expected_name):
        if name != expected_name:
            valid_names = set(self.selections().keys())
            valid_names.remove("electron_phonon_self_energy")
            did_you_mean = suggest.did_you_mean(name, valid_names)
            available_selections = '", "'.join(valid_names)
            message = f'\
The selection "{name}" is not a valid choice. {did_you_mean}Please check the \
available selections: "{available_selections}".'
            raise exception.IncorrectUsage(message)

    @base.data_access
    def _get_scalar(self, name, index):
        return getattr(self._raw_data, name)[index][()]

    @base.data_access
    def __len__(self):
        return len(self._raw_data.valid_indices)


class SparseTensor:
    def __init__(self, band_kpoint_spin_index, band_start, tensor):
        self.band_kpoint_spin_index = band_kpoint_spin_index
        self.band_start = band_start
        self.tensor = tensor

    def _get_band_kpoint_spin_index(self, iband, ikpt, isp):
        return self.band_kpoint_spin_index[iband - self.band_start, ikpt, isp]

    def __getitem__(self, arg):
        iband, ikpt, isp, *more = arg
        ibks = self._get_band_kpoint_spin_index(iband, ikpt, isp)
        if ibks == -1:
            raise IndexError(
                f"The calculation for band:{iband} kpoint:{ikpt} spin:{isp} was not performed."
            )
        return self.tensor[ibks]

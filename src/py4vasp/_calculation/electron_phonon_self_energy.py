# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._calculation.electron_phonon_chemical_potential import (
    ElectronPhononChemicalPotential,
)
from py4vasp._calculation import base, slice_
from py4vasp._util import select


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
        return "electron phonon self energy %d" % self.index

    def _get_data(self, name):
        return self.parent._get_data(name, self.index)

    def _get_scalar(self, name):
        return self.parent._get_scalar(name, self.index)

    def to_dict(self, selection=None):
        names = [
            "debye_waller",
            "fan",
            "scattering_approximation",
        ]
        dict_ = {name: self._get_data(name) for name in names}
        dict_["eigenvalues"] = self.parent.eigenvalues()
        dict_["nbands_sum"] = self._get_scalar("nbands_sum")
        dict_["delta"] = self._get_scalar("delta")
        return dict_

    @property
    def id_index(self):
        return self._get_data("id_index")

    @property
    def id_name(self):
        return self.parent.id_name

    def get_fan(self, arg):
        iband, ikpt, isp, *more = arg
        st = SparseTensor(
            self._get_data("band_kpoint_spin_index"),
            0,
            self._get_data("fan"),
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
        id_name = self.id_name()
        id_size = self.id_size()
        selections_dict = {}
        selections_dict["nbands_sum"] = self.valid_nbands_sum()
        selections_dict["selfen_approx"] = self.valid_scattering_approximation()
        selections_dict["selfen_delta"] = self.valid_delta()
        chemical_potential = ElectronPhononChemicalPotential.from_data(
            self._raw_data.chemical_potential
        )
        mu_tag, mu_val = chemical_potential.mu_tag()
        selections_dict[mu_tag] = mu_val
        return selections_dict
    
    def _generate_selections(self, selection):
        tree = select.Tree.from_selection(selection)
        for selection in tree.selections():
            yield selection

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
        selected_instances = []
        mu_tag, mu_val = self.chemical_potential_mu_tag()
        for idx in range(len(self)):
            match_all = False
            for sel in self._generate_selections(selection):
                match = True
                sel_dict = dict(zip(sel[::2], sel[1::2]))
                for key, value in sel_dict.items():
                    # Map selection keys to property names
                    if key == "nbands_sum":
                        instance_value = self._get_scalar("nbands_sum", idx)
                        match_this = instance_value == value
                    elif key == "selfen_approx":
                        instance_value = self._get_data("scattering_approximation", idx)
                        match_this = instance_value == value
                    elif key == "selfen_delta":
                        instance_value = self._get_scalar("delta", idx)
                        match_this = abs(instance_value-value)<1e-8
                    elif key == mu_tag:
                        mu_idx = self[idx].id_index[2]-1
                        instance_value = mu_val[mu_idx]
                        match_this = abs(instance_value-float(value))<1e-8
                    else:
                        possible_values = self.selections()
                        raise ValueError(f"Invalid selection {key}. Possible values are {possible_values.keys()}")
                    match = match and match_this
                match_all = match_all or match
            if match_all:
                selected_instances.append(ElectronPhononSelfEnergyInstance(self, idx))
        return selected_instances

    @base.data_access
    def _get_data(self, name, index):
        return getattr(self._raw_data, name)[index][:]

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
        return self.band_kpoint_spin_index[iband - self.band_start][ikpt, isp]

    def __getitem__(self, arg):
        iband, ikpt, isp, *more = arg
        ibks = self._get_band_kpoint_spin_index(iband, ikpt, isp)
        return self.tensor[ibks]

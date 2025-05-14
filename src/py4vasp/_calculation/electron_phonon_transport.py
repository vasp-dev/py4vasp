# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._calculation import (
    base,
    electron_phonon_chemical_potential,
    electron_phonon_self_energy,
    slice_,
)
from py4vasp._util import convert, import_, select

pd = import_.optional("pandas")


class ElectronPhononTransportInstance:
    def __init__(self, parent, index):
        self.parent = parent
        self.index = index

    def __str__(self):
        return "electron phonon transport %d" % self.index

    def get_data(self, name):
        return self.parent.read_data(name, self.index)

    def to_dict(self, selection=None):
        return {
            "temperatures": self.get_data("temperatures"),
            "transport_function": self.get_data("transport_function"),
            "electronic_conductivity": self.get_data("electronic_conductivity"),
            "mobility": self.get_data("mobility"),
            "seebeck": self.get_data("seebeck"),
            "peltier": self.get_data("peltier"),
            "electronic_thermal_conductivity": self.get_data(
                "electronic_thermal_conductivity"
            ),
            "scattering_approximation": self.scattering_approximation,
        }

    @property
    def id_index(self):
        return self.get_data("id_index")

    @property
    def id_name(self):
        return self.parent.id_name

    @property
    def nbands_sum(self):
        return

    @property
    def delta(self):
        return self.get_data("delta")

    @property
    def scattering_approximation(self):
        return self.get_data("scattering_approximation")


class ElectronPhononTransport(
    base.Refinery,
    electron_phonon_self_energy.Mixin,
    electron_phonon_chemical_potential.Mixin,
):
    "Placeholder for electron phonon transport"

    @base.data_access
    def __str__(self):
        return "electron phonon transport"

    @base.data_access
    def to_dict(self, selection=None):
        return {
            "naccumulators": len(self._raw_data.valid_indices),
        }

    @property
    @base.data_access
    def id_name(self):
        return self._raw_data.id_name[:]

    @property
    @base.data_access
    def id_size(self):
        return self._raw_data.id_size[:]

    @base.data_access
    def __getitem__(self, key):
        # TODO add logic to select instances
        return ElectronPhononTransportInstance(self, key)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @base.data_access
    def read_data(self, name, index):
        return getattr(self._raw_data, name)[index][:]

    @base.data_access
    def __len__(self):
        return len(self._raw_data.valid_indices)

    @base.data_access
    def selections(self):
        """Return a dictionary describing what options are available
        to read the transport coefficients."""
        id_name = self.id_name
        id_size = self.id_size
        selections_dict = {}
        selections_dict["nbands_sum"] = (
            self._electron_phonon_self_energy.valid_nbands_sum()
        )
        selections_dict["selfen_approx"] = (
            self._electron_phonon_self_energy.valid_scattering_approximation()
        )
        selections_dict["selfen_delta"] = (
            self._electron_phonon_self_energy.valid_delta()
        )
        mu_tag,mu_val = self._electron_phonon_chemical_potential.mu_tag()
        selections_dict[mu_tag] = mu_val
        return selections_dict

    def select(self, selection):
        parsed_selections = self._parse_selection(selection)
        selected_instances = []
        for elph_selfen_instance in self:
            # loop over selections
            for parsed_selection in parsed_selections:
                print(parsed_selection)
            continue
            selected_instances.append(elph_selfen_instance)
        return selected_instances

    def _parse_selection(self, selection):
        tree = select.Tree.from_selection(selection)
        return list(tree.selections())
        # for selection in tree.selections():
        #    return selection

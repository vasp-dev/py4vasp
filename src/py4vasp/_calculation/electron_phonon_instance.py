# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import abc


class ElectronPhononInstance(abc.ABC):

    def __init__(self, parent, index):
        self.parent = parent
        self.index = index

    def _get_data(self, name):
        return self.parent._get_data(name, self.index)

    @abc.abstractmethod
    def __str__(self):
        pass

    def print(self):
        "Print a string representation of this instance."
        print(str(self))

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    @abc.abstractmethod
    def to_dict(self):
        pass

    def _read_metadata(self):
        mu_tag, mu_val = self.parent.chemical_potential_mu_tag()
        return {
            mu_tag: mu_val[self._get_data("id_index")[2] - 1],
            "nbands_sum": self._get_data("nbands_sum"),
            "selfen_delta": self._get_data("delta"),
            "scattering_approx": self._get_data("scattering_approximation"),
        }

    def _metadata_string(self):
        metadata = self._read_metadata()
        return "\n".join([f"    {key}: {value}" for key, value in metadata.items()])

    def read(self):
        "Convenient wrapper around to_dict. Check that function for examples and optional arguments."
        return self.to_dict()

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import abc
from typing import Any, Dict


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

    def read_metadata(self) -> Dict[str, Any]:
        """Read metadata for this instance.

        The metadata contains information about the settings used for this instance,
        such as the number of bands included in the summation, the delta value used
        for the self-energy calculation, and the scattering approximation employed.

        Returns
        -------
        -
            A dictionary containing the metadata for this instance. The keys correspond to
            settings in the INCAR file, and the values are the respective settings used.
        """
        mu_tag, mu_val = self.parent.chemical_potential_mu_tag()
        optional_metadata = {}
        if nbands_sum := self._get_data("nbands_sum"):
            optional_metadata["nbands_sum"] = nbands_sum
        if selfen_delta := self._get_data("delta"):
            optional_metadata["selfen_delta"] = selfen_delta
        return {
            mu_tag: mu_val[self._get_data("id_index")[2] - 1],
            **optional_metadata,
            "scattering_approx": self._get_data("scattering_approximation"),
        }

    def _metadata_string(self):
        metadata = self.read_metadata()
        return "\n".join([f"    {key}: {value}" for key, value in metadata.items()])

    def read(self):
        "Convenient wrapper around to_dict. Check that function for examples and optional arguments."
        return self.to_dict()

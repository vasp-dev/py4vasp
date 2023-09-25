# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import exception
from py4vasp._data import base, structure
from py4vasp._util import import_

pretty = import_.optional("IPython.lib.pretty")


class Potential(base.Refinery, structure.Mixin):
    """The potential"""

    @staticmethod
    def _is_non_polarized(potential):
        return potential.shape[0] == 1

    @staticmethod
    def _is_collinear(potential):
        return potential.shape[0] == 2

    @staticmethod
    def _is_noncollinear(potential):
        return potential.shape[0] == 4

    @staticmethod
    def _does_this_potential_exist(raw_data, name):
        potential = getattr(raw_data, f"{name}_potential", None)
        return potential is not None

    def _create_potential_dict(self, name, raw_data):
        potential_dict = {}
        if not self._does_this_potential_exist(raw_data, name):
            return
        potential = getattr(raw_data, f"{name}_potential")
        if self._is_non_polarized(potential):
            potential_dict[name] = potential[0]
        elif self._is_collinear(potential):
            potential_dict[f"{name}_up"] = potential[0]
            potential_dict[f"{name}_down"] = potential[1]
            potential_dict[name] = np.mean(potential, axis=0)
        elif self._is_noncollinear(potential):
            potential_dict[name] = potential[0]
            potential_dict[f"{name}_magnetization"] = potential[1:]
        else:
            exception.RefinementError(
                """\
Currently only three readings of the potential are possible; from non-spin polarized
calculations, from collinear and non-collinear calculations. You are seeing this error
because the shapes of one of these arrays of potentials has changed."""
            )
        return potential_dict

    @base.data_access
    def to_dict(self):
        _raise_error_if_no_data(self._raw_data.total_potential)
        output = self._create_potential_dict("total", self._raw_data)
        for name in ["hartree", "ionic", "xc"]:
            potential_dict = self._create_potential_dict(name, self._raw_data)
            if potential_dict:
                output.update(potential_dict)
        return output


def _raise_error_if_no_data(data):
    if data.is_none():
        raise exception.NoData("Cannot find the total potential data.")

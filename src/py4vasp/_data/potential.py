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
    def _create_potential_dict(name, raw_data):
        potential_dict = {}
        potential = getattr(raw_data, f"{name}_potential")
        if not potential:
            return
        if potential.ndim == 3:  # non-spin polarized
            potential_dict[name] = potential
        elif potential.ndim == 4 and potential.shape[0] == 2:  # collinear
            potential_dict[f"{name}_up"] = potential[0]
            potential_dict[f"{name}_down"] = potential[1]
            potential_dict[name] = np.mean(potential, axis=0)
        elif potential.ndim == 4 and potential.shape[0] == 4:  # non-collinear
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
        hartree_potential_dict = self._create_potential_dict("hartree", self._raw_data)
        ionic_potential_dict = self._create_potential_dict("ionic", self._raw_data)
        xc_potential_dict = self._create_potential_dict("xc", self._raw_data)
        if hartree_potential_dict:
            output.update(hartree_potential_dict)
        if ionic_potential_dict:
            output.update(ionic_potential_dict)
        if xc_potential_dict:
            output.update(xc_potential_dict)
        return output


def _raise_error_if_no_data(data):
    if data.is_none():
        raise exception.NoData("Cannot find the total potential data.")

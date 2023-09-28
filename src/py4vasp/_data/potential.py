# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools

import numpy as np

from py4vasp import exception
from py4vasp._data import base, structure
from py4vasp._util import import_

pretty = import_.optional("IPython.lib.pretty")


class Potential(base.Refinery, structure.Mixin):
    """The potential"""

    def _read_potential(self, name):
        potential = getattr(self._raw_data, f"{name}_potential")
        if potential.is_none():
            return
        potential = np.moveaxis(potential, 0, -1).T
        yield name, potential[0]
        if _is_collinear(potential):
            yield f"{name}_up", potential[0] + potential[1]
            yield f"{name}_down", potential[0] - potential[1]
        elif _is_noncollinear(potential):
            yield f"{name}_magnetization", potential[1:]

    @base.data_access
    def to_dict(self):
        _raise_error_if_no_data(self._raw_data.total_potential)
        result = {"structure": self._structure.read()}
        potentials = [
            self._read_potential(potential)
            for potential in ["total", "hartree", "ionic", "xc"]
        ]
        result.update(itertools.chain(*potentials))
        return result

    @base.data_access
    def plot(self, selection="total", *, isolevel=0):
        self._raise_error_if_selection_incorrect(selection)
        potential = getattr(self._raw_data, f"{selection}_potential")
        _raise_error_if_no_data(potential, selection)
        viewer = self._structure.plot()
        options = {"isolevel": isolevel, "color": "yellow", "opacity": 0.6}
        viewer.show_isosurface(potential[0].T, **options)
        return viewer

    def _raise_error_if_selection_incorrect(self, selection):
        valid_potentials = ("total", "ionic", "xc", "hartree")
        if selection in valid_potentials:
            return
        message = f"""The selection {selection} is not a valid name for a potential. Only
        the following selections are allowed: {", ".join(valid_potentials)}. Please check
        for spelling errors"""
        raise exception.IncorrectUsage(message)


def _is_collinear(potential):
    return potential.shape[0] == 2


def _is_noncollinear(potential):
    return potential.shape[0] == 4


def _raise_error_if_no_data(data, selection="total"):
    if data.is_none():
        message = f"Cannot find the {selection} potential data."
        message += " Did you set LVTOT = T in the INCAR file?"
        if selection != "total":
            message += f" Did you set POTH5 = {selection} in the INCAR file?"
        raise exception.NoData(message)

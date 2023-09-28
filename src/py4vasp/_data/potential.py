# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools

import numpy as np

from py4vasp import exception
from py4vasp._data import base, structure
from py4vasp._util import import_, select

VALID_KINDS = ("total", "ionic", "xc", "hartree")


class Potential(base.Refinery, structure.Mixin):
    """The potential"""

    def _read_potential(self, kind):
        potential = getattr(self._raw_data, f"{kind}_potential")
        if potential.is_none():
            return
        potential = np.moveaxis(potential, 0, -1).T
        yield kind, potential[0]
        if _is_collinear(potential):
            yield f"{kind}_up", potential[0] + potential[1]
            yield f"{kind}_down", potential[0] - potential[1]
        elif _is_noncollinear(potential):
            yield f"{kind}_magnetization", potential[1:]

    @base.data_access
    def to_dict(self):
        _raise_error_if_no_data(self._raw_data.total_potential)
        result = {"structure": self._structure.read()}
        potentials = [self._read_potential(potential) for potential in VALID_KINDS]
        result.update(itertools.chain(*potentials))
        return result

    @base.data_access
    def plot(self, selection="total", *, isolevel=0):
        viewer = self._structure.plot()
        options = {"isolevel": isolevel, "color": "yellow", "opacity": 0.6}
        for kind, component in _parse_selection(selection):
            self._add_potential_isosurface(viewer, kind, component, options)
        return viewer

    def _add_potential_isosurface(self, viewer, kind, component, options):
        self._raise_error_if_kind_incorrect(kind)
        potential_data = getattr(self._raw_data, f"{kind}_potential")
        _raise_error_if_no_data(potential_data, kind)
        if component == "up":
            potential = potential_data[0] + potential_data[1]
        elif component == "down":
            potential = potential_data[0] - potential_data[1]
        else:
            potential = potential_data[0]
        viewer.show_isosurface(potential.T, **options)

    def _raise_error_if_kind_incorrect(self, kind):
        if kind in VALID_KINDS:
            return
        message = f"""The selection {kind} is not a valid name for a potential. Only
        the following selections are allowed: {", ".join(VALID_KINDS)}. Please check
        for spelling errors"""
        raise exception.IncorrectUsage(message)


def _parse_selection(selection):
    tree = select.Tree.from_selection(selection)
    possible_components = {"up", "down"}
    for selection in tree.selections():
        set_ = set(selection)
        components = set_.intersection(possible_components)
        component = components.pop() if components else None
        kinds = set_.difference(possible_components)
        kind = kinds.pop() if kinds else "total"
        yield kind, component


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

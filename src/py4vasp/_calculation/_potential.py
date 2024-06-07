# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools

import numpy as np

from py4vasp import _config, calculation, exception
from py4vasp._third_party import view
from py4vasp._util import select
from py4vasp.calculation import _base, _structure

VALID_KINDS = ("total", "ionic", "xc", "hartree")


class Potential(_base.Refinery, _structure.Mixin, view.Mixin):
    """The local potential describes the interactions between electrons and ions.

    In DFT calculations, the local potential consists of various contributions, each
    representing different aspects of the electron-electron and electron-ion
    interactions. The ionic potential arises from the attraction between electrons and
    the atomic nuclei. The Hartree potential accounts for the repulsion between
    electrons resulting from the electron density itself. Additionally, the
    exchange-correlation (xc) potential approximates the effects of electron exchange
    and correlation. The accuracy of this approximation directly influences the
    accuracy of the calculated properties.

    In VASP, the local potential is defined in real space on the FFT grid. You control
    which potentials are written with the :tag:`WRT_POTENTIAL` tag. This class provides
    the methods to read and visualize the potential. If you are interested in the
    average potential, you may also look at the :data:`~py4vasp.calculation.workfunction`.
    """

    @_base.data_access
    def __str__(self):
        potential = self._raw_data.total_potential
        if _is_collinear(potential):
            description = "collinear potential:"
        elif _is_noncollinear(potential):
            description = "noncollinear potential:"
        else:
            description = "nonpolarized potential:"
        topology = calculation.topology.from_data(self._raw_data.structure.topology)
        structure = f"structure: {topology}"
        grid = f"grid: {potential.shape[3]}, {potential.shape[2]}, {potential.shape[1]}"
        available = "available: " + ", ".join(
            kind for kind in VALID_KINDS if not self._get_potential(kind).is_none()
        )
        return "\n    ".join([description, structure, grid, available])

    @_base.data_access
    def to_dict(self):
        """Store all available contributions to the potential in a dictionary.

        Returns
        -------
        dict
            The dictionary contains the total potential as well as the potential
            differences between up and down for collinear or the directional potential
            for noncollinear calculations. If individual contributions to the potential
            are available, these are returned, too. Structural information is given for
            reference.
        """
        _raise_error_if_no_data(self._raw_data.total_potential)
        result = {"structure": self._structure.read()}
        items = [self._generate_items(kind) for kind in VALID_KINDS]
        result.update(itertools.chain(*items))
        return result

    def _generate_items(self, kind):
        potential = self._get_potential(kind)
        if potential.is_none():
            return
        potential = np.moveaxis(potential, 0, -1).T
        yield kind, potential[0]
        if _is_collinear(potential):
            yield f"{kind}_up", potential[0] + potential[1]
            yield f"{kind}_down", potential[0] - potential[1]
        elif _is_noncollinear(potential):
            yield f"{kind}_magnetization", potential[1:]

    @_base.data_access
    def to_view(self, selection="total", supercell=None, **user_options):
        """Plot an isosurface of a selected potential.

        Parameters
        ----------
        selection : str
            Select the kind of potential of which you want the isosurface.

        supercell : int or np.ndarray
            If present the data is replicated the specified number of times along each
            direction.

        user_options
            Further arguments with keyword that get directly passed on to the
            visualizer. Most importantly, you can set isolevel (in eV) to adjust the
            value at which the isosurface is drawn.

        Returns
        -------
        View
            A visualization of the potential isosurface within the crystal structure.
        """
        viewer = self._structure.plot(supercell)
        viewer.grid_scalars = [
            self._create_potential_isosurface(kind, component, **user_options)
            for kind, component in _parse_selection(selection)
        ]
        return viewer

    def _create_potential_isosurface(
        self, kind, component, isolevel=0, color=_config.VASP_CYAN, opacity=0.6
    ):
        self._raise_error_if_kind_incorrect(kind)
        potential_data = self._get_potential(kind)
        _raise_error_if_no_data(potential_data, kind)
        if component == "up":
            potential = potential_data[0] + potential_data[1]
        elif component == "down":
            potential = potential_data[0] - potential_data[1]
        else:
            potential = potential_data[0]
        return view.GridQuantity(
            quantity=potential.T[np.newaxis],
            label=f"{kind} potential" + (f"({component})" if component else ""),
            isosurfaces=[view.Isosurface(isolevel, color, opacity)],
        )

    def _get_potential(self, kind):
        return getattr(self._raw_data, f"{kind}_potential")

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

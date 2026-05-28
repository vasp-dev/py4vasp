# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import copy
import itertools
from typing import Optional, Union

import numpy as np

from py4vasp import _config, exception, raw as raw_module
from py4vasp._calculation import _stoichiometry
from py4vasp._calculation.dispatch import (
    DataSource,
    merge_default,
    merge_strings,
    quantity,
)
from py4vasp._calculation.structure import StructureHandler
from py4vasp._raw import data as raw
from py4vasp._third_party import graph, view
from py4vasp._util import documentation, import_, index, select, slicing
from py4vasp._util.density import SliceArguments, Visualizer

pretty = import_.optional("IPython.lib.pretty")

_DEFAULT = 0
_INTERNAL = "_density"
_COMPONENTS = {
    0: ["0", "unity", "sigma_0", "scalar", _INTERNAL],
    1: ["1", "sigma_x", "x", "sigma_1"],
    2: ["2", "sigma_y", "y", "sigma_2"],
    3: ["3", "sigma_z", "z", "sigma_3"],
}
_MAGNETIZATION = ("magnetization", "mag", "m")
_COMMON_PARAMETERS = f"""\
{slicing.PARAMETERS}
supercell : int | np.ndarray | None = None
    Replicate the contour plot periodically a given number of times. If you
    provide two different numbers, the resulting cell will be the two remaining
    lattice vectors multiplied by the specific number.
"""


def _join_with_emphasis(data):
    emph_data = [f"*{x}*" for x in filter(lambda key: key != _INTERNAL, data)]
    if len(data) < 3:
        return " and ".join(emph_data)
    emph_data.insert(-1, "and")
    return ", ".join(emph_data)


class DensityHandler:
    """Handler for density data — performs all data access and transformation."""

    def __init__(self, raw_density: raw.Density, selection_name=None):
        self._raw_density = raw_density
        self._selection_name = selection_name

    @classmethod
    def from_data(
        cls, raw_density: raw.Density, selection_name=None
    ) -> "DensityHandler":
        return cls(raw_density, selection_name=selection_name)

    def __str__(self) -> str:
        _raise_error_if_no_data(self._raw_density.charge)
        grid = self._raw_density.charge.shape[1:]
        raw_stoichiometry = self._raw_density.structure.stoichiometry
        stoichiometry = _stoichiometry.Stoichiometry.from_data(raw_stoichiometry)
        if self._selection == "kinetic_energy":
            name = "Kinetic energy"
        elif self.is_nonpolarized():
            name = "Nonpolarized"
        elif self.is_collinear():
            name = "Collinear"
        else:
            name = "Noncollinear"
        return f"""{name} density:
    structure: {pretty.pretty(stoichiometry)}
    grid: {grid[2]}, {grid[1]}, {grid[0]}"""

    def read(self) -> dict:
        return self.to_dict()

    def to_dict(self) -> dict:
        _raise_error_if_no_data(self._raw_density.charge)
        result = {"structure": self._structure().read()}
        result.update(self._read_density())
        return result

    def selections(self) -> dict:
        if self._raw_density.charge.is_none():
            return {}
        if self.is_nonpolarized():
            components = [_COMPONENTS[0][_DEFAULT]]
        elif self.is_collinear():
            components = [_COMPONENTS[0][_DEFAULT], _COMPONENTS[3][_DEFAULT]]
        else:
            components = [_COMPONENTS[i][_DEFAULT] for i in range(4)]
        return {"component": components}

    def to_numpy(self):
        return np.moveaxis(self._raw_density.charge, 0, -1).T

    def to_view(
        self,
        selection: Optional[str] = None,
        supercell: Optional[Union[int, np.ndarray]] = None,
        **user_options,
    ) -> view.View:
        _raise_error_if_no_data(self._raw_density.charge)
        map_ = self._create_map()
        selector = index.Selector({0: map_}, self._raw_density.charge)
        selection = selection or _INTERNAL
        tree = select.Tree.from_selection(selection)
        selections = list(self._filter_noncollinear_magnetization_from_selections(tree))
        structure_handler = self._structure()
        viewer = structure_handler.to_view(supercell)
        viewer.grid_scalars = [
            view.GridQuantity(
                quantity=selector[sel].T[np.newaxis],
                label=self._label(selector.label(sel)),
                isosurfaces=self._grid_quantity_properties(
                    selector, sel, map_, user_options
                ),
            )
            for sel in selections
        ]
        return viewer

    def to_contour(
        self,
        selection: Optional[str] = None,
        *,
        a: Optional[float] = None,
        b: Optional[float] = None,
        c: Optional[float] = None,
        normal: Optional[str] = None,
        supercell: Optional[Union[int, np.ndarray]] = None,
    ) -> graph.Graph:
        map_ = self._create_map()
        selector = index.Selector({0: map_}, self._raw_density.charge)
        selection = selection or _INTERNAL
        tree = select.Tree.from_selection(selection)
        selections = list(self._filter_noncollinear_magnetization_from_selections(tree))
        visualizer = Visualizer(self._structure())
        dataDict = {
            (self._label(selector.label(sel)) or "charge"): selector[sel].T
            for sel in selections
        }
        return visualizer.to_contour(
            dataDict, SliceArguments(a, b, c, normal, supercell)
        )

    def to_quiver(
        self,
        *,
        a: Optional[float] = None,
        b: Optional[float] = None,
        c: Optional[float] = None,
        normal: Optional[str] = None,
        supercell: Optional[Union[int, np.ndarray]] = None,
    ) -> graph.Graph:
        if self.is_collinear():
            data = self._raw_density.charge[1].T
        else:
            data = self.to_numpy()[1:]
        visualizer = Visualizer(self._structure())
        dataDict = {(self._selection or "magnetization"): data}
        return visualizer.to_quiver(
            dataDict, SliceArguments(a, b, c, normal, supercell)
        )

    def is_nonpolarized(self):
        return len(self._raw_density.charge) == 1

    def is_collinear(self):
        return len(self._raw_density.charge) == 2

    def is_noncollinear(self):
        return len(self._raw_density.charge) == 4

    @property
    def _selection(self):
        selection_map = {
            "kinetic_energy": "kinetic_energy",
            "kinetic_energy_density": "kinetic_energy",
        }
        return selection_map.get(self._selection_name)

    def _structure(self):
        return StructureHandler.from_data(self._raw_density.structure)

    def _read_density(self):
        density = self.to_numpy()
        if self._selection:
            yield self._selection, density
        else:
            yield "charge", density[0]
            if self.is_collinear():
                yield "magnetization", density[1]
            elif self.is_noncollinear():
                yield "magnetization", density[1:]

    def _filter_noncollinear_magnetization_from_selections(self, tree):
        if self._selection or not self.is_noncollinear():
            yield from tree.selections()
        else:
            filtered_selections = tree.selections(filter=set(_MAGNETIZATION))
            for filtered, unfiltered in zip(filtered_selections, tree.selections()):
                if filtered != unfiltered and len(filtered) != 1:
                    _raise_component_not_specified_error(unfiltered)
                yield filtered

    def _create_map(self):
        map_ = {
            choice: self._index_component(component)
            for component, choices in _COMPONENTS.items()
            for choice in choices
        }
        self._add_magnetization_for_charge_and_collinear(map_)
        return map_

    def _index_component(self, component):
        if self.is_collinear():
            component = (0, 2, 3, 1)[component]
        return component

    def _add_magnetization_for_charge_and_collinear(self, map_):
        if self._selection or not self.is_collinear():
            return
        for key in _MAGNETIZATION:
            map_[key] = 1

    def _grid_quantity_properties(self, selector, selection, map_, user_options):
        component_label = selector.label(selection)
        component = map_.get(component_label, -1)
        return self._isosurfaces(component, **user_options)

    def _label(self, component_label):
        if component_label == _INTERNAL:
            return self._selection or "charge"
        elif self._selection:
            return f"{self._selection}({component_label})"
        else:
            return component_label

    def _isosurfaces(self, component, isolevel=0.2, color=None, opacity=0.6):
        if self._use_symmetric_isosurface(component):
            _raise_error_if_color_is_specified(color)
            return [
                view.Isosurface(isolevel, _config.VASP_COLORS["blue"], opacity),
                view.Isosurface(-isolevel, _config.VASP_COLORS["red"], opacity),
            ]
        else:
            color = color or _config.VASP_COLORS["cyan"]
            return [view.Isosurface(isolevel, color, opacity)]

    def _use_symmetric_isosurface(self, component):
        if component > 0 and self.is_nonpolarized():
            _raise_is_nonpolarized_error()
        if component > 1 and self.is_collinear():
            _raise_is_collinear_error()
        return component > 0


@quantity("density")
class Density(view.Mixin):
    """This class accesses various densities (charge, magnetization, ...) of VASP."""

    def __init__(self, source, quantity_name="density", selection_name=None):
        self._source = source
        self._quantity_name = quantity_name
        self._selection_name = selection_name

    @classmethod
    def from_data(cls, raw_density):
        return cls(source=DataSource(raw_density))

    def __getitem__(self, selection_name) -> "Density":
        new = copy.copy(self)
        new._selection_name = selection_name
        return new

    def _handler_factory(self, raw):
        return DensityHandler.from_data(raw, selection_name=self._selection_name)

    def __str__(self):
        return merge_strings(
            self._source,
            self._quantity_name,
            self._selection_name,
            self._handler_factory,
            DensityHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def read(self, selection=None) -> dict:
        """Read the density into a dictionary."""
        return merge_default(
            self._source,
            self._quantity_name,
            self._selection_name,
            self._handler_factory,
            DensityHandler.read,
        )

    def to_dict(self, selection=None) -> dict:
        """Alias for read()."""
        return self.read(selection=selection)

    def selections(self, selection=None) -> dict:
        """Returns possible densities VASP can produce along with all available components."""
        result = merge_default(
            self._source,
            self._quantity_name,
            self._selection_name,
            self._handler_factory,
            DensityHandler.selections,
        )
        result["density"] = list(raw_module.selections(self._quantity_name))
        return result

    def to_numpy(self, selection=None):
        """Convert the density to a numpy array."""
        return merge_default(
            self._source,
            self._quantity_name,
            self._selection_name,
            self._handler_factory,
            DensityHandler.to_numpy,
        )

    def to_view(
        self,
        selection: Optional[str] = None,
        supercell: Optional[Union[int, np.ndarray]] = None,
        **user_options,
    ) -> view.View:
        """Plot the selected density as a 3d isosurface within the structure."""
        return merge_default(
            self._source,
            self._quantity_name,
            self._selection_name,
            self._handler_factory,
            DensityHandler.to_view,
            selection,
            supercell=supercell,
            **user_options,
        )

    def to_contour(
        self,
        selection: Optional[str] = None,
        *,
        a: Optional[float] = None,
        b: Optional[float] = None,
        c: Optional[float] = None,
        normal: Optional[str] = None,
        supercell: Optional[Union[int, np.ndarray]] = None,
    ) -> graph.Graph:
        """Generate a contour plot of the selected component of the density."""
        return merge_default(
            self._source,
            self._quantity_name,
            self._selection_name,
            self._handler_factory,
            DensityHandler.to_contour,
            selection,
            a=a,
            b=b,
            c=c,
            normal=normal,
            supercell=supercell,
        )

    def to_quiver(
        self,
        *,
        a: Optional[float] = None,
        b: Optional[float] = None,
        c: Optional[float] = None,
        normal: Optional[str] = None,
        supercell: Optional[Union[int, np.ndarray]] = None,
    ) -> graph.Graph:
        """Generate a quiver plot of magnetization density."""
        return merge_default(
            self._source,
            self._quantity_name,
            self._selection_name,
            self._handler_factory,
            DensityHandler.to_quiver,
            a=a,
            b=b,
            c=c,
            normal=normal,
            supercell=supercell,
        )

    def is_nonpolarized(self, selection=None):
        "Returns whether the density is not spin polarized."
        return merge_default(
            self._source,
            self._quantity_name,
            self._selection_name,
            self._handler_factory,
            DensityHandler.is_nonpolarized,
        )

    def is_collinear(self, selection=None):
        "Returns whether the density has a collinear magnetization."
        return merge_default(
            self._source,
            self._quantity_name,
            self._selection_name,
            self._handler_factory,
            DensityHandler.is_collinear,
        )

    def is_noncollinear(self, selection=None):
        "Returns whether the density has a noncollinear magnetization."
        return merge_default(
            self._source,
            self._quantity_name,
            self._selection_name,
            self._handler_factory,
            DensityHandler.is_noncollinear,
        )


def _raise_error_if_color_is_specified(color):
    if color is not None:
        msg = "Specifying the color of a magnetic isosurface is not implemented."
        raise exception.NotImplemented(msg)


def _raise_component_not_specified_error(selec_tuple):
    msg = (
        "Invalid selection: selection='"
        + ", ".join(selec_tuple)
        + "'. For a noncollinear calculation, the density has 4 components which can be represented in a 2x2 matrix. Specify the component of the density in terms of the Pauli matrices: sigma_1, sigma_2, sigma_3. E.g.: m(sigma_1)."
    )
    raise exception.IncorrectUsage(msg)


def _raise_is_nonpolarized_error():
    msg = "Density does not contain magnetization. Please rerun VASP with ISPIN = 2 or LNONCOLLINEAR = T to obtain it."
    raise exception.NoData(msg)


def _raise_is_collinear_error():
    msg = "Density does not contain noncollinear magnetization. Please rerun VASP with LNONCOLLINEAR = T to obtain it."
    raise exception.NoData(msg)


def _raise_error_if_no_data(data):
    if data.is_none():
        raise exception.NoData(
            "Density data was not found. Note that the density information is written "
            "on the demand to a different file (vaspwave.h5). Please make sure that "
            "this file exists and LCHARGH5 = T is set in the INCAR file. Another "
            'common issue is when you create `Calculation.from_file("vaspout.h5")` '
            "because this will overwrite the default file behavior."
        )

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import copy
from contextlib import suppress
from typing import Optional, Union

import numpy as np

from py4vasp import exception
from py4vasp._calculation import _stoichiometry
from py4vasp._calculation.dispatch import DataSource, merge_default, merge_strings, quantity
from py4vasp._calculation.structure import StructureHandler
from py4vasp._raw import data as raw
from py4vasp._third_party import graph
from py4vasp._util import database, documentation, import_, slicing
from py4vasp._util.density import SliceArguments, Visualizer

pretty = import_.optional("IPython.lib.pretty")

_TO_DATABASE_SUPPRESSED_EXCEPTIONS = (
    exception.Py4VaspError,
    AttributeError,
    TypeError,
    ValueError,
)

_COMMON_PARAMETERS = f"""selection : str | None = None
    Selects which of the possible available currents is used. Check the
    `selections` method for all available choices.

{slicing.PARAMETERS}
supercell : int | np.ndarray | None = None
    Replicate the contour plot periodically a given number of times. If you
    provide two different numbers, the resulting cell will be the two remaining
    lattice vectors multiplied by the specific number."""


class CurrentDensityHandler:
    """Handler for current density data — performs all data access and transformation."""

    def __init__(self, raw_current_density: raw.CurrentDensity):
        self._raw_current_density = raw_current_density

    @classmethod
    def from_data(cls, raw_current_density: raw.CurrentDensity) -> "CurrentDensityHandler":
        return cls(raw_current_density)

    def __str__(self) -> str:
        raw_stoichiometry = self._raw_current_density.structure.stoichiometry
        stoichiometry = _stoichiometry.Stoichiometry.from_data(raw_stoichiometry)
        key = self._raw_current_density.valid_indices[-1]
        grid = self._raw_current_density[key].current_density.shape[1:]
        return f"""current density:
    structure: {pretty.pretty(stoichiometry)}
    grid: {grid[2]}, {grid[1]}, {grid[0]}
    selections: {", ".join(str(index) for index in self._raw_current_density.valid_indices)}"""

    def read(self) -> dict:
        return self.to_dict()

    def to_dict(self) -> dict:
        """Read the current density and structural information into a Python dictionary."""
        return {"structure": self._structure().read(), **self._read_current_densities()}

    def to_database(self) -> dict:
        density_dict = {"current_density": {}}
        structure_ = {}
        with suppress(*_TO_DATABASE_SUPPRESSED_EXCEPTIONS):
            structure_ = self._structure().to_database()
        return database.combine_db_dicts(density_dict, structure_)

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
        """Generate a contour plot of current density."""
        label, grid_vector = self._read_current_density(selection)
        grid_scalar = np.linalg.norm(grid_vector, axis=-1)
        visualizer = Visualizer(self._structure())
        return visualizer.to_contour(
            {label: grid_scalar},
            SliceArguments(a, b, c, normal, supercell),
            isolevels=False,
        )

    def to_quiver(
        self,
        selection: Optional[str] = None,
        *,
        a: Optional[float] = None,
        b: Optional[float] = None,
        c: Optional[float] = None,
        normal: Optional[str] = None,
        supercell: Optional[Union[int, np.ndarray]] = None,
    ) -> graph.Graph:
        """Generate a quiver plot of current density."""
        label, data = self._read_current_density(selection)
        sel_data = np.moveaxis(data, -1, 0)
        visualizer = Visualizer(self._structure())
        return visualizer.to_quiver(
            {label: sel_data}, SliceArguments(a, b, c, normal, supercell)
        )

    def _structure(self):
        return StructureHandler.from_data(self._raw_current_density.structure)

    def _read_current_densities(self):
        return dict(self._read_current_density(key) for key in self._raw_current_density)

    def _read_current_density(self, key=None):
        key = key or self._raw_current_density.valid_indices[-1]
        if key not in self._raw_current_density.valid_indices:
            raise exception.IncorrectUsage(
                f"Selection {key!r} is not available. "
                f"Please use one of {list(self._raw_current_density.valid_indices)}."
            )
        return f"current_{key}", self._raw_current_density[key].current_density[:].T


@quantity("current_density")
class CurrentDensity:
    """Represents current density on the grid in the unit cell.

    A current density j is a vectorial quantity (j_x, j_y, j_z) on every grid point.
    It describes how the current flows at every point in space."""

    def __init__(self, source, quantity_name="current_density"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_current_density):
        return cls(source=DataSource(raw_current_density))

    def _handler_factory(self, raw):
        return CurrentDensityHandler.from_data(raw)

    def __str__(self):
        return merge_strings(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            CurrentDensityHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def read(self, selection=None) -> dict:
        """Read the current density and structural information into a Python dictionary."""
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            CurrentDensityHandler.read,
        )

    def to_dict(self, selection=None) -> dict:
        """Alias for read()."""
        return self.read(selection=selection)

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
        """Generate a contour plot of current density."""
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            CurrentDensityHandler.to_contour,
            selection,
            a=a,
            b=b,
            c=c,
            normal=normal,
            supercell=supercell,
        )

    def to_quiver(
        self,
        selection: Optional[str] = None,
        *,
        a: Optional[float] = None,
        b: Optional[float] = None,
        c: Optional[float] = None,
        normal: Optional[str] = None,
        supercell: Optional[Union[int, np.ndarray]] = None,
    ) -> graph.Graph:
        """Generate a quiver plot of current density."""
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            CurrentDensityHandler.to_quiver,
            selection,
            a=a,
            b=b,
            c=c,
            normal=normal,
            supercell=supercell,
        )

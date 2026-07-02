# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import copy
from typing import Optional, Union

import numpy as np

from py4vasp import exception
from py4vasp._calculation import _stoichiometry
from py4vasp._calculation.dispatch import (
    DataSource,
    _dispatch,
    merge_default,
    merge_strings,
    merge_to_database,
    quantity,
)
from py4vasp._calculation.structure import StructureHandler
from py4vasp._raw import data as raw
from py4vasp._third_party import graph
from py4vasp._util import documentation, import_, slicing
from py4vasp._util.density import SliceArguments, Visualizer

pretty = import_.optional("IPython.lib.pretty")

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
    def from_data(
        cls, raw_current_density: raw.CurrentDensity
    ) -> "CurrentDensityHandler":
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

    def to_dict(self) -> dict:
        """Read the current density and structural information into a Python dictionary."""
        return {
            "structure": self._structure().to_dict(),
            **self._read_current_densities(),
        }

    def to_database(self) -> dict:
        return {}

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
        return dict(
            self._read_current_density(key) for key in self._raw_current_density
        )

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
    It describes how the current flows at every point in space.

    Examples
    --------

    First, we create some example data that you can follow along. Please define a
    variable `path` with the path to a directory that exists and does not contain any
    VASP calculation data. Alternatively, you can use your own data if you have run
    VASP and construct `calculation` from it.

    >>> from py4vasp import demo
    >>> calculation = demo.calculation(path)

    To produce current density plots, please check the `to_contour` and `to_quiver`
    functions for a more detailed documentation.

    To produce a contour plot:

    >>> calculation.current_density.to_contour("nmr", a=0)
    Graph(series=[Contour(data=array([[...]]), ..., cut='a', ...)], ...)

    To produce a quiver plot:

    >>> calculation.current_density.to_quiver("nmr", c=0)
    Graph(series=[Contour(data=array([[[...]]]), ..., cut='c', ...)], ...)

    For your own postprocessing, you can read the current density data into a Python dict:

    >>> calculation.current_density.read("nmr")
    {'structure': {...}, 'current_x': array([[[[...]]]],  ...), 'current_y': array([[[[...]]]], ...), 'current_z': array([[[[...]]]],  ...)}

    You can inspect possible choices with:

    >>> calculation.current_density.selections("nmr")
    {'current_density': ['nmr']}

    Please check the documentation of these methods for more details on how to use them
    and which options they provide."""

    def __init__(self, source, quantity_name="current_density"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_current_density):
        return cls(source=DataSource(raw_current_density))

    def _handler_factory(self, raw):
        return CurrentDensityHandler.from_data(raw)

    def __str__(self, selection=None):
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            CurrentDensityHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def read(self) -> dict:
        """Read the current density and structural information into a Python dictionary.

        Returns
        -------
        dict
            Contains all available current density data as well as structural information.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            CurrentDensityHandler.to_dict,
        )

    def to_dict(self) -> dict:
        """Convenient alias for :py:meth:`read`. Please read the documentation there."""
        return self.read()

    @documentation.format(plane=slicing.PLANE, parameters=_COMMON_PARAMETERS)
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
        """Generate a contour plot of current density.

        {plane}

        Parameters
        ----------
        {parameters}

        Returns
        -------
        Graph
            A current density plot in the plane spanned by the 2 remaining lattice
            vectors.

        Examples
        --------

        Cut a plane at the origin of the third lattice vector.

        >>> calculation.current_density.to_contour("nmr", c=0)

        Replicate a plane in the middle of the second lattice vector 2 times in each
        direction.

        >>> calculation.current_density.to_contour("nmr", b=0.5, supercell=2)

        Take a slice along the first lattice vector and rotate it such that the normal
        of the plane aligns with the x axis.

        >>> calculation.current_density.to_contour("nmr", a=0.3, normal="x")
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            CurrentDensityHandler.to_contour,
            a=a,
            b=b,
            c=c,
            normal=normal,
            supercell=supercell,
        )

    @documentation.format(plane=slicing.PLANE, parameters=_COMMON_PARAMETERS)
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
        """Generate a quiver plot of current density.

        {plane}

        Parameters
        ----------
        {parameters}

        Returns
        -------
        Graph
            A quiver plot in the plane spanned by the 2 remaining lattice vectors.

        Examples
        --------

        Cut a plane at the origin of the third lattice vector.

        >>> calculation.current_density.to_quiver("nmr", c=0)

        Replicate a plane in the middle of the second lattice vector 2 times in each
        direction.

        >>> calculation.current_density.to_quiver("nmr", b=0.5, supercell=2)

        Take a slice along the first lattice vector and rotate it such that the normal
        of the plane aligns with the x axis.

        >>> calculation.current_density.to_quiver("nmr", a=0.3, normal="x")
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            CurrentDensityHandler.to_quiver,
            a=a,
            b=b,
            c=c,
            normal=normal,
            supercell=supercell,
        )

    def _to_database(self) -> dict:
        """Return {quantity[_selection]: handler_result} for database storage."""
        return merge_to_database(
            self._source,
            self._quantity_name,
            CurrentDensityHandler.from_data,
            CurrentDensityHandler.to_database,
        )

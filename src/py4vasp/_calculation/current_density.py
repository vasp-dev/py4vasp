# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from contextlib import suppress
from typing import Optional, Union

import numpy as np

from py4vasp import exception
from py4vasp._calculation import _stoichiometry, base, structure
from py4vasp._raw import data as raw_data
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

_COMMON_PARAMETERS = f"""\
selection : str | None = None
    Selects which of the possible available currents is used. Check the
    `selections` method for all available choices.

{slicing.PARAMETERS}
supercell : int | np.ndarray | None = None
    Replicate the contour plot periodically a given number of times. If you
    provide two different numbers, the resulting cell will be the two remaining
    lattice vectors multiplied by the specific number."""


class CurrentDensity(base.Refinery, structure.Mixin):
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

    To produce current density plots, please check the `to_contour` and `to_quiver` functions for
    a more detailed documentation.

    To produce a contour plot:

    >>> calculation.current_density.to_contour("nmr", a=0)
    Graph(series=[Contour(data=array([[...]]), ..., cut='a', ...)], ...)

    To produce a quiver plot:

    >>> calculation.current_density.to_quiver("nmr", c=0)
    Graph(series=[Contour(data=array([[[...]]]), ..., cut='c', ...)], ...)

    For your own postprocessing, you can read the current density data into a Python dict:

    >>> calculation.current_density.read("nmr")
    {'structure': {...}, 'current_x': array([[[[...]]]], ...), 'current_y': array([[[[...]]]], ...), 'current_z': array([[[[...]]]], ...)}

    You can inspect possible choices with:

    >>> calculation.current_density.selections("nmr")
    {'current_density': ['nmr']}

    Please check the documentation of these methods for more details on how to use them and which options they provide.
    """

    _raw_data: raw_data.CurrentDensity

    @base.data_access
    def __str__(self):
        raw_stoichiometry = self._raw_data.structure.stoichiometry
        stoichiometry = _stoichiometry.Stoichiometry.from_data(raw_stoichiometry)
        key = self._raw_data.valid_indices[-1]
        grid = self._raw_data[key].current_density.shape[1:]
        return f"""\
current density:
    structure: {pretty.pretty(stoichiometry)}
    grid: {grid[2]}, {grid[1]}, {grid[0]}
    selections: {", ".join(str(index) for index in self._raw_data.valid_indices)}"""

    @base.data_access
    def to_dict(self):
        """Read the current density and structural information into a Python dictionary.

        Returns
        -------
        dict
            Contains all available current density data as well as structural information.
        """
        return {"structure": self._structure.read(), **self._read_current_densities()}

    def _read_current_densities(self):
        return dict(self._read_current_density(key) for key in self._raw_data)

    def _read_current_density(self, key=None):
        key = key or self._raw_data.valid_indices[-1]
        return f"current_{key}", self._raw_data[key].current_density[:].T

    @base.data_access
    def _to_database(self, *args, **kwargs):
        density_dict = {"current_density": {}}
        structure_ = {}
        with suppress(*_TO_DATABASE_SUPPRESSED_EXCEPTIONS):
            structure_ = structure.Structure.from_data(
                self._raw_data.structure
            )._read_to_database(*args, **kwargs)
        return database.combine_db_dicts(density_dict, structure_)

    @base.data_access
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
            A current density plot in the plane spanned by the 2 remaining lattice vectors.

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
        label, grid_vector = self._read_current_density(selection)
        grid_scalar = np.linalg.norm(grid_vector, axis=-1)

        # set up Visualizer
        visualizer = Visualizer(self._structure)
        return visualizer.to_contour(
            {label: grid_scalar},
            SliceArguments(a, b, c, normal, supercell),
            isolevels=False,
        )

    @base.data_access
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
        # set up data
        label, data = self._read_current_density(selection)
        sel_data = np.moveaxis(data, -1, 0)

        # set up Visualizer
        visualizer = Visualizer(self._structure)
        return visualizer.to_quiver(
            {label: sel_data}, SliceArguments(a, b, c, normal, supercell)
        )

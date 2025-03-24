# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np

from py4vasp import exception
from py4vasp._calculation import _stoichiometry, base, structure
from py4vasp._third_party import graph
from py4vasp._util import documentation, import_, slicing

pretty = import_.optional("IPython.lib.pretty")

_COMMON_PARAMETERS = f"""\
selection : str or None
    Selects which of the possible available currents is used. Check the
    `selections` method for all available choices.

{slicing.PARAMETERS}
supercell : int or np.ndarray
    Replicate the contour plot periodically a given number of times. If you
    provide two different numbers, the resulting cell will be the two remaining
    lattice vectors multiplied by the specific number."""


class CurrentDensity(base.Refinery, structure.Mixin):
    """Represents current density on the grid in the unit cell.

    A current density j is a vectorial quantity (j_x, j_y, j_z) on every grid point.
    It describes how the current flows at every point in space.
    """

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
    @documentation.format(plane=slicing.PLANE, parameters=_COMMON_PARAMETERS)
    def to_contour(
        self, selection=None, *, a=None, b=None, c=None, normal=None, supercell=None
    ):
        """Generate a contour plot of current density.

        {plane}

        Parameters
        ----------
        {parameters}

        Returns
        -------
        graph
            A current density plot in the plane spanned by the 2 remaining lattice vectors.

        Examples
        --------

        Cut a plane at the origin of the third lattice vector.

        >>> calculation.current_density.to_contour(c=0)

        Replicate a plane in the middle of the second lattice vector 2 times in each
        direction.

        >>> calculation.current_density.to_contour(b=0.5, supercell=2)

        Take a slice along the first lattice vector and rotate it such that the normal
        of the plane aligns with the x axis.

        >>> calculation.current_density.to_contour(a=0.3, normal="x")
        """
        cut, fraction = slicing.get_cut(a, b, c)
        plane = slicing.plane(self._structure.lattice_vectors(), cut, normal)
        label, grid_vector = self._read_current_density(selection)
        grid_scalar = np.linalg.norm(grid_vector, axis=-1)
        grid_scalar = slicing.grid_scalar(grid_scalar, plane, fraction)
        contour_plot = graph.Contour(grid_scalar, plane, label)
        if supercell is not None:
            contour_plot.supercell = np.ones(2, dtype=np.int_) * supercell
        return graph.Graph([contour_plot])

    @base.data_access
    @documentation.format(plane=slicing.PLANE, parameters=_COMMON_PARAMETERS)
    def to_quiver(
        self, selection=None, *, a=None, b=None, c=None, normal=None, supercell=None
    ):
        """Generate a quiver plot of current density.

        {plane}


        Parameters
        ----------
        {parameters}

        Returns
        -------
        graph
            A quiver plot in the plane spanned by the 2 remaining lattice vectors.

        Examples
        --------

        Cut a plane at the origin of the third lattice vector.

        >>> calculation.current_density.to_quiver(c=0)

        Replicate a plane in the middle of the second lattice vector 2 times in each
        direction.

        >>> calculation.current_density.to_quiver(b=0.5, supercell=2)

        Take a slice along the first lattice vector and rotate it such that the normal
        of the plane aligns with the x axis.

        >>> calculation.current_density.to_quiver(a=0.3, normal="x")
        """
        cut, fraction = slicing.get_cut(a, b, c)
        plane = slicing.plane(self._structure.lattice_vectors(), cut, normal)
        label, data = self._read_current_density(selection)
        sliced_data = slicing.grid_vector(np.moveaxis(data, -1, 0), plane, fraction)
        quiver_plot = graph.Contour(sliced_data, plane, label)
        if supercell is not None:
            quiver_plot.supercell = np.ones(2, dtype=np.int_) * supercell
        return graph.Graph([quiver_plot])

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np

from py4vasp import exception
from py4vasp._calculation import _stoichiometry, base, structure
from py4vasp._third_party import graph
from py4vasp._util import database, documentation, import_, slicing
from py4vasp._util.density import SliceArguments, Visualizer

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
    def _to_database(self, *args, **kwargs):
        density_dict = {"current_density": {}}
        try:
            key = self._raw_data.valid_indices[-1]
            grid = self._raw_data[key].current_density.shape[1:]
            density_dict = {
                "current_density": {
                    # TODO move to Setup dataclass instead
                    "grid_shape_coarse": [grid[2], grid[1], grid[0]],
                    "grid_shape_fine": None, # TODO implement
                }
            }
        except Exception as exc:
            pass
        structure_ = structure.Structure.from_data(
            self._raw_data.structure
        )._read_to_database(*args, **kwargs)
        return database.combine_db_dicts(density_dict, structure_)

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
        # set up data
        label, data = self._read_current_density(selection)
        sel_data = np.moveaxis(data, -1, 0)

        # set up Visualizer
        visualizer = Visualizer(self._structure)
        return visualizer.to_quiver(
            {label: sel_data}, SliceArguments(a, b, c, normal, supercell)
        )

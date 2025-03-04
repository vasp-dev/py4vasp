# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)


import numpy as np

from py4vasp import _config
from py4vasp._calculation import _stoichiometry, base, structure
from py4vasp._third_party import graph, view
from py4vasp._util import documentation, import_, index, select, slicing

pretty = import_.optional("IPython.lib.pretty")

_DEFAULT_SELECTION: str = "isotropic"


class Nics(base.Refinery, structure.Mixin, view.Mixin):
    """This class accesses information on the nucleus-independent chemical shift (NICS)."""

    @base.data_access
    def to_dict(self):
        """Read NICS into a dictionary.

        Parameters
        ----------

        Returns
        -------
        dict
            Contains the structure information as well as the nucleus-independent chemical shift represented on a grid in the unit cell.
        """
        result = {
            "structure": self._structure.read(),
            "nics": self.to_numpy(),
        }
        return result

    @base.data_access
    def __str__(self):
        grid = self._raw_data.nics.shape[1:]
        raw_stoichiometry = self._raw_data.structure.stoichiometry
        stoichiometry = _stoichiometry.Stoichiometry.from_data(raw_stoichiometry)
        return f"""\
nucleus-independent chemical shift:
    structure: {pretty.pretty(stoichiometry)}
    grid: {grid[2]}, {grid[1]}, {grid[0]}
    tensor shape: 3x3"""

    @staticmethod
    def _init_directions_dict():
        return {
            "isotropic": [0, 4, 8],
            "xx": 0,
            "xy": 1,
            "xz": 2,
            "yx": 3,
            "yy": 4,
            "yz": 5,
            "zx": 6,
            "zy": 7,
            "zz": 8,
        }

    @staticmethod
    def _read_selected_data(data, selection):
        selector = index.Selector(
            {3: Nics._init_directions_dict()}, data, reduction=np.average
        )
        tree = select.Tree.from_selection(selection)
        return {
            selector.label(selection): selector[selection]
            for selection in tree.selections()
        }

    @base.data_access
    def to_numpy(self, selection=None):
        """Convert NICS to a numpy array.

        The resulting shape will be the NICS grid data with respect to the selection.

        Parameters
        ----------
        selection : str or None
            The tensor element(s) to extract.
            Can be None (in which case the whole tensor is returned), isotropic, or one of "xx", "xy", ...

        Returns
        -------
        np.ndarray
            All components of NICS.
        """
        transposed_nics = np.array(self._raw_data.nics).T
        curr_shape = transposed_nics.shape
        if selection is None:
            transposed_nics = transposed_nics.reshape((*curr_shape[:-1], 3, 3))
        else:
            transposed_nics = np.squeeze(
                list(Nics._read_selected_data(transposed_nics, selection).values())
            )
        return transposed_nics

    def _isosurfaces(self, isolevel=1.0, opacity=0.6):
        return [
            view.Isosurface(isolevel, _config.VASP_COLORS["blue"], opacity),
            view.Isosurface(-isolevel, _config.VASP_COLORS["red"], opacity),
        ]

    @base.data_access
    def to_view(self, selection=None, supercell=None, **user_options):
        """Plot the selected chemical shift as a 3d isosurface within the structure.

        Parameters
        ----------
        selection : str or None
            Axis along which to plot.
            Can be one of "xx", "xy", ...
            Can also be "isotropic" to plot the trace.
            If selection is None, it defaults to "isotropic".

        supercell : int or np.ndarray
            If present the data is replicated the specified number of times along each
            direction.

        user_options
            Further arguments with keyword that get directly passed on to the
            visualizer. Most importantly, you can set isolevel to adjust the
            value at which the isosurface is drawn.

        Returns
        -------
        View
            Visualize an isosurface of the selected chemical shift within the 3d structure.

        Examples
        --------
        >>> from py4vasp import calculation

        Plot the isotropic chemical shift as a 3d isosurface.
        >>> calculation.nics.plot()

        Plot the chemical shift with "xx" selection as a 3d isosurface.
        >>> calculation.nics.plot(selection="xx")

        Plot the isotropic chemical shift with specified isolevel as a 3d isosurface.
        >>> calculation.nics.plot(isolevel=0.6)
        """
        selection = selection or _DEFAULT_SELECTION
        viewer = self._structure.plot(supercell)
        selected_data = Nics._read_selected_data(
            np.array(self._raw_data.nics).T, selection
        )
        viewer.grid_scalars = [
            view.GridQuantity(
                quantity=(v)[np.newaxis],
                label=f"{k} NICS",
                isosurfaces=self._isosurfaces(**user_options),
            )
            for k, v in selected_data.items()
        ]
        return viewer

    @base.data_access
    @documentation.format(plane=slicing.PLANE, parameters=slicing.PARAMETERS)
    def to_contour(
        self, selection=None, *, a=None, b=None, c=None, normal=None, supercell=None
    ):
        """Generate a contour plot of chemical shift.

        {plane}

        Parameters
        ----------
        {parameters}

        selection : str or None
            Axis along which to plot.
            Can be one of "xx", "xy", ...
            Can also be "isotropic" to plot the trace.
            If selection is None, it defaults to "isotropic".

        supercell : int or np.ndarray
            If present the data is replicated the specified number of times along each
            direction.

        Returns
        -------
        graph
            A chemical shift plot in the plane spanned by the 2 remaining lattice vectors.

        Examples
        --------
        >>> from py4vasp import calculation

        Cut a plane through the isotropic chemical shift at the origin of the third lattice
        vector.

        >>> calculation.nics.to_contour(c=0)

        Replicate a plane in the middle of the second lattice vector 2 times in each
        direction.

        >>> calculation.nics.to_contour(b=0.5, supercell=2)

        Take a slice of the chemical shift with "xy" selection along the first lattice
        vector and
        rotate it such that the plane normal aligns with the x axis.

        >>> calculation.nics.to_contour(a=0.3, selection=0.3, normal="x")

        Cut a plan through the isotropic chemical shift at the origin of the third lattice
        vector, then show isosurface level values along contour lines.

        >>> plot = calculation.nics.to_contour(c=0, selection=0.3, normal="x")
        >>> plot.series[0].show_contour_values = True
        >>> plot.show()
        """
        selection = selection or _DEFAULT_SELECTION
        cut, fraction = slicing.get_cut(a, b, c)
        plane = slicing.plane(self._structure.lattice_vectors(), cut, normal)
        selected_data = Nics._read_selected_data(
            np.array(self._raw_data.nics).T, selection
        )
        contour_plots = []
        for k, v in selected_data.items():
            grid_scalar = slicing.grid_scalar(v, plane, fraction)
            contour_plot = graph.Contour(
                grid_scalar,
                plane,
                f"{k} NICS contour ({cut})",
                isolevels=True,
            )
            if supercell is not None:
                contour_plot.supercell = np.ones(2, dtype=np.int_) * supercell
            contour_plots.append(contour_plot)
        return graph.Graph(contour_plots)

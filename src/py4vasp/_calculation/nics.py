# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)


import numpy as np

from py4vasp import _config
from py4vasp._calculation import base, structure
from py4vasp._third_party import view
from py4vasp._util import index, select

_DEFAULT_SELECTION: str = "isotropic"


class Nics(base.Refinery, structure.Mixin, view.Mixin):
    """This class accesses information on the nucleus-independent chemical shift (NICS)."""

    @base.data_access
    def to_dict(self):
        """Read nics into a dictionary.

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
        """Plot the selected density as a 3d isosurface within the structure.

        Parameters
        ----------
        selection : str
            Axis along which to plot.
            Can be one of "xx", "xy", ...

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
            Visualize an isosurface of the density within the 3d structure.
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

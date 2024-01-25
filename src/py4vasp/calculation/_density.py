# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import data, exception
from py4vasp._data import base, structure
from py4vasp._util import import_

pretty = import_.optional("IPython.lib.pretty")


class _ViewerWrapper:
    def __init__(self, viewer):
        self._viewer = viewer
        self._options = {"isolevel": 0.2, "color": "yellow", "opacity": 0.6}

    def show_isosurface(self, data, **options):
        options = {**self._options, **options}
        self._viewer.show_isosurface(data, **options)


class Density(base.Refinery, structure.Mixin):
    """The charge and magnetization density.

    You can use this class to extract the density data of the VASP calculation
    and to have a quick glance at the resulting density.
    """

    @base.data_access
    def __str__(self):
        _raise_error_if_no_data(self._raw_data.charge)
        grid = self._raw_data.charge.shape[1:]
        topology = data.Topology.from_data(self._raw_data.structure.topology)
        if self.nonpolarized():
            name = "nonpolarized"
        elif self.collinear():
            name = "collinear"
        else:
            name = "noncollinear"
        return f"""{name} density:
    structure: {pretty.pretty(topology)}
    grid: {grid[2]}, {grid[1]}, {grid[0]}"""

    @base.data_access
    def to_dict(self):
        """Read the electronic density into a dictionary.

        Returns
        -------
        dict
            Contains the structure information as well as the density represented
            of a grid in the unit cell.
        """
        _raise_error_if_no_data(self._raw_data.charge)
        result = {"structure": self._structure.read()}
        result.update(self._read_density())
        return result

    def _read_density(self):
        density = np.moveaxis(self._raw_data.charge, 0, -1).T
        yield "charge", density[0]
        if self.collinear():
            yield "magnetization", density[1]
        elif self.noncollinear():
            yield "magnetization", density[1:]

    @base.data_access
    def plot(self, selection="charge", **user_options):
        """Plot the selected density as a 3d isosurface within the structure.

        Parameters
        ----------
        selection : str
            Can be either *charge* or *magnetization*, depending on which quantity
            should be visualized.
        user_options
            Further arguments with keyword that get directly passed on to the
            visualizer. Most importantly, you can set isolevel to adjust the
            value at which the isosurface is drawn.

        Returns
        -------
        Viewer3d
            Visualize an isosurface of the density within the 3d structure.
        """
        _raise_error_if_no_data(self._raw_data.charge)
        viewer = self._structure.plot()
        if selection == "charge":
            self._plot_charge(_ViewerWrapper(viewer), **user_options)
        elif selection == "magnetization":
            self._plot_magnetism(_ViewerWrapper(viewer), **user_options)
        else:
            msg = f"'{selection}' is an unknown option, please use 'charge' or 'magnetization' instead."
            raise exception.IncorrectUsage(msg)
        return viewer

    @base.data_access
    def nonpolarized(self):
        "Returns whether the density is not spin polarized."
        return len(self._raw_data.charge) == 1

    @base.data_access
    def collinear(self):
        "Returns whether the density has a collinear magnetization."
        return len(self._raw_data.charge) == 2

    @base.data_access
    def noncollinear(self):
        "Returns whether the density has a noncollinear magnetization."
        return len(self._raw_data.charge) == 4

    def _plot_charge(self, viewer, **user_options):
        viewer.show_isosurface(self._raw_data.charge[0], **user_options)

    def _plot_magnetism(self, viewer, **user_options):
        if self.nonpolarized():
            _raise_is_nonpolarized_error()
        if self.collinear():
            return self._plot_collinear_magnetism(viewer, **user_options)
        if self.noncollinear():
            return self._plot_noncollinear_magnetism(viewer, **user_options)

    def _plot_collinear_magnetism(self, viewer, **user_options):
        _raise_error_if_color_is_specified(**user_options)
        magnetization = self._raw_data.charge[1]
        viewer.show_isosurface(magnetization, color="blue", **user_options)
        viewer.show_isosurface(-magnetization, color="red", **user_options)

    def _plot_noncollinear_magnetism(self, viewer, **user_options):
        magnetization = np.linalg.norm(self._raw_data.charge[1:], axis=0)
        viewer.show_isosurface(magnetization, **user_options)


def _raise_is_nonpolarized_error():
    msg = "Density does not contain magnetization. Please rerun VASP with ISPIN = 2 or LNONCOLLINEAR = T to obtain it."
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


def _raise_error_if_color_is_specified(**user_options):
    if "color" in user_options:
        msg = "Specifying the color of a magnetic isosurface is not implemented."
        raise exception.NotImplemented(msg)

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from IPython.lib.pretty import pretty

from py4vasp import data, exception
from py4vasp._data import base, structure


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
        grid = self._raw_data.charge.shape[1:]
        return f"""density:
    structure: {pretty(data.Topology.from_data(self._raw_data.structure.topology))}
    grid: {grid[0]}, {grid[1]}, {grid[2]}
    {"spin polarized" if self._spin_polarized() else ""}
        """.strip()

    @base.data_access
    def to_dict(self):
        """Read the electronic density into a dictionary.

        Returns
        -------
        dict
            Contains the structure information as well as the density represented
            of a grid in the unit cell.
        """
        return {
            "structure": self._structure.read(),
            "charge": self._raw_data.charge[0],
            "magnetization": self._magnetization_if_present(),
        }

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
        viewer = self._structure.plot()
        if selection == "charge":
            self._plot_charge(_ViewerWrapper(viewer), **user_options)
        elif selection == "magnetization":
            self._plot_magnetism(_ViewerWrapper(viewer), **user_options)
        else:
            msg = f"'{selection}' is an unknown option, please use 'charge' or 'magnetization' instead."
            raise exception.IncorrectUsage(msg)
        return viewer

    def _magnetization_if_present(self):
        if self._spin_polarized():
            return self._raw_data.charge[1]
        else:
            return None

    def _spin_polarized(self):
        return len(self._raw_data.charge) > 1

    def _plot_charge(self, viewer, **user_options):
        viewer.show_isosurface(self._raw_data.charge[0], **user_options)

    def _plot_magnetism(self, viewer, **user_options):
        self._raise_error_if_not_spin_polarized()
        _raise_error_if_color_is_specified(**user_options)
        viewer.show_isosurface(self._raw_data.charge[1], color="blue", **user_options)
        viewer.show_isosurface(-self._raw_data.charge[1], color="red", **user_options)

    def _raise_error_if_not_spin_polarized(self):
        if not self._spin_polarized():
            msg = "Density does not contain magnetization. Please rerun VASP with ISPIN = 2 to obtain it."
            raise exception.NoData(msg)


def _raise_error_if_color_is_specified(**user_options):
    if "color" in user_options:
        msg = "Specifying the color of a magnetic isosurface is not implemented."
        raise exception.NotImplemented(msg)

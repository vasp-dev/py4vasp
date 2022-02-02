# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp.data import Viewer3d, Structure, Topology
from py4vasp.data._base import DataBase, RefinementDescriptor
from IPython.lib.pretty import pretty
from pathlib import Path
import py4vasp.exceptions as exceptions

_filename = "vaspwave.h5"


class _ViewerWrapper:
    def __init__(self, viewer):
        self._viewer = viewer
        self._options = {"isolevel": 0.2, "color": "yellow", "opacity": 0.6}

    def show_isosurface(self, data, **options):
        options = {**self._options, **options}
        self._viewer.show_isosurface(data, **options)


class Density(DataBase):
    """The charge and magnetization density.

    You can use this class to extract the density data of the Vasp calculation
    and to have a quick glance at the resulting density.

    Parameters
    ----------
    raw_density : RawDensity
        Dataclass containing the raw density data as well as structural information.
    """

    read = RefinementDescriptor("_to_dict")
    to_dict = RefinementDescriptor("_to_dict")
    plot = RefinementDescriptor("_to_viewer3d")
    to_viewer3d = RefinementDescriptor("_to_viewer3d")
    __str__ = RefinementDescriptor("_to_string")

    @classmethod
    def from_file(cls, file=None):
        if file is None:
            file = _filename
        elif (isinstance(file, Path) or isinstance(file, str)) and Path(file).is_dir():
            file = file / _filename
        return super().from_file(file)

    def _to_string(self):
        grid = self._raw_data.charge.shape[1:]
        return f"""density:
    structure: {pretty(Topology(self._raw_data.structure.topology))}
    grid: {grid[0]}, {grid[1]}, {grid[2]}
    {"spin polarized" if self._spin_polarized() else ""}
        """.strip()

    def _to_dict(self):
        """Read the electionic density into a dictionary.

        Returns
        -------
        dict
            Contains the structure information as well as the density represented
            of a grid in the unit cell.
        """
        return {
            "structure": self._structure().read(),
            "charge": self._raw_data.charge[0],
            "magnetization": self._magnetization_if_present(),
        }

    def _to_viewer3d(self, selection="charge", **user_options):
        """Plot the selected density as a 3d isosurface within the structure.

        Parameters
        ----------
        selection : str
            Can be either *charge* or *magnetization*, dependending on which quantity
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
        viewer = self._structure().plot()
        if selection == "charge":
            self._plot_charge(_ViewerWrapper(viewer), **user_options)
        elif selection == "magnetization":
            self._plot_magnetism(_ViewerWrapper(viewer), **user_options)
        else:
            msg = f"'{selection}' is an unknown option, please use 'charge' or 'magnetization' instead."
            raise exceptions.IncorrectUsage(msg)
        return viewer

    def _structure(self):
        return Structure(self._raw_data.structure)

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
            msg = "Density does not contain magnetization. Please rerun Vasp with ISPIN = 2 to obtain it."
            raise exceptions.NoData(msg)


def _raise_error_if_color_is_specified(**user_options):
    if "color" in user_options:
        msg = "Specifying the color of a magnetic isosurface is not implemented."
        raise exceptions.NotImplemented(msg)

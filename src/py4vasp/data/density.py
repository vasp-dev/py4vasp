from py4vasp.data import _util, Viewer3d, Structure, Topology
from IPython.lib.pretty import pretty
import py4vasp.exceptions as exceptions

_filename = "vaspwave.h5"


class _ViewerWrapper:
    def __init__(self, viewer):
        self._viewer = viewer
        self._options = {"isolevel": 0.2, "color": "yellow", "opacity": 0.6}

    def show_isosurface(self, data, **options):
        options = {**self._options, **options}
        self._viewer.show_isosurface(data, **options)


@_util.add_specific_wrappers({"plot": "to_viewer3d"})
class Density(_util.Data):
    """ The charge and magnetization density.

    You can use this class to extract the density data of the Vasp calculation
    and to have a quick glance at the resulting density.

    Parameters
    ----------
    raw_density : raw.Density
        Dataclass containing the raw density data as well as structural information.
    """

    @classmethod
    @_util.add_doc(_util.from_file_doc("electronic density", filename=_filename))
    def from_file(cls, file=None):
        if file is None:
            file = _filename
        return _util.from_file(cls, file, "density")

    def _repr_pretty_(self, p, cycle):
        structure = f"   structure: {pretty(Topology(self._raw.structure.topology))}"
        grid = "   grid: {}, {}, {}".format(*self._raw.charge.shape[1:])
        spin_polarized = "\n   spin polarized" if len(self._raw.charge) > 1 else ""
        p.text(f"density:\n{structure}\n{grid}{spin_polarized}")

    def to_dict(self):
        """ Read the electionic density into a dictionary.

        Returns
        -------
        dict
            Contains the structure information as well as the density represented
            of a grid in the unit cell.
        """
        return {
            "structure": Structure(self._raw.structure).read(),
            "charge": self._raw.charge[0],
            **self._magnetization_if_present(),
        }

    def _magnetization_if_present(self):
        if len(self._raw.charge) > 1:
            return {"magnetization": self._raw.charge[1]}
        else:
            return {}

    def to_viewer3d(self, quantity="charge", **user_options):
        """ Plot the selected density as a 3d isosurface within the structure.

        Parameters
        ----------
        quantity : str
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
        viewer = Structure(self._raw.structure).plot()
        if quantity == "charge":
            self._plot_charge(_ViewerWrapper(viewer), **user_options)
        elif quantity == "magnetization":
            self._plot_magnetism(_ViewerWrapper(viewer), **user_options)
        else:
            msg = f"'{quantity}' is an unknown option, please use 'charge' or 'magnetization' instead."
            raise exceptions.IncorrectUsage(msg)
        return viewer

    def _plot_charge(self, viewer, **user_options):
        viewer.show_isosurface(self._raw.charge[0], **user_options)

    def _plot_magnetism(self, viewer, **user_options):
        self._raise_error_if_magnetization_not_present()
        self._raise_error_if_color_is_specified(**user_options)
        viewer.show_isosurface(self._raw.charge[1], color="blue", **user_options)
        viewer.show_isosurface(-self._raw.charge[1], color="red", **user_options)

    def _raise_error_if_magnetization_not_present(self):
        if len(self._raw.charge) < 2:
            msg = "Density does not contain magnetization. Please rerun Vasp with ISPIN = 2 to obtain it."
            raise exceptions.NoData(msg)

    def _raise_error_if_color_is_specified(self, **user_options):
        if "color" in user_options:
            msg = "Specifying the color of a magnetic isosurface is not implemented."
            raise exceptions.NotImplemented(msg)

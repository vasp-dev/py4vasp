import numpy as np
import plotly.graph_objects as go
import typing

from py4vasp.data._base import DataBase, RefinementDescriptor
from py4vasp.data._selection import Selection as _Selection
import py4vasp.data._export as _export
import py4vasp._util.convert as _convert
import py4vasp._util.selection as _selection


class Dielectric(DataBase, _export.Image):
    """The dielectric function resulting from electrons and ions.

    You can use this class to extract the dielectric function of a Vasp calculation.
    Vasp evaluates actually evaluates the (symmetric) dielectric tensor, so all
    the returned quantities are 3x3 matrices. For plotting purposes this is reduced
    to the 6 independent variables.

    Parameters
    ----------
    raw_dielectric : RawDielectric
        Dataclass containing the raw data necessary to produce a dielectric function.
    """

    read = RefinementDescriptor("_to_dict")
    to_dict = RefinementDescriptor("_to_dict")
    plot = RefinementDescriptor("_to_plotly")
    to_plotly = RefinementDescriptor("_to_plotly")
    __str__ = RefinementDescriptor("_to_string")

    def _to_string(self):
        energies = self._raw_data.energies
        return f"""
dielectric function:
    energies: [{energies[0]:0.2f}, {energies[-1]:0.2f}] {len(energies)} points
    directions: isotropic, xx, yy, zz, xy, yz, xz
        """.strip()

    def _to_dict(self):
        f"""Read the data into a dictionary.

        Returns
        -------
        dict
            Contains the energies at which the dielectric function was evaluated
            and the dielectric tensor (3x3 matrix) at these energies."""
        return {
            "energies": self._raw_data.energies[:],
            "function": _convert.to_complex(self._raw_data.function[:]),
        }

    def _to_plotly(self, selection=None):
        f"""Read the data and generate a plotly figure.

        Parameters
        ----------
        selection : str
            Specify along which directions and which components of the dielectric
            function you want to plot. Defaults to *isotropic* and both the real
            and the complex part.

        Returns
        -------
        plotly.graph_objects.Figure
            plotly figure containing the dielectric function for the selected
            directions and components."""
        selection = _default_selection_if_none(selection)
        choices = _parse_selection(selection)
        data = self._to_dict()
        plots = [_make_plot(data, *choice) for choice in choices]
        default = {
            "xaxis": {"title": {"text": "Energy (eV)"}},
            "yaxis": {"title": {"text": r"$\epsilon$"}},
        }
        return go.Figure(data=plots, layout=default)


def _default_selection_if_none(selection):
    return "isotropic" if selection is None else selection


class _Choice(typing.NamedTuple):
    direction: str = "isotropic"
    component: str = _Selection.default


def _parse_selection(selection):
    tree = _selection.SelectionTree.from_selection(selection)
    default_choice = _Choice()
    yield from _parse_recursive(tree, default_choice)


def _parse_recursive(tree, current_choice):
    for node in tree.nodes:
        new_choice = _update_choice(current_choice, str(node))
        if len(node.nodes) == 0:
            yield from _setup_component_choices(new_choice)
        else:
            yield from _parse_recursive(node, new_choice)


def _update_choice(current_choice, part):
    if part in ("isotropic", "xx", "yy", "zz", "xy", "xz", "yz"):
        return current_choice._replace(direction=part)
    elif part in ("Re", "real"):
        return current_choice._replace(component="real")
    elif part in ("Im", "imag", "imaginary"):
        return current_choice._replace(component="imag")
    else:
        assert False


def _setup_component_choices(choice):
    if choice.component == _Selection.default:
        yield choice._replace(component="real")
        yield choice._replace(component="imag")
    else:
        yield choice


def _make_plot(data, selection, component):
    x = data["energies"]
    y = getattr(_select_data(data["function"], selection), component)
    subscript = "" if selection == "isotropic" else f"_{{{selection}}}"
    name = f"{component[:2].capitalize()}($\\epsilon{subscript}$)"
    return _scatter_plot(x, y, name)


def _select_data(tensor, selection):
    x, y, z = range(3)
    if selection == "isotropic":
        return np.trace(tensor) / 3
    elif selection == "xx":
        return tensor[x, x]
    elif selection == "yy":
        return tensor[y, y]
    elif selection == "zz":
        return tensor[z, z]
    elif selection == "xy":
        return 0.5 * (tensor[x, y] + tensor[y, x])
    elif selection == "yz":
        return 0.5 * (tensor[y, z] + tensor[z, y])
    elif selection == "xz":
        return 0.5 * (tensor[x, z] + tensor[z, x])


def _scatter_plot(x, y, name):
    return go.Scatter(x=x, y=y, name=name)

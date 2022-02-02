# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import plotly.graph_objects as go
import typing

from py4vasp.data._base import DataBase, RefinementDescriptor
from py4vasp.data._selection import Selection as _Selection
import py4vasp.data._export as _export
import py4vasp._util.convert as _convert
import py4vasp._util.selection as _selection


class DielectricFunction(DataBase, _export.Image):
    """The dielectric function resulting from electrons and ions.

    You can use this class to extract the dielectric function of a Vasp calculation.
    Vasp evaluates actually evaluates the (symmetric) dielectric tensor, so all
    the returned quantities are 3x3 matrices. For plotting purposes this is reduced
    to the 6 independent variables.

    Parameters
    ----------
    raw_dielectric_function : RawDielectricFunction
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
    components: {", ".join(self._components())}
    directions: isotropic, xx, yy, zz, xy, yz, xz
        """.strip()

    def _components(self):
        components = ()
        if self._raw_data.density_density is not None:
            components += ("density",)
        if self._raw_data.current_current is not None:
            components += ("current",)
        if self._raw_data.ion is not None:
            components += ("ion",)
        return components

    def _to_dict(self):
        f"""Read the data into a dictionary.

        Returns
        -------
        dict
            Contains the energies at which the dielectric function was evaluated
            and the dielectric tensor (3x3 matrix) at these energies."""
        data = self._raw_data
        return {
            "energies": data.energies[:],
            "density_density": _convert_to_complex_if_not_none(data.density_density),
            "current_current": _convert_to_complex_if_not_none(data.current_current),
            "ion": _convert_to_complex_if_not_none(data.ion),
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
        data = self._to_dict()
        choices = _parse_selection(selection, data)
        plots = [_make_plot(data, *choice) for choice in choices]
        default = {
            "xaxis": {"title": {"text": "Energy (eV)"}},
            "yaxis": {"title": {"text": "dielectric function ϵ"}},
        }
        return go.Figure(data=plots, layout=default)


def _convert_to_complex_if_not_none(array):
    if array is None:
        return None
    else:
        return _convert.to_complex(array[:])


def _default_selection_if_none(selection):
    return "isotropic" if selection is None else selection


def _parse_selection(selection, data):
    tree = _selection.SelectionTree.from_selection(selection)
    yield from _parse_recursive(tree, _default_choice(data))


class _Choice(typing.NamedTuple):
    component: str
    direction: str = "isotropic"
    real_or_imag: str = _selection.all


def _default_choice(data):
    if data["density_density"] is not None:
        return _Choice("density")
    else:
        return _Choice("ion")


def _parse_recursive(tree, current_choice):
    for node in tree.nodes:
        new_choice = _update_choice(current_choice, str(node))
        if len(node.nodes) == 0:
            yield from _setup_component_choices(new_choice)
        else:
            yield from _parse_recursive(node, new_choice)


def _update_choice(current_choice, part):
    if part in ("density", "current", "ion"):
        return current_choice._replace(component=part)
    elif part in ("isotropic", "xx", "yy", "zz", "xy", "xz", "yz"):
        return current_choice._replace(direction=part)
    elif part in ("Re", "real"):
        return current_choice._replace(real_or_imag="real")
    elif part in ("Im", "imag", "imaginary"):
        return current_choice._replace(real_or_imag="imag")
    else:
        assert False


def _setup_component_choices(choice):
    if choice.real_or_imag == _selection.all:
        yield choice._replace(real_or_imag="real")
        yield choice._replace(real_or_imag="imag")
    else:
        yield choice


def _make_plot(data, *choice):
    x = data["energies"]
    y = _select_data(data, *choice)
    name = _build_name(*choice)
    return _scatter_plot(x, y, name)


def _select_data(data, component, direction, real_or_imag):
    data_component = _select_data_component(data, component)
    data_direction = _select_data_direction(data_component, direction)
    return getattr(data_direction, real_or_imag)


def _select_data_component(data, component):
    if component == "density":
        return data["density_density"]
    elif component == "current":
        return data["current_current"]
    else:
        return data["ion"]


def _select_data_direction(tensor, direction):
    x, y, z = range(3)
    if direction == "isotropic":
        return np.trace(tensor) / 3
    elif direction == "xx":
        return tensor[x, x]
    elif direction == "yy":
        return tensor[y, y]
    elif direction == "zz":
        return tensor[z, z]
    elif direction == "xy":
        return 0.5 * (tensor[x, y] + tensor[y, x])
    elif direction == "yz":
        return 0.5 * (tensor[y, z] + tensor[z, y])
    elif direction == "xz":
        return 0.5 * (tensor[x, z] + tensor[z, x])


def _build_name(component, direction, real_or_imag):
    name = f"{real_or_imag[:2].capitalize()},{component}"
    if direction != "isotropic":
        name += f",{direction}"
    return name


def _scatter_plot(x, y, name):
    return go.Scatter(x=x, y=y, name=name)

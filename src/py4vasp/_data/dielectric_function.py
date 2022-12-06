# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import typing

import numpy as np

from py4vasp._data import base
from py4vasp._third_party import graph
from py4vasp._util import convert, select


class DielectricFunction(base.Refinery, graph.Mixin):
    """The dielectric function resulting from electrons and ions.

    You can use this class to extract the dielectric function of a Vasp calculation.
    Vasp evaluates actually evaluates the (symmetric) dielectric tensor, so all
    the returned quantities are 3x3 matrices. For plotting purposes this is reduced
    to the 6 independent variables.
    """

    @base.data_access
    def __str__(self):
        energies = self._raw_data.energies
        return f"""
dielectric function:
    energies: [{energies[0]:0.2f}, {energies[-1]:0.2f}] {len(energies)} points
{self._components()}    directions: isotropic, xx, yy, zz, xy, yz, xz
        """.strip()

    def _components(self):
        if self._raw_data.current_current.is_none():
            return ""
        else:
            return "    components: density, current\n"

    @base.data_access
    def to_dict(self):
        """Read the data into a dictionary.

        Returns
        -------
        dict
            Contains the energies at which the dielectric function was evaluated
            and the dielectric tensor (3x3 matrix) at these energies."""
        data = convert.to_complex(np.array(self._raw_data.dielectric_function))
        return {
            "energies": self._raw_data.energies[:],
            "dielectric_function": data,
            **self._add_current_current_if_available(),
        }

    @base.data_access
    def to_graph(self, selection=None):
        """Read the data and generate a figure with the selected directions.

        Parameters
        ----------
        selection : str
            Specify along which directions and which components of the dielectric
            function you want to plot. Defaults to *isotropic* and both the real
            and the complex part.

        Returns
        -------
        Graph
            figure containing the dielectric function for the selected
            directions and components."""
        selection = _default_selection_if_none(selection)
        data = self.to_dict()
        choices = _parse_selection(selection, "current_current" in data)
        return graph.Graph(
            series=[_make_plot(data, *choice) for choice in choices],
            xlabel="Energy (eV)",
            ylabel="dielectric function ϵ",
        )

    def _add_current_current_if_available(self):
        if self._raw_data.current_current.is_none():
            return {}
        data = convert.to_complex(np.array(self._raw_data.current_current))
        return {"current_current": data}


def _default_selection_if_none(selection):
    return "isotropic" if selection is None else selection


def _parse_selection(selection, has_current_current):
    tree = select.Tree.from_selection(selection)
    yield from _parse_recursive(tree, _default_choice(has_current_current))


def _default_choice(has_current_current):
    if has_current_current:
        return _Choice("density")
    else:
        return _Choice()


class _Choice(typing.NamedTuple):
    component: str = None
    direction: str = "isotropic"
    real_or_imag: str = select.all


def _parse_recursive(tree, current_choice):
    for node in tree.nodes:
        new_choice = _update_choice(current_choice, str(node))
        if len(node.nodes) == 0:
            yield from _setup_component_choices(new_choice)
        else:
            yield from _parse_recursive(node, new_choice)


def _update_choice(current_choice, part):
    if part in ("current", "density"):
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
    if choice.real_or_imag == select.all:
        yield choice._replace(real_or_imag="real")
        yield choice._replace(real_or_imag="imag")
    else:
        yield choice


def _make_plot(data, *choice):
    return graph.Series(
        x=data["energies"], y=_select_data(data, *choice), name=_build_name(*choice)
    )


def _select_data(data, component, direction, real_or_imag):
    data_component = _select_data_component(data, component)
    data_direction = _select_data_direction(data_component, direction)
    return getattr(data_direction, real_or_imag)


def _select_data_component(data, component):
    if component == "current":
        return data["current_current"]
    else:
        return data["dielectric_function"]


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
    name = real_or_imag[:2].capitalize()
    if component:
        name = f"{component},{name}"
    if direction != "isotropic":
        name += f",{direction}"
    return name

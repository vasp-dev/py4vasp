# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import typing

import numpy as np

from py4vasp._data import base
from py4vasp._third_party import graph
from py4vasp._util import convert, index, select


class DielectricFunction(base.Refinery, graph.Mixin):
    """The dielectric function resulting from electrons and ions.

    You can use this class to extract the dielectric function of a Vasp calculation.
    VASP evaluates actually evaluates the (symmetric) dielectric tensor, so all
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
        if self._has_current_component():
            return "    components: density, current\n"
        else:
            return ""

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

    def _add_current_current_if_available(self):
        if self._has_current_component():
            data = convert.to_complex(np.array(self._raw_data.current_current))
            return {"current_current": data}
        else:
            return {}

    def _has_current_component(self):
        return not self._raw_data.current_current.is_none()

    @base.data_access
    def to_graph(self, selection=None):
        """Read the data and generate a figure with the selected directions.

        Parameters
        ----------
        selection : str
            Specify along which directions and which components of the dielectric
            function you want to plot. Defaults to *isotropic* and both the real
            and the complex part. You can use the `selections` routine if you are
            not sure which options are available.

        Returns
        -------
        Graph
            figure containing the dielectric function for the selected
            directions and components."""
        selection = self._replace_complex_labels(selection or "")
        return graph.Graph(
            series=self._make_series(selection),
            xlabel="Energy (eV)",
            ylabel="dielectric function ϵ",
        )

    @base.data_access
    def selections(self):
        "Returns a dictionary of possible selections for component, direction, and complex value."
        components = (
            ["density", "current"] if self._has_current_component() else ["density"]
        )
        return {
            "components": components,
            "directions": [key for key in self._init_directions_dict() if key],
            "complex": ["real", "Re", "imag", "Im"],
        }

    def _replace_complex_labels(self, selection):
        selection = selection.replace("real", "Re")
        return selection.replace("imaginary", "Im").replace("imag", "Im")

    def _make_series(self, selection):
        energies = self._raw_data.energies[:]
        selector = self._make_selector()
        return [
            graph.Series(energies, selector[selection], selector.label(selection))
            for selection in self._generate_selections(selection)
        ]

    def _make_selector(self):
        maps = {
            3: self._init_complex_dict(),
            0: self._init_components_dict(),
            1: self._init_directions_dict(),
        }
        return index.Selector(maps, self._get_data(), reduction=np.average)

    def _init_components_dict(self):
        return {None: 0, "density": 0, "current": 1}

    def _init_directions_dict(self):
        return {
            None: [0, 4, 8],
            "isotropic": [0, 4, 8],
            "xx": 0,
            "yy": 4,
            "zz": 8,
            "xy": [1, 3],
            "xz": [2, 6],
            "yz": [5, 7],
        }

    def _init_complex_dict(self):
        return {"Re": 0, "Im": 1}

    def _get_data(self):
        *_, number_points, complex_ = self._raw_data.dielectric_function.shape
        if self._has_current_component():
            new_shape = (9, number_points, complex_)
            density = np.reshape(self._raw_data.dielectric_function, new_shape)
            current = np.reshape(self._raw_data.current_current, new_shape)
            return np.array([density, current])
        else:
            new_shape = (1, 9, number_points, complex_)
            return np.reshape(self._raw_data.dielectric_function, new_shape)

    def _generate_selections(self, selection):
        tree = select.Tree.from_selection(selection)
        for selection in tree.selections():
            if not self._component_selected(selection):
                selection = selection + ("density",)
            if self._complex_selected(selection):
                yield selection
            else:
                yield selection + ("Re",)
                yield selection + ("Im",)

    def _component_selected(self, selection):
        if self._has_current_component():
            return select.contains(selection, "density") or select.contains(
                selection, "current"
            )
        else:
            return True

    def _complex_selected(self, selection):
        return select.contains(selection, "Re") or select.contains(selection, "Im")

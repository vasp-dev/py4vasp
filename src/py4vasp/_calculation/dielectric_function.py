# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np

from py4vasp import raw
from py4vasp._calculation.dispatch import (
    _dispatch,
    DataSource,
    merge_default,
    merge_graphs,
    merge_strings,
    quantity,
)
from py4vasp._raw.data_db import DielectricFunction_DB
from py4vasp._third_party import graph
from py4vasp._util import check, convert, index, select


class DielectricFunctionHandler:
    """Handler for the dielectric_function quantity. Works with exactly one raw.DielectricFunction object."""

    def __init__(self, raw_dielectric_function: raw.DielectricFunction):
        self._raw_dielectric_function = raw_dielectric_function

    @classmethod
    def from_data(
        cls, raw_dielectric_function: raw.DielectricFunction
    ) -> "DielectricFunctionHandler":
        return cls(raw_dielectric_function)

    def to_dict(self) -> dict:
        """Read the data into a dictionary.

        Returns
        -------
        dict
            Contains the energies at which the dielectric function was evaluated
            and the dielectric tensor (3x3 matrix) at these energies.
        """
        data = convert.to_complex(
            np.array(self._raw_dielectric_function.dielectric_function)
        )
        return {
            "energies": self._raw_dielectric_function.energies[:],
            "dielectric_function": data,
            **self._add_current_current_if_available(),
            **self._add_q_point_if_available(),
        }

    def to_database(self) -> dict:
        """Serialize dielectric function data for database storage."""
        return {
            "dielectric_function": DielectricFunction_DB(
                energy_min=(
                    float(np.min(self._raw_dielectric_function.energies[:]))
                    if not check.is_none(self._raw_dielectric_function.energies)
                    else None
                ),
                energy_max=(
                    float(np.max(self._raw_dielectric_function.energies[:]))
                    if not check.is_none(self._raw_dielectric_function.energies)
                    else None
                ),
            )
        }

    def to_graph(self, selection=None) -> graph.Graph:
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
            directions and components.
        """
        selection = self._replace_complex_labels(selection or "")
        return graph.Graph(
            series=self._make_series(selection),
            xlabel="Energy (eV)",
            ylabel="dielectric function ϵ",
        )

    def selections(self) -> dict:
        """Returns a dictionary of possible selections for component, direction, and complex value."""
        complex_selections = {"complex": ["real", "Re", "imag", "Im"]}
        if not self._has_tensor_data():
            return complex_selections
        components = (
            ["density", "current"] if self._has_current_component() else ["density"]
        )
        return {
            "components": components,
            "directions": [key for key in self._init_directions_dict() if key],
            **complex_selections,
        }

    def __str__(self) -> str:
        energies = self._raw_dielectric_function.energies
        header = f"""\
dielectric function:
    energies: [{energies[0]:0.2f}, {energies[-1]:0.2f}] {len(energies)} points"""
        if self._has_tensor_data():
            footer = "directions: isotropic, xx, yy, zz, xy, yz, xz"
        else:
            qpoint_label = ", ".join(
                f"{q:0.3f}" for q in self._raw_dielectric_function.q_point
            )
            footer = f"q-point: [{qpoint_label}]"
        if self._has_current_component():
            return f"""\
{header}
    components: density, current
    {footer}"""
        else:
            return f"""\
{header}
    {footer}"""

    def _add_current_current_if_available(self):
        if self._has_current_component():
            data = convert.to_complex(
                np.array(self._raw_dielectric_function.current_current)
            )
            return {"current_current": data}
        else:
            return {}

    def _has_current_component(self):
        return not check.is_none(self._raw_dielectric_function.current_current)

    def _add_q_point_if_available(self):
        if self._has_q_point():
            return {"q_point": self._raw_dielectric_function.q_point[:]}
        else:
            return {}

    def _has_q_point(self):
        return not check.is_none(self._raw_dielectric_function.q_point)

    def _replace_complex_labels(self, selection):
        selection = selection.replace("real", "Re")
        return selection.replace("imaginary", "Im").replace("imag", "Im")

    def _make_series(self, selection):
        energies = self._raw_dielectric_function.energies[:]
        selector = self._make_selector()
        return [
            graph.Series(
                energies, selector[selection], self._create_label(selector, selection)
            )
            for selection in self._generate_selections(selection)
        ]

    def _make_selector(self):
        if self._has_tensor_data():
            maps = {
                3: self._init_complex_dict(),
                0: self._init_components_dict(),
                1: self._init_directions_dict(),
            }
        else:
            maps = {
                1: self._init_complex_dict(),
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
        *_, number_points, complex_ = (
            self._raw_dielectric_function.dielectric_function.shape
        )
        if self._has_current_component():
            new_shape = (9, number_points, complex_)
            density = np.reshape(
                self._raw_dielectric_function.dielectric_function, new_shape
            )
            current = np.reshape(
                self._raw_dielectric_function.current_current, new_shape
            )
            return np.array([density, current])
        elif self._has_tensor_data():
            new_shape = (1, 9, number_points, complex_)
            return np.reshape(
                self._raw_dielectric_function.dielectric_function, new_shape
            )
        else:
            return self._raw_dielectric_function.dielectric_function

    def _create_label(self, selector, selection):
        if self._has_tensor_data():
            return selector.label(selection)
        else:
            q_point_label = ",".join(
                str(convert.Fraction(q)) for q in self._raw_dielectric_function.q_point
            )
            return f"{selector.label(selection)}_q=[{q_point_label}]"

    def _has_tensor_data(self):
        return self._raw_dielectric_function.dielectric_function.ndim == 4

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


@quantity("dielectric_function")
class DielectricFunction(graph.Mixin):
    """The dielectric function describes the material response to an electric field.

    The dielectric function is a fundamental concept that describes how a material
    responds to an external electric field. It is a frequency-dependent complex-valued
    3x3 matrix that relates the polarization of a material to the applied electric
    field. The dielectric function is essential in understanding optical properties,
    such as refractive index and absorption.

    There are many different ways to compute dielectric functions with VASP. This
    class provides a common interface to all of them. You can pass a `selection`
    argument to any of the methods of this class to select which dielectric function
    you are interested in. Please make sure the INCAR file you use is compatible
    with the setup.

    The 3x3 matrix is symmetric so for the plotting routines, py4vasp uses only the
    six distinct components (xx, yy, zz, xy, xz, yz). The default is the isotropic
    dielectric function but you can also select specific components by providing
    one of the six components as selection.
    """

    def __init__(self, source, quantity_name: str = "dielectric_function"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(
        cls, raw_dielectric_function: raw.DielectricFunction
    ) -> "DielectricFunction":
        """Create a DielectricFunction dispatcher from raw data (convenience for testing)."""
        return cls(source=DataSource(raw_dielectric_function))

    @property
    def path(self):
        """Returns the path from which the output is obtained."""
        return self._path

    def _handler_factory(self, raw_data):
        return DielectricFunctionHandler.from_data(raw_data)

    def read(self, selection: str | None = None) -> dict:
        """Read the data into a dictionary.

        Returns
        -------
        dict
            Contains the energies at which the dielectric function was evaluated
            and the dielectric tensor (3x3 matrix) at these energies.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            DielectricFunctionHandler.to_dict,
        )

    def to_dict(self, selection: str | None = None) -> dict:
        """Public alias for read(). Check that method for examples and optional arguments."""
        return self.read(selection=selection)

    def to_graph(self, selection: str | None = None) -> graph.Graph:
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
            directions and components.
        """
        return merge_graphs(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            DielectricFunctionHandler.to_graph,
        )

    def selections(self, selection: str | None = None) -> dict:
        """Returns a dictionary of possible selections for component, direction, and complex value."""
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            DielectricFunctionHandler.selections,
        )

    def __str__(self, selection: str | None = None) -> str:
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            DielectricFunctionHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def _to_database(self, selection=None) -> dict:
        """Return {selection_name: handler_result_dict} for database storage."""
        return _dispatch(
            self._source,
            self._quantity_name,
            selection,
            DielectricFunctionHandler.from_data,
            DielectricFunctionHandler.to_database,
        )

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np

from py4vasp import raw
from py4vasp._calculation.dispatch import (
    DataSource,
    _dispatch,
    merge_default,
    merge_graphs,
    merge_strings,
    merge_to_database,
    quantity,
)
from py4vasp._raw.definition import unique_selections as _schema_sources
from py4vasp._raw.models import DielectricFunctionModel
from py4vasp._third_party import graph
from py4vasp._util import check, convert, index, select

# The name of this quantity in the schema and in the dictionary selections returns, so
# that the key naming the sources matches the attribute the user reaches them through.
_DATA_QUANTITY = "dielectric_function"


class DielectricFunctionHandler:
    """Handler for the dielectric_function quantity. Works with exactly one raw.DielectricFunction object."""

    def __init__(self, raw_dielectric_function: raw.DielectricFunction):
        self._raw_dielectric_function = raw_dielectric_function

    @classmethod
    def from_data(
        cls, raw_dielectric_function: raw.DielectricFunction
    ) -> "DielectricFunctionHandler":
        return cls(raw_dielectric_function)

    def to_dict(self, selection=None) -> dict:
        """Read the data into a dictionary.

        Without a selection the whole 3x3 tensor is returned. A selection reduces it to
        the named directions, so that the isotropic average does not have to be taken by
        hand at the call site.

        Returns
        -------
        dict
            Contains the energies at which the dielectric function was evaluated and
            either the dielectric tensor (3x3 matrix) or one complex spectrum per
            selected direction.
        """
        if not selection:
            return self._whole_tensor()
        return {
            "energies": self._energies(),
            **dict(self._selected_spectra(selection)),
        }

    def _whole_tensor(self) -> dict:
        data = convert.to_complex(
            np.array(self._raw_dielectric_function.dielectric_function)
        )
        return {
            "energies": self._energies(),
            "dielectric_function": data,
            **self._add_current_current_if_available(),
            **self._add_q_point_if_available(),
        }

    def _selected_spectra(self, selection):
        selector = self._make_selector()
        tree = select.Tree.from_selection(self._replace_complex_labels(selection))
        for choice in tree.selections():
            yield selector.label(choice), self._complex_spectrum(selector, choice)

    def _complex_spectrum(self, selector, choice):
        # the selector offers Re and Im as one more axis to choose from, so a spectrum
        # that leaves them unselected is averaged over the two and loses its phase
        real = np.array(selector[choice + ("Re",)])
        imaginary = np.array(selector[choice + ("Im",)])
        return real + 1j * imaginary

    def _energies(self):
        return self._raw_dielectric_function.energies[:]

    def to_database(self) -> dict:
        """Serialize dielectric function data for database storage."""
        return DielectricFunctionModel(
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
        """Returns the sources, components, directions, and complex values to select from."""
        # The sources come first because they are the selection read takes; without them
        # this method cannot answer what a user asks it, which is what to pass where.
        sources = {_DATA_QUANTITY: list(_schema_sources(_DATA_QUANTITY))}
        complex_selections = {"complex": ["real", "Re", "imag", "Im"]}
        if not self._has_tensor_data():
            return {**sources, **complex_selections}
        components = (
            ["density", "current"] if self._has_current_component() else ["density"]
        )
        return {
            **sources,
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
        energies = self._energies()
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

    There are many different ways to compute dielectric functions with VASP. This class
    provides a common interface to all of them. Every method takes a *selection*, and
    that selection does two different things. It chooses **which** dielectric function
    to use -- ``selections()["dielectric_function"]`` lists them, and your INCAR file
    decides which of them your calculation actually contains. It also chooses **which
    part** of the 3x3 tensor you want. The matrix is symmetric, so there are six
    distinct components (xx, yy, zz, xy, xz, yz) besides their average, *isotropic*.

    Reading without a direction gives you the complete tensor, so you can do your own
    algebra with it. Reading with one gives you a single complex spectrum, so that the
    isotropic average does not have to be assembled by hand at the call site. Plotting
    defaults to *isotropic* and draws the real and the imaginary part.

    Examples
    --------
    First, we create some example data so that you can follow along. Please define a
    variable `path` with the path to a directory that does not exist yet. Alternatively,
    use your own data if you have run VASP.

    >>> from py4vasp import demo
    >>> calculation = demo.calculation(path)

    The `selections` routine reports both kinds of selection at once

    >>> calculation.dielectric_function.selections()
    {'dielectric_function': [...], 'components': ['density', 'current'],
     'directions': ['isotropic', 'xx', 'yy', 'zz', 'xy', 'xz', 'yz'],
     'complex': ['real', 'Re', 'imag', 'Im']}

    A summary of the mesh the dielectric function is evaluated on is printed by

    >>> print(calculation.dielectric_function)
    dielectric function:
        energies: [0.00, 12.00] 301 points
        components: density, current
        directions: isotropic, xx, yy, zz, xy, yz, xz
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

    def _handler_factory(self, raw_data):
        return DielectricFunctionHandler.from_data(raw_data)

    def read(self, selection: str | None = None) -> dict:
        """Read the data into a dictionary.

        Parameters
        ----------
        selection : str
            Choose which dielectric function VASP computed and, for one with tensor
            data, which directions of it to reduce to. Without a direction you get the
            whole 3x3 tensor. Use :py:meth:`selections` to see both lists.

        Returns
        -------
        dict
            Contains the energies at which the dielectric function was evaluated and
            either the dielectric tensor (3x3 matrix) at these energies or, if you
            selected directions, one complex spectrum per direction.

        Examples
        --------
        First, we create some example data so that you can follow along. Please define a
        variable `path` with the path to a directory that does not exist yet.
        Alternatively, use your own data if you have run VASP.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        Without a selection you obtain the complete tensor

        >>> dielectric = calculation.dielectric_function.read()
        >>> dielectric["dielectric_function"].shape
        (3, 3, 301)

        Select a direction to reduce that tensor to a single complex spectrum

        >>> isotropic = calculation.dielectric_function.read("isotropic")
        >>> sorted(isotropic)
        ['energies', 'isotropic']

        The isotropic average is the mean of the diagonal, so you do not have to take
        it yourself

        >>> import numpy as np
        >>> tensor = dielectric["dielectric_function"]
        >>> bool(np.allclose(isotropic["isotropic"], np.trace(tensor) / 3))
        True

        Ask for several directions at once and each comes back under its own key

        >>> sorted(calculation.dielectric_function.read("xx, yy, zz"))
        ['energies', 'xx', 'yy', 'zz']
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

        Examples
        --------
        First, we create some example data so that you can follow along. Please define a
        variable `path` with the path to a directory that does not exist yet.
        Alternatively, use your own data if you have run VASP.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        Without a selection the isotropic average is drawn, real and imaginary part

        >>> calculation.dielectric_function.to_graph()
        Graph(series=[Series(..., label='Re_density', ...), Series(..., label='Im_density', ...)],
              xlabel='Energy (eV)', ...)

        Select directions to compare the components of the tensor with each other

        >>> calculation.dielectric_function.to_graph("xx, zz")
        Graph(series=[Series(..., label='Re_density_xx', ...), Series(..., label='Im_density_xx', ...),
              Series(..., label='Re_density_zz', ...), Series(..., label='Im_density_zz', ...)], ...)

        Restrict the plot to one part of the complex spectrum

        >>> calculation.dielectric_function.to_graph("imag")
        Graph(series=[Series(..., label='Im_density', ...)], ...)
        """
        return merge_graphs(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            DielectricFunctionHandler.to_graph,
        )

    def selections(self, selection: str | None = None) -> dict:
        """Return the sources, components, directions, and complex values to select from.

        The ``dielectric_function`` entry lists the sources, which is which dielectric
        function VASP computed. The remaining entries are the parts of the tensor; both
        :py:meth:`read` and :py:meth:`to_graph` accept them.

        Examples
        --------
        First, we create some example data so that you can follow along. Please define a
        variable `path` with the path to a directory that does not exist yet.
        Alternatively, use your own data if you have run VASP.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        >>> selections = calculation.dielectric_function.selections()
        >>> selections["directions"]
        ['isotropic', 'xx', 'yy', 'zz', 'xy', 'xz', 'yz']

        A dielectric function evaluated at finite **q** is a scalar, so it reports no
        directions at all and asking for one raises an error.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            DielectricFunctionHandler.selections,
        )

    def print(self, selection: str | None = None) -> None:
        """Print a string representation of this quantity.

        Parameters
        ----------
        selection : str | None
            Select which source of the quantity is printed. If you select multiple
            sources, py4vasp prints one block per source.
        """
        print(self.__str__(selection))

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

    def _to_database(self) -> dict:
        """Return {quantity[_selection]: handler_result} for database storage."""
        return merge_to_database(
            self._source,
            self._quantity_name,
            DielectricFunctionHandler.from_data,
            DielectricFunctionHandler.to_database,
        )

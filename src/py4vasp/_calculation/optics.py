# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Optical properties (transmission, absorption, reflectivity, color) derived from
the dielectric function that VASP computes."""

import numpy as np

from py4vasp import exception, raw
from py4vasp._calculation import _optics_color
from py4vasp._calculation.dispatch import (
    DataSource,
    merge_default,
    merge_graphs,
    merge_strings,
    merge_to_database,
    quantity,
)
from py4vasp._raw.models import OpticsModel
from py4vasp._raw.definition import unique_selections as _schema_sources
from py4vasp._third_party import graph
from py4vasp._util import convert, index, select
from py4vasp._util.color import Color

# The optics quantity owns no raw data of its own; it derives every quantity from the
# dielectric function. Dispatch therefore accesses the "dielectric_function" schema
# entry, so a selection picks both the dielectric function source and the direction.
_DATA_QUANTITY = "dielectric_function"

# Planck constant times the speed of light, converts photon energy to wavelength via
# λ[nm] = HBAR_C / E[eV].
HBAR_C = 1239.84  # eV·nm


def _reflectivity(epsilon):
    """Fresnel reflectivity at normal incidence R = |(n - 1) / (n + 1)|²."""
    n = np.sqrt(epsilon)
    return np.abs((n - 1) / (n + 1)) ** 2


def _absorption(epsilon, energies):
    """Absorption coefficient α = 2kE/ħc, normalized to its maximum (see docstring)."""
    extinction = np.sqrt(epsilon).imag
    absorption = 2 * extinction * energies / HBAR_C
    return absorption / np.max(absorption)


def _transmission(epsilon, energies):
    """Transmission estimate T = clip(1 - R - A, 0, 1) (energy conservation)."""
    reflectivity = _reflectivity(epsilon)
    absorption = _absorption(epsilon, energies)
    return np.clip(1 - reflectivity - absorption, 0, 1)


# Maps each optical coefficient to a callable(epsilon, energies) -> spectrum.
_COEFFICIENTS = {
    "transmission": _transmission,
    "absorption": _absorption,
    "reflectivity": lambda epsilon, energies: _reflectivity(epsilon),
}


class OpticsHandler:
    """Transforms a single raw.DielectricFunction into optical properties."""

    def __init__(self, raw_dielectric_function: raw.DielectricFunction):
        self._raw_dielectric_function = raw_dielectric_function

    @classmethod
    def from_data(
        cls, raw_dielectric_function: raw.DielectricFunction
    ) -> "OpticsHandler":
        return cls(raw_dielectric_function)

    def to_dict(self, selection=None) -> dict:
        """Read transmission, absorption, and reflectivity into a dictionary.

        Returns
        -------
        dict
            Contains the energies and the transmission, absorption, and reflectivity
            spectra for the selected direction (default: isotropic). If the selection
            resolves to multiple directions, the spectra are nested under each direction
            label.
        """
        energies = self._energies()
        results = {
            label: {
                "transmission": _transmission(epsilon, energies),
                "absorption": _absorption(epsilon, energies),
                "reflectivity": _reflectivity(epsilon),
            }
            for label, epsilon in self._dielectric_function(selection)
        }
        if len(results) == 1:
            return {"energies": energies, **next(iter(results.values()))}
        return {"energies": energies, **results}

    def to_graph(self, selection=None, default_components=None) -> graph.Graph:
        """Plot the selected optical coefficients.

        The selection chooses the coefficient (``transmission``, ``absorption``,
        ``reflectivity``) in addition to the direction. If no coefficient is part of the
        selection, *default_components* is used (all three by default). The y label is
        the coefficient name when only a single coefficient is shown and "coefficient"
        otherwise.
        """
        energies = self._energies()
        series = []
        shown = set()
        components = self._components(selection, default_components)
        for component, label, epsilon in components:
            shown.add(component)
            spectrum = _COEFFICIENTS[component](epsilon, energies)
            series.append(
                graph.Series(energies, spectrum, self._label(component, label))
            )
        ylabel = shown.pop() if len(shown) == 1 else "coefficient"
        return graph.Graph(series=series, xlabel="Energy (eV)", ylabel=ylabel)

    def transmission_graph(self, selection=None) -> graph.Graph:
        return self.to_graph(selection, default_components=["transmission"])

    def absorption_graph(self, selection=None) -> graph.Graph:
        return self.to_graph(selection, default_components=["absorption"])

    def reflectivity_graph(self, selection=None) -> graph.Graph:
        return self.to_graph(selection, default_components=["reflectivity"])

    def color(self, selection=None, *, illuminant="D65", cmf="1931_2"):
        """Perceived sRGB color(s) for the selected coefficient(s) and direction(s).

        Uses the same selection logic as :meth:`to_graph`, but derives the color from a
        single coefficient defaulting to the reflectivity. Each color is returned as a
        :class:`~py4vasp._util.color.Color`. Selecting several coefficients or directions
        returns a dictionary of colors keyed by the selection.
        """
        energies = self._energies()
        results = {}
        components = self._components(selection, ["reflectivity"])
        for component, direction_label, epsilon in components:
            spectrum = _COEFFICIENTS[component](epsilon, energies)
            rgb = self._rgb(spectrum, energies, illuminant, cmf)
            label = self._label(component, direction_label)
            results[label] = Color(rgb, label=label)
        if len(results) == 1:
            return next(iter(results.values()))
        return results

    def to_database(self) -> OpticsModel:
        """Serialize scalar optical properties for database storage.

        Extracts the energy range and the range of each coefficient for the isotropic
        average, along with the perceived color (from the isotropic reflectivity under
        the default illuminant and observer) as sRGB and HEX.
        """
        data = self.to_dict()
        color = self.color()
        return OpticsModel(
            energy_min=float(np.min(data["energies"])),
            energy_max=float(np.max(data["energies"])),
            reflectivity_min=float(np.min(data["reflectivity"])),
            reflectivity_max=float(np.max(data["reflectivity"])),
            absorption_min=float(np.min(data["absorption"])),
            absorption_max=float(np.max(data["absorption"])),
            transmission_min=float(np.min(data["transmission"])),
            transmission_max=float(np.max(data["transmission"])),
            color_rgb=[float(channel) for channel in color.rgb],
            color_hex=color.hex,
        )

    def _components(self, selection, default_components):
        """Yield ``(component, direction_label, epsilon)`` for each selected combination.

        The coefficient tokens are filtered out of the selection to obtain the direction,
        which the Selector reduces on the complex dielectric function. A coefficient
        present in a branch overrides *default_components* for that branch.
        """
        default_components = default_components or list(_COEFFICIENTS)
        selector = self._make_selector()
        tree = select.Tree.from_selection(selection or "")
        branches = list(tree.selections())
        directions = list(tree.selections(filter=set(_COEFFICIENTS)))
        for branch, direction in zip(branches, directions):
            components = [c for c in _COEFFICIENTS if select.contains(branch, c)]
            epsilon = convert.to_complex(np.ascontiguousarray(selector[direction]))
            label = selector.label(direction) or "isotropic"
            for component in components or default_components:
                yield component, label, epsilon

    def _rgb(self, spectrum, energies, illuminant, cmf):
        # Convert photon energies to wavelengths and sort them in ascending order as
        # expected by the color-matching routine.
        mask = energies > 0
        wavelengths = HBAR_C / energies[mask]
        order = np.argsort(wavelengths)
        return _optics_color.spectrum_to_rgb(
            wavelengths[order],
            spectrum[mask][order],
            illuminant=illuminant,
            cmf=cmf,
        )

    def _label(self, name, direction):
        return name if direction == "isotropic" else f"{name}_{direction}"

    def selections(self) -> dict:
        """Returns the sources, components, and directions that can be selected."""
        return {
            "optics": list(_schema_sources(_DATA_QUANTITY)),
            "components": list(_COEFFICIENTS),
            "directions": [key for key in self._init_directions_dict() if key],
        }

    def __str__(self) -> str:
        energies = self._raw_dielectric_function.energies
        directions = ", ".join(key for key in self._init_directions_dict() if key)
        return f"""\
optics:
    energies: [{energies[0]:0.2f}, {energies[-1]:0.2f}] {len(energies)} points
    directions: {directions}"""

    def _energies(self):
        return self._raw_dielectric_function.energies[:]

    def _dielectric_function(self, selection):
        """Complex dielectric function ε along each selected direction.

        Yields ``(label, epsilon)`` where *epsilon* is the complex ε spectrum for the
        selected direction (default: isotropic average of the diagonal). The selection
        is routed through the standard ``select.Tree``/``index.Selector`` machinery so
        that combinations (``"xx + yy"``), lists (``"xx, yy"``) and the named directions
        behave exactly as for the dielectric function itself.
        """
        selector = self._make_selector()
        tree = select.Tree.from_selection(selection or "")
        for direction in tree.selections(filter=set(_COEFFICIENTS)):
            epsilon = convert.to_complex(np.ascontiguousarray(selector[direction]))
            label = selector.label(direction) or "isotropic"
            yield label, epsilon

    def _make_selector(self):
        maps = {0: self._init_directions_dict()}
        return index.Selector(maps, self._get_data(), reduction=np.average)

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

    def _get_data(self):
        if not self._has_tensor_data():
            message = (
                "Optical properties require a dielectric function with directional "
                "(3x3 tensor) data, but the selected dielectric function is a scalar."
            )
            raise exception.IncorrectUsage(message)
        *_, number_points, complex_ = (
            self._raw_dielectric_function.dielectric_function.shape
        )
        new_shape = (9, number_points, complex_)
        return np.reshape(
            np.array(self._raw_dielectric_function.dielectric_function), new_shape
        )

    def _has_tensor_data(self):
        return self._raw_dielectric_function.dielectric_function.ndim == 4


@quantity("optics")
class Optics(graph.Mixin):
    """Optical properties of a material derived from its dielectric function.

    From the complex dielectric function :math:`\\varepsilon(\\omega)` that VASP computes,
    this quantity derives the reflectivity :math:`R = |(\\sqrt\\varepsilon - 1) /
    (\\sqrt\\varepsilon + 1)|^2` at normal incidence, the absorption, and the transmission,
    as well as the perceived color of the material. The absorption is normalized to its
    maximum and the transmission is estimated as :math:`T = 1 - R - A`, so it is a rough
    indicator rather than a Beer--Lambert transmission through a sample of a given
    thickness.

    Every method accepts a ``selection`` string using the common py4vasp grammar. A
    selection may combine three kinds of tokens:

    * the dielectric function to use (e.g. ``bse``, ``ipa``, ``dft``); the available ones
      depend on your calculation,
    * the coefficient to evaluate (``transmission``, ``absorption``, ``reflectivity``),
    * the direction (``isotropic`` (default), ``xx``, ``yy``, ``zz``, ``xy``, ``xz``,
      ``yz``).

    You can nest tokens (``bse(reflectivity(xx))``), list them with a comma to obtain
    independent results (``xx, yy``), or add them to combine the dielectric function
    before deriving the coefficient (``xx + yy``). Use the :meth:`selections` routine if
    you are unsure which options are available.

    Examples
    --------
    First, we create some example data so that we can illustrate how to use this class.
    You can also use your own VASP calculation data if you have it available.

    >>> from py4vasp import demo
    >>> calculation = demo.calculation(path)

    The `selections` routine reports which dielectric functions, coefficients, and
    directions you can select.

    >>> calculation.optics.selections()
    {'optics': [...], 'components': ['transmission', 'absorption', 'reflectivity'],
        'directions': ['isotropic', 'xx', 'yy', 'zz', 'xy', 'xz', 'yz']}
    """

    def __init__(self, source, quantity_name: str = "optics"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_dielectric_function: raw.DielectricFunction) -> "Optics":
        """Create an Optics dispatcher from raw data (convenience for testing)."""
        return cls(source=DataSource(raw_dielectric_function))

    @property
    def path(self):
        """Returns the path from which the output is obtained."""
        return self._path

    def _handler_factory(self, raw_data):
        return OpticsHandler.from_data(raw_data)

    def read(self, selection: str | None = None) -> dict:
        """Read transmission, absorption, and reflectivity into a dictionary.

        Parameters
        ----------
        selection : str
            Select which dielectric function and which direction(s) to evaluate.
            Defaults to the isotropic average. Use the `selections` routine to discover
            the available options.

        Returns
        -------
        dict
            Contains the energies and the transmission, absorption, and reflectivity
            spectra for the selected direction(s).

        Examples
        --------
        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        Read the spectra of the isotropic average into a dictionary

        >>> calculation.optics.read()
        {'energies': array([...]), 'transmission': array([...]),
            'absorption': array([...]), 'reflectivity': array([...])}

        Select a specific direction instead of the isotropic average

        >>> calculation.optics.read("xx")
        {'energies': array([...]), 'transmission': array([...]),
            'absorption': array([...]), 'reflectivity': array([...])}
        """
        return merge_default(
            self._source,
            _DATA_QUANTITY,
            selection,
            self._handler_factory,
            OpticsHandler.to_dict,
        )

    def to_dict(self, selection: str | None = None) -> dict:
        """Public alias for read(). See that method for the optional arguments."""
        return self.read(selection=selection)

    def reflectivity(self, selection: str | None = None) -> graph.Graph:
        """Plot the reflectivity spectrum. See :meth:`transmission` for the arguments."""
        return merge_graphs(
            self._source,
            _DATA_QUANTITY,
            selection,
            self._handler_factory,
            OpticsHandler.reflectivity_graph,
        )

    def absorption(self, selection: str | None = None) -> graph.Graph:
        """Plot the absorption spectrum. See :meth:`transmission` for the arguments.

        The absorption is normalized to its maximum along the selected direction.
        """
        return merge_graphs(
            self._source,
            _DATA_QUANTITY,
            selection,
            self._handler_factory,
            OpticsHandler.absorption_graph,
        )

    def transmission(self, selection: str | None = None) -> graph.Graph:
        """Plot the transmission spectrum for the selected direction(s).

        Parameters
        ----------
        selection : str
            Select which dielectric function and which direction(s) to evaluate.
            Defaults to the isotropic average.

        Returns
        -------
        Graph
            A figure of the transmission for the selected direction(s).

        Examples
        --------
        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        Plot the transmission of the isotropic average

        >>> graph = calculation.optics.transmission()
        >>> [series.label for series in graph.series]
        ['transmission']

        Compare the transmission along two Cartesian directions

        >>> graph = calculation.optics.transmission("xx, zz")
        >>> [series.label for series in graph.series]
        ['transmission_xx', 'transmission_zz']
        """
        return merge_graphs(
            self._source,
            _DATA_QUANTITY,
            selection,
            self._handler_factory,
            OpticsHandler.transmission_graph,
        )

    def color(
        self,
        selection: str | None = None,
        *,
        illuminant: str = "D65",
        cmf: str = "1931_2",
    ):
        """Compute the perceived sRGB color of the material.

        The color is obtained by lighting a spectrum of the material with a standard
        *illuminant* and integrating against the CIE color matching function *cmf*.
        The selection chooses the coefficient (defaulting to the reflectivity) and the
        direction, using the same grammar as :meth:`plot`.

        Parameters
        ----------
        selection : str
            Select the dielectric function, the coefficient (``reflectivity``,
            ``transmission``, ``absorption``), and the direction(s) to evaluate.
            Defaults to the reflectivity of the isotropic average.
        illuminant : str
            Standard illuminant used to light the material (default "D65"). Use
            :func:`py4vasp._calculation._optics_color.list_illuminants` for the options.
        cmf : str
            CIE color matching function / standard observer (default "1931_2").

        Returns
        -------
        Color
            A :class:`~py4vasp._util.color.Color` that renders as a labeled swatch in
            Jupyter and exposes its HEX and sRGB values. If the selection resolves to
            multiple coefficients or directions, a dictionary of colors is returned.

        Examples
        --------
        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        Compute the perceived color from the reflectivity under the default D65 daylight

        >>> color = calculation.optics.color()
        >>> color.label()
        'reflectivity'

        Derive the color from the transmission and light it with an incandescent lamp

        >>> color = calculation.optics.color("transmission", illuminant="A")
        >>> color.label()
        'transmission'
        """
        return merge_default(
            self._source,
            _DATA_QUANTITY,
            selection,
            self._handler_factory,
            OpticsHandler.color,
            illuminant=illuminant,
            cmf=cmf,
        )

    def to_graph(self, selection: str | None = None) -> graph.Graph:
        """Merge the transmission, absorption, and reflectivity into a single figure.

        This is the routine behind :meth:`plot`. Restricting the selection to a single
        coefficient yields the same figure as the dedicated method, e.g.
        ``plot("transmission")`` is equivalent to ``transmission()``.

        Parameters
        ----------
        selection : str
            Select which dielectric function, coefficient(s), and direction(s) to
            evaluate. Defaults to all three coefficients of the isotropic average.

        Returns
        -------
        Graph
            A figure overlaying the selected coefficients for the selected direction(s).

        Examples
        --------
        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        Overlay the transmission, absorption, and reflectivity

        >>> graph = calculation.optics.plot()
        >>> [series.label for series in graph.series]
        ['transmission', 'absorption', 'reflectivity']

        Restrict the figure to selected coefficients

        >>> graph = calculation.optics.plot("absorption, reflectivity")
        >>> [series.label for series in graph.series]
        ['absorption', 'reflectivity']
        """
        return merge_graphs(
            self._source,
            _DATA_QUANTITY,
            selection,
            self._handler_factory,
            OpticsHandler.to_graph,
        )

    def selections(self, selection: str | None = None) -> dict:
        """Return the dielectric functions, coefficients, and directions to select from."""
        return merge_default(
            self._source,
            _DATA_QUANTITY,
            selection,
            self._handler_factory,
            OpticsHandler.selections,
        )

    def __str__(self, selection: str | None = None) -> str:
        return merge_strings(
            self._source,
            _DATA_QUANTITY,
            selection,
            self._handler_factory,
            OpticsHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def _to_database(self) -> dict:
        """Return {optics[_selection]: handler_result} for database storage.

        The optics quantity derives from the dielectric function, so data is looked up
        under "dielectric_function" but stored under the "optics" key.
        """
        return merge_to_database(
            self._source,
            _DATA_QUANTITY,
            OpticsHandler.from_data,
            OpticsHandler.to_database,
            key_name="optics",
        )

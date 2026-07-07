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
    quantity,
)
from py4vasp._raw.definition import unique_selections as _schema_sources
from py4vasp._third_party import graph
from py4vasp._util import convert, index, select

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

    def reflectivity_graph(self, selection=None) -> graph.Graph:
        return self._coefficient_graph("reflectivity", selection)

    def absorption_graph(self, selection=None) -> graph.Graph:
        return self._coefficient_graph("absorption", selection)

    def transmission_graph(self, selection=None) -> graph.Graph:
        return self._coefficient_graph("transmission", selection)

    def color(
        self, selection=None, *, illuminant="D65", cmf="1931_2", spectrum="reflectivity"
    ):
        """Perceived sRGB color(s) of the material for the selected direction(s)."""
        coefficient = self._color_spectrum(spectrum)
        energies = self._energies()
        results = {
            label: self._rgb(coefficient(epsilon, energies), energies, illuminant, cmf)
            for label, epsilon in self._dielectric_function(selection)
        }
        if len(results) == 1:
            return next(iter(results.values()))
        return results

    def _color_spectrum(self, spectrum):
        if spectrum not in ("reflectivity", "transmission"):
            message = (
                f'The color cannot be derived from "{spectrum}". Please select either '
                '"reflectivity" (default) or "transmission".'
            )
            raise exception.IncorrectUsage(message)
        return _COEFFICIENTS[spectrum]

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

    def to_graph(self, selection=None) -> graph.Graph:
        """Merge transmission, absorption, and reflectivity into a single figure."""
        energies = self._energies()
        series = [
            graph.Series(
                energies, coefficient(epsilon, energies), self._label(name, label)
            )
            for label, epsilon in self._dielectric_function(selection)
            for name, coefficient in _COEFFICIENTS.items()
        ]
        return graph.Graph(series=series, xlabel="Energy (eV)", ylabel="coefficient")

    def _coefficient_graph(self, name, selection) -> graph.Graph:
        energies = self._energies()
        coefficient = _COEFFICIENTS[name]
        series = [
            graph.Series(
                energies, coefficient(epsilon, energies), self._label(name, label)
            )
            for label, epsilon in self._dielectric_function(selection)
        ]
        return graph.Graph(series=series, xlabel="Energy (eV)", ylabel=name)

    def _label(self, name, direction):
        return name if direction == "isotropic" else f"{name}_{direction}"

    def selections(self) -> dict:
        """Returns the dielectric function sources and directions that can be selected."""
        return {
            "optics": list(_schema_sources(_DATA_QUANTITY)),
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
        for sel in select.Tree.from_selection(selection or "").selections():
            epsilon = convert.to_complex(np.ascontiguousarray(selector[sel]))
            label = selector.label(sel) or "isotropic"
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

    From the complex dielectric function VASP computes, this quantity derives the
    transmission, absorption, and reflectivity spectra as well as the perceived RGB
    color of the material. Pass a ``selection`` to any method to choose which
    dielectric function (e.g. ``bse``) and which direction (e.g. ``xx``, ``isotropic``)
    you are interested in.
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
        """Plot the reflectivity spectrum for the selected direction(s)."""
        return merge_graphs(
            self._source,
            _DATA_QUANTITY,
            selection,
            self._handler_factory,
            OpticsHandler.reflectivity_graph,
        )

    def absorption(self, selection: str | None = None) -> graph.Graph:
        """Plot the (max-normalized) absorption spectrum for the selected direction(s)."""
        return merge_graphs(
            self._source,
            _DATA_QUANTITY,
            selection,
            self._handler_factory,
            OpticsHandler.absorption_graph,
        )

    def transmission(self, selection: str | None = None) -> graph.Graph:
        """Plot the transmission spectrum for the selected direction(s)."""
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
        spectrum: str = "reflectivity",
    ):
        """Compute the perceived sRGB color of the material.

        The color is obtained by lighting the material's reflectivity (or transmission)
        spectrum with a standard *illuminant* and integrating against the CIE color
        matching function *cmf*.

        Parameters
        ----------
        selection : str
            Select which dielectric function and which direction(s) to evaluate.
            Defaults to the isotropic average.
        illuminant : str
            Standard illuminant used to light the material (default "D65"). Use
            :func:`py4vasp._calculation._optics_color.list_illuminants` for the options.
        cmf : str
            CIE color matching function / standard observer (default "1931_2").
        spectrum : str
            Whether to derive the color from "reflectivity" (default) or "transmission".

        Returns
        -------
        np.ndarray
            The sRGB color as a length-3 array in [0, 1]. If the selection resolves to
            multiple directions, a dictionary of colors keyed by direction is returned.
        """
        return merge_default(
            self._source,
            _DATA_QUANTITY,
            selection,
            self._handler_factory,
            OpticsHandler.color,
            illuminant=illuminant,
            cmf=cmf,
            spectrum=spectrum,
        )

    def to_graph(self, selection: str | None = None) -> graph.Graph:
        """Merge the transmission, absorption, and reflectivity into a single figure.

        Parameters
        ----------
        selection : str
            Select which dielectric function and which direction(s) to evaluate.
            Defaults to the isotropic average.

        Returns
        -------
        Graph
            A figure overlaying the transmission, absorption, and reflectivity spectra
            for the selected direction(s).
        """
        return merge_graphs(
            self._source,
            _DATA_QUANTITY,
            selection,
            self._handler_factory,
            OpticsHandler.to_graph,
        )

    def selections(self, selection: str | None = None) -> dict:
        """Returns a dictionary of the directions along which optics can be evaluated."""
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

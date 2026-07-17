# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import copy
import itertools
import typing

import numpy as np

from py4vasp import exception, raw
from py4vasp._calculation import slice_
from py4vasp._calculation.dispatch import (
    DataSource,
    merge_default,
    merge_graphs,
    merge_strings,
    merge_to_database,
    quantity,
)
from py4vasp._raw.data_db import Bandgap_DB
from py4vasp._third_party import graph
from py4vasp._util import convert, documentation, select


class _Gap(typing.NamedTuple):
    bottom: str
    top: str


GAPS = {
    "fundamental": _Gap("valence band maximum", "conduction band minimum"),
    "direct": _Gap("direct gap bottom", "direct gap top"),
}

COMPONENTS = ("independent", "up", "down")


class BandgapHandler:
    """Handler for the bandgap quantity. Works with exactly one raw.Bandgap object."""

    def __init__(self, raw_bandgap: raw.Bandgap, steps=None):
        self._raw_bandgap = raw_bandgap
        self._steps = steps

    @classmethod
    def from_data(cls, raw_bandgap: raw.Bandgap, steps=None) -> "BandgapHandler":
        return cls(raw_bandgap, steps=steps)

    def to_dict(self) -> dict:
        return {
            **self._gap_dict("fundamental"),
            **self._band_dict("fundamental"),
            **self._kpoint_dict("VBM"),
            **self._kpoint_dict("CBM"),
            **self._gap_dict("direct"),
            **self._band_dict("direct"),
            **self._kpoint_dict("direct"),
            "fermi_energy": self._get("Fermi energy", component=0),
        }

    def fundamental(self) -> np.ndarray:
        """Return the fundamental bandgap."""
        return self._gap("fundamental", component=0)

    def direct(self) -> np.ndarray:
        """Return the direct bandgap."""
        return self._gap("direct", component=0)

    def valence_band_maximum(self) -> np.ndarray:
        """Return the valence band maximum."""
        return self._get(GAPS["fundamental"].bottom, component=0)

    def conduction_band_minimum(self) -> np.ndarray:
        """Return the conduction band minimum."""
        return self._get(GAPS["fundamental"].top, component=0)

    def to_graph(self, selection="fundamental, direct") -> graph.Graph:
        """Plot the direct and fundamental bandgap along the trajectory."""
        series = [self._make_series(*choice) for choice in self._parse(selection)]
        return graph.Graph(series, xlabel="Step", ylabel="bandgap (eV)")

    def __str__(self):
        template = """\
Band structure
--------------
                 {header}
val. band max:   {val_band_max}
cond. band min:  {cond_band_min}
fundamental gap: {fundamental}
VBM @ kpoint:    {kpoint_vbm}
CBM @ kpoint:    {kpoint_cbm}

lower band:      {lower_band}
upper band:      {upper_band}
direct gap:      {direct}
@ kpoint:        {kpoint_direct}

Fermi energy:    {fermi_energy}"""
        return template.format(
            header=self._output_header(),
            val_band_max=self._output_energy("valence band maximum"),
            cond_band_min=self._output_energy("conduction band minimum"),
            fundamental=self._output_gap("fundamental"),
            kpoint_vbm=self._output_kpoint("VBM"),
            kpoint_cbm=self._output_kpoint("CBM"),
            lower_band=self._output_energy("direct gap bottom"),
            upper_band=self._output_energy("direct gap top"),
            direct=self._output_gap("direct"),
            kpoint_direct=self._output_kpoint("direct"),
            fermi_energy=self._output_energy("Fermi energy", component=slice(0, 1)),
        )

    def to_database(self) -> dict:
        bandgap_dict = {
            "valence_band_maximum": self._output_energy(
                "valence band maximum", to_string=False
            ),
            "conduction_band_minimum": self._output_energy(
                "conduction band minimum", to_string=False
            ),
            "fundamental_bandgap": self._output_gap("fundamental", to_string=False),
            "kpoint_vbm": self._output_kpoint("VBM", to_string=False),
            "kpoint_cbm": self._output_kpoint("CBM", to_string=False),
            "lower_band_direct_bandgap": self._output_energy(
                "direct gap bottom", to_string=False
            ),
            "upper_band_direct_bandgap": self._output_energy(
                "direct gap top", to_string=False
            ),
            "direct_bandgap": self._output_gap("direct", to_string=False),
            "kpoint_direct_bandgap": self._output_kpoint("direct", to_string=False),
        }

        final_dict = {}
        for k, v in bandgap_dict.items():
            final_dict[f"{k}_spin_independent"] = (
                v[0] if not isinstance(v, float) else float(v)
            )
            final_dict[f"{k}_spin_up"] = (
                (v[1] if not isinstance(v[1], float) else float(v[1]))
                if self._spin_polarized()
                else None
            )
            final_dict[f"{k}_spin_down"] = (
                (v[2] if not isinstance(v[2], float) else float(v[2]))
                if self._spin_polarized()
                else None
            )
        return Bandgap_DB(**final_dict)

    # --- Private helpers ---

    def _output_header(self):
        if self._spin_polarized():
            return "      spin independent             spin component 1             spin component 2"
        else:
            return "      spin independent"

    def _output_energy(self, label, component=slice(None), to_string=True):
        energies = self._get(label, steps=self._last_step_in_slice, component=component)
        if not (to_string):
            return energies
        return (9 * " ").join(map("{:20.6f}".format, energies))

    def _output_gap(self, label, to_string=True):
        gaps = self._gap(label, steps=self._last_step_in_slice)
        if not (to_string):
            return gaps
        return (9 * " ").join(map("{:20.6f}".format, gaps))

    def _output_kpoint(self, label, to_string=True):
        kpoints = self._kpoint(label, steps=self._last_step_in_slice)
        to_string_convert = lambda kpoint: " ".join(map("{:8.4f}".format, kpoint))
        if not (to_string):
            return np.array(kpoints).round(decimals=10).tolist()
        return " " + "   ".join(map(to_string_convert, kpoints))

    @property
    def _last_step_in_slice(self):
        if self._steps is None or self._steps == -1:
            return -1
        if isinstance(self._steps, slice):
            return (self._steps.stop or 0) - 1
        return self._steps

    @property
    def _slice(self):
        steps = self._steps
        if steps is None or steps == -1:
            return slice(-1, None)
        if isinstance(steps, slice):
            return steps
        return slice(steps, steps + 1)

    def _gap_dict(self, label):
        gaps = self._gap(label).T
        return {f"{label}{suffix}": gap for gap, suffix in zip(gaps, self._suffixes())}

    def _band_dict(self, label):
        result = {}
        bottoms = self._get(GAPS[label].bottom)
        tops = self._get(GAPS[label].top)
        for bottom, top, suffix in zip(bottoms.T, tops.T, self._suffixes()):
            result[GAPS[label].bottom.replace(" ", "_") + suffix] = bottom
            result[GAPS[label].top.replace(" ", "_") + suffix] = top
        return result

    def _kpoint_dict(self, label):
        kpoint = self._kpoint(label)
        return {
            f"kpoint_{label}{suffix}": kpoint[..., i, :]
            for i, suffix in enumerate(self._suffixes())
        }

    def _suffixes(self):
        return ("", "_up", "_down") if self._spin_polarized() else ("",)

    def _parse(self, selection):
        tree = select.Tree.from_selection(selection)
        for selection in tree.selections():
            self._raise_error_if_unused_selection(selection)
            components = self._parse_components(selection)
            labels = self._parse_labels(selection)
            yield from itertools.product(labels, components)

    def _raise_error_if_unused_selection(self, selection):
        if rest := set(selection).difference(GAPS).difference(COMPONENTS):
            raise exception.IncorrectUsage(
                f"A part of your selection {rest} could not be mapped to a valid selection"
            )

    def _parse_components(self, selection):
        components = set(COMPONENTS).intersection(selection)
        if not components:
            components = ("independent",)
        elif not self._spin_polarized():
            raise exception.IncorrectUsage(
                f"You selected a component {components} but the VASP calculation did not include spin polarization."
            )
        return components

    def _parse_labels(self, selection):
        labels = set(GAPS).intersection(selection)
        if not labels:
            labels = GAPS.keys()
        elif len(labels) > 1:
            raise exception.IncorrectUsage(
                f"Two conflicting labels selected {labels}. Please check your input."
            )
        return labels

    def _make_series(self, label, component):
        steps = np.arange(len(self._raw_bandgap.values))[self._slice] + 1
        gaps = self._gap(label, component=COMPONENTS.index(component))
        if component != "independent":
            label = f"{label}_{component}"
        return graph.Series(steps, np.atleast_1d(gaps), label)

    def _spin_polarized(self):
        return self._raw_bandgap.values.shape[1] == 3

    def _gap(self, label, **kwargs):
        top = self._get(GAPS[label].top, **kwargs)
        bottom = self._get(GAPS[label].bottom, **kwargs)
        return top - bottom

    def _kpoint(self, label, **kwargs):
        kpoint = [
            self._get(f"kx ({label})", **kwargs),
            self._get(f"ky ({label})", **kwargs),
            self._get(f"kz ({label})", **kwargs),
        ]
        return np.moveaxis(kpoint, 0, -1)

    def _get(self, desired_label, *, steps=None, component=slice(None)):
        steps = steps if steps is not None else self._steps
        if steps is None:
            steps = -1
        return next(
            self._raw_bandgap.values[steps, component, index]
            for index, label in enumerate(self._raw_bandgap.labels[:])
            if convert.text_to_string(label) == desired_label
        )


@quantity("bandgap")
@documentation.format(examples=slice_.examples("bandgap"))
class Bandgap(graph.Mixin):
    """This class describes the band extrema during the relaxation or MD simulation.

    The bandgap represents the energy difference between the highest energy electrons
    in the valence band and the lowest energy electrons in the conduction band of a
    material. The fundamental gap occurs between the energy states of electrons in the
    valence and conduction bands irrespective of the **k** point. In contrast, the
    direct gap means that transition from valence to conduction band does not change
    the **k** momentum.

    To study bandgap the extrema of the valence and conduction band play an important
    role. This class reports the valence band maximum as well as the conduction band
    minimum. For collinear calculations (ISPIN = 2) all values are reported separately
    for both spins as well as ignoring the spin. This simplifies comparison to
    experimental data, where the transitions either conserve the spin or not.

    {examples}
    """

    def __init__(self, source, quantity_name: str = "bandgap", steps=None):
        self._source = source
        self._quantity_name = quantity_name
        self._steps = steps

    @classmethod
    def from_data(cls, raw_bandgap: raw.Bandgap):
        """Create a Bandgap dispatcher from raw data (convenience for testing)."""
        return cls(source=DataSource(raw_bandgap))

    def __getitem__(self, steps) -> "Bandgap":
        new = copy.copy(self)
        new._steps = steps
        return new

    def _handler_factory(self, raw_data):
        return BandgapHandler.from_data(raw_data, steps=self._steps)

    @documentation.format(examples=slice_.examples("bandgap", "read"))
    def read(self, selection: str | None = None) -> dict:
        """Read the bandgap data from a VASP relaxation or MD trajectory.

        Returns
        -------
        dict
            Contains the fundamental and direct gap as well as the coordinates of the
            k points where the relevant points in the band structure are.

        {examples}
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            BandgapHandler.to_dict,
        )

    def to_dict(self, selection: str | None = None) -> dict:
        """Convenient alias for :py:meth:`read`. Please read the documentation there."""
        return self.read(selection=selection)

    @documentation.format(examples=slice_.examples("bandgap", "fundamental"))
    def fundamental(self, selection: str | None = None) -> np.ndarray:
        """Return the fundamental bandgap.

        The fundamental bandgap is between the maximum of the valence band and the
        minimum of the conduction band.

        Returns
        -------
        np.ndarray
            The value of the bandgap for all selected steps.

        {examples}
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            BandgapHandler.fundamental,
        )

    @documentation.format(examples=slice_.examples("bandgap", "direct"))
    def direct(self, selection: str | None = None) -> np.ndarray:
        """Return the direct bandgap.

        The direct bandgap is the minimal distance between a valence and conduction
        band at a single k point and for a single spin.

        Returns
        -------
        np.ndarray
            The value of the bandgap for all selected steps.

        {examples}
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            BandgapHandler.direct,
        )

    @documentation.format(examples=slice_.examples("bandgap", "valence_band_maximum"))
    def valence_band_maximum(self, selection: str | None = None) -> np.ndarray:
        """Return the valence band maximum.

        Returns
        -------
        np.ndarray
            The value of the valence band maximum for all selected steps.

        {examples}
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            BandgapHandler.valence_band_maximum,
        )

    @documentation.format(
        examples=slice_.examples("bandgap", "conduction_band_minimum")
    )
    def conduction_band_minimum(self, selection: str | None = None) -> np.ndarray:
        """Return the conduction band minimum.

        Returns
        -------
        np.ndarray
            The value of the conduction band minimum for all selected steps.

        {examples}
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            BandgapHandler.conduction_band_minimum,
        )

    @documentation.format(examples=slice_.examples("bandgap", "to_graph"))
    def to_graph(self, selection="fundamental, direct") -> graph.Graph:
        """Plot the direct and fundamental bandgap along the trajectory.

        Parameters
        ----------
        selection : str
            Select which bandgap to include in the plot. By default the fundamental
            and the direct one are included. In spin-polarized calculations, you can
            also select up or down to obtain the bandgap without spin flips.

        Returns
        -------
        Graph
            Figure with the ionic step on the x axis and the value of the bandgap on
            the y axis.

        {examples}
        """
        return merge_graphs(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            BandgapHandler.to_graph,
        )

    def __str__(self, selection: str | None = None):
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            BandgapHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def _spin_polarized(self):
        """Convenience for tests that access this on the dispatcher."""
        with self._source.access(self._quantity_name) as raw_data:
            return raw_data.values.shape[1] == 3

    def _to_database(self) -> dict:
        """Return {quantity[_selection]: handler_result} for database storage."""
        return merge_to_database(
            self._source,
            self._quantity_name,
            self._handler_factory,
            BandgapHandler.to_database,
        )

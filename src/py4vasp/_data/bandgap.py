# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp._data import base, slice_
from py4vasp._third_party import graph
from py4vasp._util import convert, documentation


@documentation.format(examples=slice_.examples("bandgap"))
class Bandgap(slice_.Mixin, base.Refinery, graph.Mixin):
    """Extract information about the band extrema during the relaxation or MD simulation.

    Contains utility functions to access the fundamental and direct bandgap as well as
    the k-point coordinates at which these are found.

    {examples}
    """

    @base.data_access
    def __str__(self):
        data = self.to_dict()
        return """\
bandgap:
    step: {step}
    fundamental:{fundamental:10.6f}
    direct:    {direct:10.6f}
kpoint:
    val. band min: {kpoint_vbm}
    cond. band max:{kpoint_cbm}
    direct gap:   {kpoint_direct}""".format(
            step=np.arange(len(self._raw_data.values))[self._slice][-1] + 1,
            fundamental=self._last_element(data["fundamental"]),
            direct=self._last_element(data["direct"]),
            kpoint_vbm=self._kpoint_str(data["kpoint_VBM"]),
            kpoint_cbm=self._kpoint_str(data["kpoint_CBM"]),
            kpoint_direct=self._kpoint_str(data["kpoint_direct"]),
        )

    def _kpoint_str(self, kpoint):
        kpoint = self._last_element(kpoint)
        return " ".join(map("{:10.6f}".format, kpoint))

    def _last_element(self, scalar_or_array):
        if self._is_slice:
            return scalar_or_array[-1]
        else:
            return scalar_or_array

    def _spin_polarized(self):
        return self._raw_data.values.shape[1] == 3

    @base.data_access
    @documentation.format(examples=slice_.examples("bandgap", "to_dict"))
    def to_dict(self):
        """Read the bandgap data from a VASP relaxation or MD trajectory.

        Returns
        -------
        dict
            Contains the fundamental and direct gap as well as the coordinates of the
            k points where the relevant points in the band structure are.

        {examples}
        """
        return {
            **self._fundamental_gap(),
            **self._kpoint_dict("VBM"),
            **self._kpoint_dict("CBM"),
            **self._direct_gap(),
            **self._kpoint_dict("direct"),
            "fermi_energy": self._get("Fermi energy", spin=0),
        }

    def _fundamental_gap(self):
        vbm = self._get("valence band maximum")
        cbm = self._get("conduction band minimum")
        return {
            f"fundamental{suffix}": cbm[..., i] - vbm[..., i]
            for i, suffix in enumerate(self._suffixes())
        }

    def _direct_gap(self):
        top = self._get("direct gap top")
        bottom = self._get("direct gap bottom")
        return {
            f"direct{suffix}": top[..., i] - bottom[..., i]
            for i, suffix in enumerate(self._suffixes())
        }

    def _kpoint_dict(self, label):
        kpoint = self._kpoint(label)
        return {
            f"kpoint_{label}{suffix}": kpoint[..., i, :]
            for i, suffix in enumerate(self._suffixes())
        }

    def _suffixes(self):
        return ("", "_up", "_down") if self._spin_polarized() else ("",)

    @base.data_access
    @documentation.format(examples=slice_.examples("bandgap", "fundamental"))
    def fundamental(self, selection="minimal"):
        """Return the fundamental bandgap.

        The fundamental bandgap is between the maximum of the valence band and the
        minimum of the conduction band.

        Returns
        -------
        np.ndarray
            The value of the bandgap for all selected steps.

        {examples}
        """
        cbm = self._get("conduction band minimum", 0)
        vbm = self._get("valence band maximum", 0)
        return cbm - vbm

    @base.data_access
    @documentation.format(examples=slice_.examples("bandgap", "direct"))
    def direct(self, selection="minimal"):
        """Return the direct bandgap.

        The direct bandgap is the minimal distance between a valence and conduction
        band at a single k point and for a single spin.

        Returns
        -------
        np.ndarray
            The value of the bandgap for all selected steps.

        {examples}
        """
        if selection == "minimal":
            return np.squeeze(
                np.min(
                    self._get("direct gap top") - self._get("direct gap bottom"),
                    axis=-1,
                )
            )
        spin = 0 if selection == "up" else 1
        top = self._get("direct gap top", spin)
        bottom = self._get("direct gap bottom", spin)
        return np.squeeze(top - bottom)

    def _kpoint(self, label, spin=slice(None)):
        kpoint = [
            self._get(f"kx ({label})", spin),
            self._get(f"ky ({label})", spin),
            self._get(f"kz ({label})", spin),
        ]
        return np.moveaxis(kpoint, 0, -1)

    def _get(self, desired_label, spin=slice(None)):
        return next(
            self._raw_data.values[self._steps, spin, index]
            for index, label in enumerate(self._raw_data.labels[:])
            if convert.text_to_string(label) == desired_label
        )

    @base.data_access
    @documentation.format(examples=slice_.examples("bandgap", "to_graph"))
    def to_graph(self):
        """Plot the direct and fundamental bandgap along the trajectory.

        Returns
        -------
        Graph
            Figure with the ionic step on the x axis and the value of the bandgap on
            the y axis.

        {examples}"""
        return graph.Graph(
            [self._make_series("fundamental"), self._make_series("direct")],
            xlabel="Step",
            ylabel="bandgap (eV)",
        )

    def _make_series(self, label):
        steps = np.arange(len(self._raw_data.values))[self._slice] + 1
        gaps = np.atleast_1d(getattr(self, label)())
        return graph.Series(steps, gaps, label)

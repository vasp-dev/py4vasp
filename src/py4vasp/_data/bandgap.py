# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp._data import base, slice_
from py4vasp._third_party import graph
from py4vasp._util import convert, documentation


@documentation.format(examples=slice_.examples("bandgap"))
class Bandgap(slice_.Mixin, base.Refinery, graph.Mixin):
    """Extract information about the band extrema during the relaxation or MD simulation.

    Contains utility functions to access the fundamental and optical bandgap as well as
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
    optical:    {optical:10.6f}
kpoint:
    val. band min: {kpoint_vbm}
    cond. band max:{kpoint_cbm}
    optical gap:   {kpoint_optical}""".format(
            step=np.arange(len(self._raw_data.values))[self._slice][-1] + 1,
            fundamental=self._last_element(data["fundamental"]),
            optical=self._last_element(data["optical"]),
            kpoint_vbm=self._kpoint_str(data["kpoint_VBM"]),
            kpoint_cbm=self._kpoint_str(data["kpoint_CBM"]),
            kpoint_optical=self._kpoint_str(data["kpoint_optical"]),
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
        return self._raw_data.values.shape[1] == 2

    @base.data_access
    @documentation.format(examples=slice_.examples("bandgap", "to_dict"))
    def to_dict(self):
        """Read the bandgap data from a VASP relaxation or MD trajectory.

        Returns
        -------
        dict
            Contains the fundamental and optical gap as well as the coordinates of the
            k points where the relevant points in the band structure are.

        {examples}
        """
        return {
            **self._minimal_gap(),
            "optical": self.optical(),
            "kpoint_optical": np.squeeze(self._kpoint("optical")),
            "fermi_energy": self._get_old("Fermi energy"),
        }

    def _minimal_gap(self):
        vbm = self._get("valence band maximum")
        cbm = self._get("conduction band minimum")
        max_spin = np.argmax(vbm, axis=-1)
        min_spin = np.argmin(cbm, axis=-1)
        vbm = np.array([band[spin] for band, spin in zip(vbm, max_spin)])
        cbm = np.array([band[spin] for band, spin in zip(cbm, min_spin)])
        kpoint_vbm = [kpt[spin] for kpt, spin in zip(self._kpoint("VBM"), max_spin)]
        kpoint_cbm = [kpt[spin] for kpt, spin in zip(self._kpoint("CBM"), min_spin)]
        return {
            "fundamental": np.squeeze(cbm - vbm),
            "kpoint_VBM": np.squeeze(kpoint_vbm),
            "kpoint_CBM": np.squeeze(kpoint_cbm),
        }

    @base.data_access
    @documentation.format(examples=slice_.examples("bandgap", "fundamental"))
    def fundamental(self):
        """Return the fundamental bandgap.

        The fundamental bandgap is between the maximum of the valence band and the
        minimum of the conduction band.

        Returns
        -------
        np.ndarray
            The value of the bandgap for all selected steps.

        {examples}
        """
        return self._get_old("conduction band minimum") - self._get_old(
            "valence band maximum"
        )

    @base.data_access
    @documentation.format(examples=slice_.examples("bandgap", "optical"))
    def optical(self):
        """Return the optical bandgap.

        The optical bandgap is the minimal distance between a valence and conduction
        band at a single k point and for a single spin.

        Returns
        -------
        np.ndarray
            The value of the bandgap for all selected steps.

        {examples}
        """
        return self._get_old("optical gap top") - self._get_old("optical gap bottom")

    def _kpoint(self, label, spin=slice(None)):
        kpoint = [
            self._get(f"kx ({label})", spin),
            self._get(f"ky ({label})", spin),
            self._get(f"kz ({label})", spin),
        ]
        return np.moveaxis(kpoint, 0, -1)

    def _kpoint_old(self, label):
        return self._kpoint(label, spin=0)

    def _get(self, desired_label, spin=slice(None)):
        return next(
            self._raw_data.values[self._slice, spin, index]
            for index, label in enumerate(self._raw_data.labels[:])
            if convert.text_to_string(label) == desired_label
        )

    def _get_old(self, desired_label):
        return np.squeeze(self._get(desired_label, spin=0))

    @base.data_access
    @documentation.format(examples=slice_.examples("bandgap", "to_graph"))
    def to_graph(self):
        """Plot the optical and fundamental bandgap along the trajectory.

        Returns
        -------
        Graph
            Figure with the ionic step on the x axis and the value of the bandgap on
            the y axis.

        {examples}"""
        return graph.Graph(
            [self._make_series("fundamental"), self._make_series("optical")],
            xlabel="Step",
            ylabel="bandgap (eV)",
        )

    def _make_series(self, label):
        steps = np.arange(len(self._raw_data.values))[self._slice] + 1
        gaps = np.atleast_1d(getattr(self, label)())
        return graph.Series(steps, gaps, label)
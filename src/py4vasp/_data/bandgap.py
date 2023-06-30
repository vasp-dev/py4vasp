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

    @base.data_access
    def to_dict(self):
        return {
            "fundamental": self.fundamental(),
            "kpoint_VBM": self._kpoint("VBM"),
            "kpoint_CBM": self._kpoint("CBM"),
            "optical": self.optical(),
            "kpoint_optical": self._kpoint("optical"),
            "fermi_energy": self._get("Fermi energy"),
        }

    @base.data_access
    def fundamental(self):
        return self._get("conduction band minimum") - self._get("valence band maximum")

    @base.data_access
    def optical(self):
        return self._get("optical gap top") - self._get("optical gap bottom")

    def _kpoint(self, label):
        kpoint = [
            self._get(f"kx ({label})"),
            self._get(f"ky ({label})"),
            self._get(f"kz ({label})"),
        ]
        return np.array(kpoint).T

    def _get(self, desired_label):
        return next(
            self._raw_data.values[self._steps, index]
            for index, label in enumerate(self._raw_data.labels[:])
            if convert.text_to_string(label) == desired_label
        )

    @base.data_access
    def to_graph(self):
        return graph.Graph(
            [self._make_series("fundamental"), self._make_series("optical")],
            xlabel="Step",
            ylabel="bandgap (eV)",
        )

    def _make_series(self, label):
        steps = np.arange(len(self._raw_data.values))[self._slice] + 1
        gaps = np.atleast_1d(getattr(self, label)())
        return graph.Series(steps, gaps, label)

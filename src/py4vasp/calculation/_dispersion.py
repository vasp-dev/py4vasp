# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

import py4vasp._third_party.graph as _graph
from py4vasp import data
from py4vasp._data import base


class Dispersion(base.Refinery):
    """Generic class for all dispersions (electrons, phonons).

    Provides some utility functionalities common to all dispersions to avoid duplication
    of code."""

    @base.data_access
    def __str__(self):
        return f"""band data:
    {self._kpoints.number_kpoints()} k-points
    {self._raw_data.eigenvalues.shape[-1]} bands"""

    @base.data_access
    def to_dict(self):
        """Read the dispersion into a dictionary.

        Returns
        -------
        dict
            Contains the **k**-point distances and associated labels as well as the
            eigenvalues of all the bands.
        """
        return {
            "kpoint_distances": self._kpoints.distances(),
            "kpoint_labels": self._kpoints.labels(),
            "eigenvalues": self._raw_data.eigenvalues[:],
        }

    @property
    def _kpoints(self):
        return data.Kpoint.from_data(self._raw_data.kpoints)

    @base.data_access
    def plot(self, projections=None):
        """Generate a graph of the dispersion.

        The bands are plotted along the k-point distances. The k-point labels are added
        as ticks if present. Pass a dictionary with projections to generate a fatband
        plot based on the weights passed.

        Parameters
        ----------
        projections : dict
            The key will be used for the legend of the figure. The values will be used
            to broaden the lines. Must have the same shape as the eigenvalues of the
            dispersion.

        Returns
        -------
        Graph
            Contains the band structure for all the **k** points. If projections are
            passed, the width of the band is adjusted accordingly.
        """
        data = self.to_dict()
        projections = self._use_projections_or_default(projections)
        return _graph.Graph(
            series=_band_structure(data, projections),
            xticks=_xticks(data, self._kpoints.line_length()),
        )

    def _use_projections_or_default(self, projections):
        if projections is not None:
            return projections
        elif self._spin_polarized():
            return {"up": None, "down": None}
        else:
            return {"bands": None}

    def _spin_polarized(self):
        eigenvalues = self._raw_data.eigenvalues
        return eigenvalues.ndim == 3 and eigenvalues.shape[0] == 2


def _band_structure(data, projections):
    return [_make_series(data, projection) for projection in projections.items()]


def _make_series(data, projection):
    name, width = _get_name_and_width(projection)
    x = data["kpoint_distances"]
    y = _get_bands(data["eigenvalues"], name)
    return _graph.Series(x, y, name, width=width)


def _get_name_and_width(projection):
    name, width = projection
    if width is not None:
        width = width.T
    return name, width


def _get_bands(eigenvalues, name):
    if eigenvalues.ndim == 2:
        return eigenvalues.T
    elif "down" in name:
        return eigenvalues[1].T
    else:
        return eigenvalues[0].T


def _xticks(data, line_length):
    ticks, labels = _degenerate_ticks_and_labels(data, line_length)
    return _filter_unique(ticks, labels)


def _degenerate_ticks_and_labels(data, line_length):
    labels = _tick_labels(data)
    mask = _use_labels_and_line_edges(labels, line_length)
    return data["kpoint_distances"][mask], labels[mask]


def _tick_labels(data):
    if data["kpoint_labels"] is None:
        return np.zeros(len(data["kpoint_distances"]), str)
    else:
        return np.array(data["kpoint_labels"])


def _use_labels_and_line_edges(labels, line_length):
    mask = labels != ""
    mask[::line_length] = True
    mask[-1] = True
    return mask


def _filter_unique(ticks, labels):
    result = {}
    for tick, label in zip(ticks, labels):
        if tick in result:
            previous_label = result[tick]
            if previous_label != "" and previous_label != label:
                label = previous_label + "|" + label
        result[tick] = label
    return result

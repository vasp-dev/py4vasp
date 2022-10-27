# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
from py4vasp import data
from py4vasp.data import _base
import py4vasp._third_party.graph as _graph


class Dispersion(_base.Refinery):
    """Generic class for all dispersions (electrons, phonons).

    Provides some utility functionalities common to all dispersions to avoid duplication
    of code."""

    def to_dict(self):
        return {
            "kpoint_distances": self._kpoints.distances(),
            "kpoint_labels": self._kpoints.labels(),
            "eigenvalues": self._raw_data.eigenvalues[:],
        }

    @property
    def _kpoints(self):
        return data.Kpoint.from_data(self._raw_data.kpoints)

    def plot(self):
        data = self.to_dict()
        return _graph.Graph(
            series=_band_structure(data),
            xticks=_xticks(data, self._kpoints.line_length()),
        )


def _band_structure(data):
    bands = _make_3d(data["eigenvalues"])
    return [_graph.Series(data["kpoint_distances"], component.T) for component in bands]


def _make_3d(eigenvalues):
    if eigenvalues.ndim == 2:
        return eigenvalues[None, :, :]
    else:
        return eigenvalues


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

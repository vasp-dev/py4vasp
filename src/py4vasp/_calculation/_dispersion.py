# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

import py4vasp._third_party.graph as _graph
from py4vasp._calculation import projector
from py4vasp._calculation.dispatch import (
    DataSource,
    _dispatch,
    merge_default,
    merge_strings,
    merge_to_database,
    quantity,
)
from py4vasp._calculation.kpoint import KpointHandler
from py4vasp._raw import data as raw
from py4vasp._raw.data_db import Dispersion_DB
from py4vasp._util import check


class DispersionHandler:
    """Handler for dispersion (band/phonon) data."""

    def __init__(self, raw_dispersion: raw.Dispersion):
        self._raw_dispersion = raw_dispersion

    @classmethod
    def from_data(cls, raw_dispersion: raw.Dispersion) -> "DispersionHandler":
        return cls(raw_dispersion)

    def __str__(self):
        return f"""band data:
    {self._kpoints().number_kpoints()} k-points
    {self._raw_dispersion.eigenvalues.shape[-1]} bands"""

    def read(self) -> dict:
        return self.to_dict()

    def to_dict(self) -> dict:
        kpoints = self._kpoints()
        kpoint_labels = kpoints.labels()
        labels_dict = {} if kpoint_labels is None else {"kpoint_labels": kpoint_labels}
        return {
            "kpoint_distances": kpoints.distances(),
            **labels_dict,
            "eigenvalues": self._raw_dispersion.eigenvalues[:],
        }

    def to_database(self) -> dict:
        eigenvalues = (
            self._raw_dispersion.eigenvalues[:]
            if not check.is_none(self._raw_dispersion.eigenvalues)
            else None
        )
        min_eigenvalue = float(np.min(eigenvalues)) if eigenvalues is not None else None
        max_eigenvalue = float(np.max(eigenvalues)) if eigenvalues is not None else None

        min_eigenvalue_up, max_eigenvalue_up = None, None
        min_eigenvalue_down, max_eigenvalue_down = None, None
        if self._spin_polarized():
            eigenvalues_up = eigenvalues[0]
            eigenvalues_down = eigenvalues[1]
            min_eigenvalue_up = float(np.min(eigenvalues_up))
            max_eigenvalue_up = float(np.max(eigenvalues_up))
            min_eigenvalue_down = float(np.min(eigenvalues_down))
            max_eigenvalue_down = float(np.max(eigenvalues_down))

        return Dispersion_DB(
            eigenvalue_min=min_eigenvalue,
            eigenvalue_max=max_eigenvalue,
            eigenvalue_min_up=min_eigenvalue_up,
            eigenvalue_max_up=max_eigenvalue_up,
            eigenvalue_min_down=min_eigenvalue_down,
            eigenvalue_max_down=max_eigenvalue_down,
        )

    def plot(self, projections=None):
        data = self.to_dict()
        projections = self._use_projections_or_default(projections)
        return _graph.Graph(
            series=_band_structure(data, projections),
            xticks=_xticks(data, self._kpoints().line_length()),
        )

    def _kpoints(self):
        return KpointHandler.from_data(self._raw_dispersion.kpoints)

    def _use_projections_or_default(self, projections):
        if projections is not None:
            return projections
        elif self._spin_polarized():
            return {"up": None, "down": None}
        else:
            return {"bands": None}

    def _spin_polarized(self):
        eigenvalues = self._raw_dispersion.eigenvalues
        return eigenvalues.ndim == 3 and eigenvalues.shape[0] == 2


@quantity("_dispersion")
class Dispersion:
    """Generic class for all dispersions (electrons, phonons)."""

    def __init__(self, source, quantity_name="dispersion"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_dispersion):
        return cls(source=DataSource(raw_dispersion))

    def _handler_factory(self, raw):
        return DispersionHandler.from_data(raw)

    def __str__(self, selection=None):
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            DispersionHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def read(self, selection=None) -> dict:
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            DispersionHandler.read,
        )

    def to_dict(self, selection=None) -> dict:
        return self.read(selection=selection)

    def plot(self, projections=None):
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            DispersionHandler.plot,
            projections,
        )

    def _to_database(self) -> dict:
        """Return {quantity[_selection]: handler_result} for database storage."""
        return merge_to_database(
            self._source,
            self._quantity_name,
            DispersionHandler.from_data,
            DispersionHandler.to_database,
        )


def _band_structure(data, projections):
    spin_projections = projections.get(projector.SPIN_PROJECTION, [])
    return [
        _make_series(data, label, weight, label in spin_projections)
        for label, weight in projections.items()
        if label != projector.SPIN_PROJECTION
    ]


def _make_series(data, label, weight, is_spin_projection):
    options = {}
    if not check.is_none(weight):
        options["weight"] = weight.T
    if is_spin_projection:
        options["marker"] = "o"
        options["weight_mode"] = "color"
    x = data["kpoint_distances"]
    y = _get_bands(data["eigenvalues"], label)
    return _graph.Series(x, y, label, **options)


def _get_bands(eigenvalues, label):
    if eigenvalues.ndim == 2:
        return eigenvalues.T
    elif "down" in label:
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
    kpoint_labels = data.get("kpoint_labels")
    if kpoint_labels is None:
        return np.zeros(len(data["kpoint_distances"]), str)
    else:
        return np.array(kpoint_labels)


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

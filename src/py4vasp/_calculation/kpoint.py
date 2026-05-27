# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import functools
from contextlib import suppress
from fractions import Fraction
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from py4vasp import exception
from py4vasp._calculation.dispatch import DataSource, merge_default, merge_strings, quantity
from py4vasp._raw import data as raw
from py4vasp._raw.data_db import Kpoint_DB
from py4vasp._util import check, convert

_TO_DATABASE_SUPPRESSED_EXCEPTIONS = (
    exception.Py4VaspError,
    exception.RefinementError,
)


def _safe_call(func):
    with suppress(*_TO_DATABASE_SUPPRESSED_EXCEPTIONS):
        return func()
    return None


class KpointHandler:
    """Handler for k-point data."""

    def __init__(self, raw_kpoint: raw.Kpoint):
        self._raw_kpoint = raw_kpoint

    @classmethod
    def from_data(cls, raw_kpoint: raw.Kpoint) -> "KpointHandler":
        return cls(raw_kpoint)

    def __str__(self):
        text = f"""k-points
{len(self._raw_kpoint.coordinates)}
reciprocal"""
        for kpoint, weight in zip(self._raw_kpoint.coordinates, self._raw_kpoint.weights):
            text += "\n" + f"{kpoint[0]} {kpoint[1]} {kpoint[2]}  {weight}"
        return text

    def read(self) -> dict:
        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        labels = self.labels()
        labels_dict = {} if labels is None else {"labels": labels}
        return {
            "mode": self.mode(),
            "line_length": self.line_length(),
            "number_kpoints": self.number_kpoints(),
            "coordinates": self._raw_kpoint.coordinates[:],
            "weights": self._raw_kpoint.weights[:],
            **labels_dict,
        }

    def to_database(self) -> dict:
        number_x = self._raw_kpoint.number_x
        number_y = self._raw_kpoint.number_y
        number_z = self._raw_kpoint.number_z
        has_grid = not (any(check.is_none(n) for n in (number_x, number_y, number_z)))
        grid_kpoints = (
            None
            if not has_grid
            else [number_x, number_y, number_z]
        )
        user_labels = None
        if not check.is_none(self._raw_kpoint.label_indices):
            user_labels = [k for k in self._labels_from_file() if k != ""]
            user_labels = None if len(user_labels) == 0 else user_labels
        sampled_points = sorted(set(user_labels)) if user_labels is not None else None
        mode = _safe_call(self.mode)
        line_length = _safe_call(self.line_length)
        num_kpoints_total = _safe_call(self.number_kpoints)
        num_lines = _safe_call(self.number_lines)
        return {
            "kpoint": Kpoint_DB(
                mode=mode,
                line_length=line_length,
                num_kpoints_total=num_kpoints_total,
                num_lines=num_lines,
                num_kpoints_grid=grid_kpoints,
                labels=user_labels,
                labels_unique=sampled_points,
            )
        }

    def line_length(self) -> int:
        if self.mode() == "line":
            return self._raw_kpoint.number
        return self.number_kpoints()

    def number_lines(self) -> int:
        return int(self.number_kpoints() // self.line_length())

    def number_kpoints(self) -> int:
        return len(self._raw_kpoint.coordinates)

    def distances(self) -> np.ndarray:
        cell = _last_step(self._raw_kpoint.cell.lattice_vectors)
        cartesian_kpoints = np.linalg.solve(cell, self._raw_kpoint.coordinates[:].T).T
        kpoint_lines = np.split(cartesian_kpoints, self.number_lines())
        kpoint_norms = [_line_distances(line) for line in kpoint_lines]
        concatenate_distances = lambda current, addition: (
            np.concatenate((current, addition + current[-1]))
        )
        return functools.reduce(concatenate_distances, kpoint_norms)

    def mode(self) -> str:
        mode = convert.text_to_string(self._raw_kpoint.mode).strip() or "# empty string"
        first_char = mode[0].lower()
        if first_char == "a":
            return "automatic"
        elif first_char == "b":
            return "generating lattice"
        elif first_char == "e":
            return "explicit"
        elif first_char == "g":
            return "gamma"
        elif first_char == "l":
            return "line"
        elif first_char == "m":
            return "monkhorst"
        else:
            raise exception.RefinementError(
                f"Could not understand the mode \'{mode}\' when refining the raw kpoints data."
            )

    def labels(self) -> list[str] | None:
        if not self._raw_kpoint.label_indices.is_none():
            return self._labels_from_file()
        elif self.mode() == "line":
            return self._labels_at_band_edges()
        else:
            return None

    def path_indices(self, start: ArrayLike, finish: ArrayLike) -> np.ndarray:
        direction = np.array(finish) - np.array(start)
        deltas = self._raw_kpoint.coordinates - np.array(start)
        areas = np.linalg.norm(np.cross(direction, deltas), axis=1)
        return np.flatnonzero(np.isclose(areas, 0))

    def _labels_from_file(self):
        labels = [""] * len(self._raw_kpoint.coordinates)
        for label, index in zip(self._raw_kpoint.labels, self._raw_indices()):
            labels[index] = convert.text_to_string(label.strip())
        return labels

    def _raw_indices(self):
        indices = np.array(self._raw_kpoint.label_indices)
        if self.mode() == "line":
            line_length = self.line_length()
            return line_length * (indices // 2) - (indices + 1) % 2
        else:
            return indices - 1

    def _labels_at_band_edges(self):
        line_length = self.line_length()
        band_edge = lambda index: not (0 < index % line_length < line_length - 1)
        return [
            _kpoint_label(kpoint) if band_edge(index) else ""
            for index, kpoint in enumerate(self._raw_kpoint.coordinates)
        ]

    def _reciprocal_lattice_vectors(self):
        scale = self._raw_kpoint.cell.scale
        lattice_vectors = scale * _last_step(self._raw_kpoint.cell.lattice_vectors)
        volume = np.linalg.det(lattice_vectors)
        return (2.0 * np.pi / volume) * np.array(
            [
                np.cross(lattice_vectors[1], lattice_vectors[2]),
                np.cross(lattice_vectors[2], lattice_vectors[0]),
                np.cross(lattice_vectors[0], lattice_vectors[1]),
            ]
        )


@quantity("kpoint")
class Kpoint:
    """The k-point mesh used in the VASP calculation."""

    def __init__(self, source, quantity_name="kpoint"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_kpoint):
        return cls(source=DataSource(raw_kpoint))

    def _handler_factory(self, raw):
        return KpointHandler.from_data(raw)

    def __str__(self):
        return merge_strings(
            self._source, self._quantity_name, None,
            self._handler_factory, KpointHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def read(self, selection=None) -> dict:
        return merge_default(
            self._source, self._quantity_name, None,
            self._handler_factory, KpointHandler.read,
        )

    def to_dict(self, selection=None) -> dict:
        return self.read(selection=selection)

    def line_length(self) -> int:
        return merge_default(
            self._source, self._quantity_name, None,
            self._handler_factory, KpointHandler.line_length,
        )

    def number_lines(self) -> int:
        return merge_default(
            self._source, self._quantity_name, None,
            self._handler_factory, KpointHandler.number_lines,
        )

    def number_kpoints(self) -> int:
        return merge_default(
            self._source, self._quantity_name, None,
            self._handler_factory, KpointHandler.number_kpoints,
        )

    def distances(self) -> np.ndarray:
        return merge_default(
            self._source, self._quantity_name, None,
            self._handler_factory, KpointHandler.distances,
        )

    def mode(self) -> str:
        return merge_default(
            self._source, self._quantity_name, None,
            self._handler_factory, KpointHandler.mode,
        )

    def labels(self) -> list[str] | None:
        return merge_default(
            self._source, self._quantity_name, None,
            self._handler_factory, KpointHandler.labels,
        )

    def path_indices(self, start: ArrayLike, finish: ArrayLike) -> np.ndarray:
        return merge_default(
            self._source, self._quantity_name, None,
            self._handler_factory, KpointHandler.path_indices,
            start, finish,
        )

    def selections(self):
        from py4vasp._raw import definition as raw_module
        return {self._quantity_name: list(raw_module.selections(self._quantity_name))}

    def _reciprocal_lattice_vectors(self):
        return merge_default(
            self._source, self._quantity_name, None,
            self._handler_factory, KpointHandler._reciprocal_lattice_vectors,
        )


def _last_step(lattice_vectors):
    if lattice_vectors.ndim == 2:
        return lattice_vectors
    else:
        return lattice_vectors[-1]


def _line_distances(coordinates):
    distances = np.zeros(len(coordinates))
    norms = np.linalg.norm(coordinates[1:] - coordinates[:-1], axis=1)
    distances[1:] = np.cumsum(norms)
    return distances


def _kpoint_label(kpoint):
    fractions = [convert.Fraction(coordinate).latex() for coordinate in kpoint]
    return f"$[{fractions[0]} {fractions[1]} {fractions[2]}]$"

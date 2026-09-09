# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools

import numpy as np

from py4vasp import raw
from py4vasp._demo import showcase
from py4vasp._demo.showcase import cell

# High-symmetry points of the body-centred tetragonal Brillouin zone with c > a, as
# fraction of the primitive reciprocal lattice vectors. Taken from the standard setting
# of Setyawan and Curtarolo, Comput. Mater. Sci. 49, 299 (2010); the two shape
# parameters depend on the lattice constants, and showcase.cell gives the cell in that
# same standard setting, so the points below are exact rather than approximate.
_ZETA = cell.LATTICE_CONSTANT**2 / (2 * cell.HEIGHT**2)
_ETA = (1 + cell.LATTICE_CONSTANT**2 / cell.HEIGHT**2) / 4
SPECIAL_POINTS = {
    "GM": (0.0, 0.0, 0.0),
    "N": (0.0, 0.5, 0.0),
    "P": (0.25, 0.25, 0.25),
    "X": (0.0, 0.0, 0.5),
    "Z": (0.5, 0.5, -0.5),
    "SIGMA": (-_ETA, _ETA, _ETA),
    "Y": (-_ZETA, _ZETA, 0.5),
}
# A contiguous path visiting the valence band maximum at Gamma and the conduction band
# minimum at P, so the band structure shows that the gap of Sr2TiO4 is indirect.
PATH = ("GM", "X", "P", "N", "GM")

# High-symmetry points of the face-centred cubic Brillouin zone, from the same
# standard setting. The path visits every one of them and returns to Gamma.
FCC_SPECIAL_POINTS = {
    "GM": (0.0, 0.0, 0.0),
    "X": (0.5, 0.0, 0.5),
    "W": (0.5, 0.25, 0.75),
    "L": (0.5, 0.5, 0.5),
    "K": (0.375, 0.375, 0.75),
    "U": (0.625, 0.25, 0.625),
}
FCC_PATH = ("GM", "X", "W", "L", "GM")
_LABELS = {name: name for name in ("X", "P", "N", "Z", "W", "L", "K", "U")}
_LABELS["GM"] = r"$\Gamma$"


def line_mode(labels="with_labels") -> raw.Kpoint:
    """Band-structure path through the Brillouin zone of Sr2TiO4.

    Parameters
    ----------
    labels
        Pass ``"no_labels"`` to leave the high-symmetry points unnamed, as a KPOINTS
        file without labels would. py4vasp then names the band edges by their
        coordinates instead.
    """
    return _path(SPECIAL_POINTS, PATH, cell.Sr2TiO4(), labels)


def line_mode_Fe3O4(labels="with_labels") -> raw.Kpoint:
    """Band-structure path through the Brillouin zone of magnetite."""
    return _path(FCC_SPECIAL_POINTS, FCC_PATH, cell.Fe3O4(), labels)


def _path(special_points, path, raw_cell, labels) -> raw.Kpoint:
    corners = [special_points[label] for label in path]
    segments = [
        np.linspace(start, finish, showcase.LINE_LENGTH)
        for start, finish in zip(corners, corners[1:])
    ]
    coordinates = np.concatenate(segments)
    return raw.Kpoint(
        mode="line",
        number=showcase.LINE_LENGTH,
        coordinates=raw.VaspData(coordinates),
        weights=raw.VaspData(np.ones(len(coordinates))),
        cell=raw_cell,
        **_labels(labels, path),
    )


def mesh() -> np.ndarray:
    """Gamma-centred mesh a density of states is integrated over.

    Returns
    -------
    -
        Fractional reciprocal coordinates of shape ``(kpoint, 3)``. The mesh contains
        Gamma and P, where the band extrema of the model lie, so a density of states
        integrated over it reproduces the gap of the band structure exactly.
    """
    axes = (np.arange(number) / number for number in showcase.KPOINT_GRID)
    return np.array(list(itertools.product(*axes)))


def _labels(labels, path):
    if labels != "with_labels":
        return {}
    # VASP numbers the endpoints of the segments, so segment i starts at 2 * i - 1 and
    # ends at 2 * i. Naming both endpoints of every segment is what a KPOINTS file in
    # line mode does, and it tells py4vasp the path is continuous: a corner shared by two
    # segments carries the same name from both sides, so the tick reads "X" rather than
    # "X|", which is how py4vasp marks a jump in the path.
    number_endpoints = 2 * (len(path) - 1)
    names = [_LABELS[path[(index + 1) // 2]] for index in range(number_endpoints)]
    return {
        "labels": raw.VaspData(np.array(names, dtype="S")),
        "label_indices": raw.VaspData(np.arange(1, number_endpoints + 1)),
    }

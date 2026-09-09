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
# parameters depend on the lattice constants.
_ZETA = cell.SR2TIO4_A**2 / (2 * cell.SR2TIO4_C**2)
_ETA = (1 + cell.SR2TIO4_A**2 / cell.SR2TIO4_C**2) / 4
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
_LABELS = {"GM": r"$\Gamma$"}


def line_mode() -> raw.Kpoint:
    """Labelled band-structure path through the Brillouin zone of Sr2TiO4."""
    corners = [SPECIAL_POINTS[label] for label in PATH]
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
        cell=cell.Sr2TiO4(),
        labels=raw.VaspData(np.array([_label(name) for name in PATH], dtype="S")),
        label_indices=raw.VaspData(np.array(_label_indices())),
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


def _label(name):
    return _LABELS.get(name, name)


def _label_indices():
    # VASP numbers the endpoints of the segments, so segment i starts at 2 * i - 1 and
    # ends at 2 * i. Labelling the end of every segment plus the very first point names
    # each corner of the path exactly once.
    return [1, *(2 * segment for segment in range(1, len(PATH)))]

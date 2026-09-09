# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools

import numpy as np

from py4vasp import _demo, raw
from py4vasp._demo import showcase

# Lattice constants of the conventional I4/mmm cell of Sr2TiO4 in Angstrom. spglib
# analyzes py4vasp._demo.structure.Sr2TiO4 to these; the literature reports a = 3.884
# and c = 12.60 for the K2NiF4-type structure.
SR2TIO4_A = 3.9277
SR2TIO4_C = 12.6828

# High-symmetry points of the body-centred tetragonal Brillouin zone with c > a, as
# fraction of the primitive reciprocal lattice vectors. Taken from the standard setting
# of Setyawan and Curtarolo, Comput. Mater. Sci. 49, 299 (2010); the two shape
# parameters depend on the lattice constants. py4vasp._demo.cell.Sr2TiO4 uses a basis
# that is this standard one turned in space -- the two have the same metric tensor, so
# they share their fractional coordinates and these tabulated points apply unchanged.
_ZETA = SR2TIO4_A**2 / (2 * SR2TIO4_C**2)
_ETA = (1 + SR2TIO4_A**2 / SR2TIO4_C**2) / 4
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
_LABELS = {"GM": r"$\Gamma$", "X": "X", "P": "P", "N": "N", "Z": "Z"}


def line_mode(labels="with_labels") -> raw.Kpoint:
    """Band-structure path through the Brillouin zone of Sr2TiO4.

    Parameters
    ----------
    labels
        Pass ``"no_labels"`` to leave the high-symmetry points unnamed, as a KPOINTS
        file without labels would. py4vasp then names the band edges by their
        coordinates instead.
    """
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
        cell=_demo.cell.Sr2TiO4(),
        **_labels(labels),
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


def _labels(labels):
    if labels != "with_labels":
        return {}
    # VASP numbers the endpoints of the segments, so segment i starts at 2 * i - 1 and
    # ends at 2 * i. Naming both endpoints of every segment is what a KPOINTS file in
    # line mode does, and it tells py4vasp the path is continuous: a corner shared by two
    # segments carries the same name from both sides, so the tick reads "X" rather than
    # "X|", which is how py4vasp marks a jump in the path.
    number_endpoints = 2 * (len(PATH) - 1)
    names = [_LABELS.get(PATH[(index + 1) // 2]) for index in range(number_endpoints)]
    return {
        "labels": raw.VaspData(np.array(names, dtype="S")),
        "label_indices": raw.VaspData(np.arange(1, number_endpoints + 1)),
    }

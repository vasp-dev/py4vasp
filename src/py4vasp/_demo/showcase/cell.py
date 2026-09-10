# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw
from py4vasp._demo import showcase

# Lattice constants of the conventional I4/mmm cell of Sr2TiO4 in Angstrom. spglib
# analyzes py4vasp._demo.structure.Sr2TiO4 to these values, so the showcase describes the
# same crystal as the test data; the literature reports a = 3.884 and c = 12.60 for the
# K2NiF4-type structure, about a percent smaller as one expects of measured against
# calculated constants.
LATTICE_CONSTANT = 3.92771
HEIGHT = 12.68276

# Cubic lattice constant of magnetite in Angstrom, as the literature reports it for
# the Fd-3m spinel.
FE3O4_LATTICE_CONSTANT = 8.394

# Cubic lattice constant of copper in Angstrom, as the literature reports it for the
# face-centred cubic metal.
CU_LATTICE_CONSTANT = 3.615

# Lattice constants of graphite in Angstrom as the literature reports them for the
# Bernal-stacked crystal: a = 2.4612 and c = 6.7079, so the layers sit half of c apart.
# py4vasp._demo.cell.Graphite uses 2.44105 instead, a hundredth of an Angstrom short.
GRAPHITE_LATTICE_CONSTANT = 2.4612
GRAPHITE_INTERLAYER_DISTANCE = 3.35395
# Height of the slab cell. The four layers span three interlayer distances, so this
# leaves close to fourteen Angstrom of vacuum, far more than the five py4vasp demands
# of a cell it places a scanning tip above. The generous margin is what gives the
# plane-averaged potential a flat plateau to read a work function off, rather than a
# tail of the slab that is still decaying when the next cell begins.
GRAPHITE_HEIGHT = 24.0

INITIAL_COMPRESSION = 0.98  # the relaxation starts from a cell 2% too small


def lattice_vectors() -> np.ndarray:
    """Primitive lattice vectors of body-centred tetragonal Sr2TiO4 in Angstrom.

    Given in the standard setting for two reasons. The high-symmetry points of the
    Brillouin zone are tabulated for it, so the labels along the band-structure path are
    the real ones. And the vectors are exact, whereas the ones of
    :func:`py4vasp._demo.cell.Sr2TiO4` are rounded to a precision that leaves the crystal
    a little off-tetragonal, so its symmetry only resolves to I4/mmm at a tolerance ten
    times looser than the one py4vasp analyzes symmetry with.
    """
    a, c = LATTICE_CONSTANT, HEIGHT
    return np.array(
        [
            [-a / 2, a / 2, c / 2],
            [a / 2, -a / 2, c / 2],
            [a / 2, a / 2, -c / 2],
        ]
    )


def Sr2TiO4() -> raw.Cell:
    """Cell of Sr2TiO4 over the steps of the showcase relaxation.

    The last step is the relaxed cell and the earlier ones are compressed. This is the
    single owner of the showcase cell: the structure and the k points have to agree on
    it, because the file stores one cell for the whole calculation and the first write to
    a path is the one that lands.
    """
    scaling = showcase.converge(INITIAL_COMPRESSION, 1.0)
    return raw.Cell(
        lattice_vectors=_demo.wrap_data(np.multiply.outer(scaling, lattice_vectors())),
        scale=raw.VaspData(1.0),
    )


def Fe3O4() -> raw.Cell:
    """Cell of magnetite over the steps of the showcase relaxation.

    The primitive cell of the face-centred cubic spinel lattice, which holds a quarter of
    the conventional cubic cell and therefore fourteen of its fifty-six atoms.
    """
    return _face_centred_cubic(FE3O4_LATTICE_CONSTANT)


def Cu() -> raw.Cell:
    """Cell of copper over the steps of the showcase relaxation."""
    return _face_centred_cubic(CU_LATTICE_CONSTANT)


def _face_centred_cubic(lattice_constant) -> raw.Cell:
    lattice_vectors = lattice_constant / 2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    scaling = showcase.converge(INITIAL_COMPRESSION, 1.0)
    return raw.Cell(
        lattice_vectors=_demo.wrap_data(np.multiply.outer(scaling, lattice_vectors)),
        scale=raw.VaspData(1.0),
    )


def Graphite() -> raw.Cell:
    """Hexagonal cell of the graphite slab, with the vacuum along the third vector.

    The two in-plane vectors enclose 120 degrees, which is the setting that puts the
    carbon atoms of a layer on the two sublattices of a honeycomb at (0, 0) and
    (1/3, 2/3).
    """
    a, c = GRAPHITE_LATTICE_CONSTANT, GRAPHITE_HEIGHT
    lattice_vectors = [
        [a, 0.0, 0.0],
        [-a / 2, a * np.sqrt(3) / 2, 0.0],
        [0.0, 0.0, c],
    ]
    return raw.Cell(
        lattice_vectors=_demo.wrap_data(lattice_vectors), scale=raw.VaspData(1.0)
    )

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

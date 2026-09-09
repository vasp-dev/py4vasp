# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import raw

# lattice constants of the conventional I4/mmm cell of Sr2TiO4 in Angstrom. The values
# of py4vasp._demo.structure.Sr2TiO4 analyze to these; the literature reports a = 3.884
# and c = 12.60 for the K2NiF4-type structure.
SR2TIO4_A = 3.9277
SR2TIO4_C = 12.6828


def Sr2TiO4():
    """Primitive cell of body-centred tetragonal Sr2TiO4 in the standard setting.

    py4vasp._demo uses an equivalent but rotated basis. The standard setting is chosen
    here because the high-symmetry points of the Brillouin zone are tabulated for it, so
    the labels along the band-structure path are the real ones rather than invented.
    """
    a, c = SR2TIO4_A, SR2TIO4_C
    lattice_vectors = [
        [-a / 2, a / 2, c / 2],
        [a / 2, -a / 2, c / 2],
        [a / 2, a / 2, -c / 2],
    ]
    return raw.Cell(
        lattice_vectors=raw.VaspData(np.array(lattice_vectors)),
        scale=raw.VaspData(1.0),
    )

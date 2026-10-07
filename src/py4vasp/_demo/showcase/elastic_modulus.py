# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools

import numpy as np

from py4vasp import raw
from py4vasp._demo.showcase import structure
from py4vasp._util import convert

# Index of a pair of Cartesian directions in the Voigt order py4vasp prints the modulus
# in: xx, yy, zz, xy, yz, zx.
_VOIGT = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (0, 1): 3, (1, 2): 4, (0, 2): 5}

# Clamped-ion elastic constants of Sr2TiO4 in GPa, of the size a DFT calculation finds
# for this layered titanate: the crystal is softer along the stacking direction z than
# in the plane, and tetragonal, so x and y are equivalent.
_C11, _C33, _C12, _C13, _C44, _C66 = 309.0, 237.0, 115.0, 92.0, 66.0, 105.0
_CLAMPED_ION = (
    (_C11, _C12, _C13, 0.0, 0.0, 0.0),
    (_C12, _C11, _C13, 0.0, 0.0, 0.0),
    (_C13, _C13, _C33, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, _C66, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, _C44, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, _C44),
)
# Letting the ions relax under strain can only lower the energy, so the relaxed-ion
# modulus is the clamped-ion one minus a positive semidefinite contribution. In Sr2TiO4
# the TiO6 octahedra relax mostly against a strain along the stacking direction.
_IONIC_SOFTENING = (
    (12.0, -4.0, 6.0, 0.0, 0.0, 0.0),
    (-4.0, 12.0, 6.0, 0.0, 0.0, 0.0),
    (6.0, 6.0, 21.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 3.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 9.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 9.0),
)


def Sr2TiO4() -> raw.ElasticModulus:
    """Clamped- and relaxed-ion elastic modulus of the tetragonal showcase crystal."""
    clamped_ion = np.array(_CLAMPED_ION)
    relaxed_ion = clamped_ion - np.array(_IONIC_SOFTENING)
    return raw.ElasticModulus(
        clamped_ion=_to_cartesian(clamped_ion),
        relaxed_ion=_to_cartesian(relaxed_ion),
        structure=structure.Sr2TiO4(),
    )


def _to_cartesian(voigt_in_gpa):
    tensor = np.zeros((3, 3, 3, 3))
    for i, j, k, l in itertools.product(range(3), repeat=4):
        first = _VOIGT[tuple(sorted((i, j)))]
        second = _VOIGT[tuple(sorted((k, l)))]
        tensor[i, j, k, l] = voigt_in_gpa[first, second]
    return tensor / convert.KBAR_TO_GPA  # VASP reports the modulus in kBar

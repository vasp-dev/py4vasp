# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import raw
from py4vasp._demo import (
    band,
    born_effective_charge,
    cell,
    dispersion,
    kpoint,
    projector,
    stoichiometry,
    structure,
)

# constants for vector dimensions
AXES = 3
# constants for the shape of demo data
NUMBER_ATOMS = 7
NUMBER_BANDS = 3
NUMBER_STEPS = 4
# constants for the magnetic configuration
NONPOLARIZED = 1
COLLINEAR = 2
NONCOLLINEAR = 4


def wrap_data(data):
    return raw.VaspData(np.array(data))


def wrap_random_data(shape, present=True, seed=None):
    if present:
        rng = np.random.default_rng(seed)
        data = 10 * rng.standard_normal(shape)
        return raw.VaspData(data)
    else:
        return raw.VaspData(None)


def wrap_orbital_types(use_orbitals, orbital_types):
    if use_orbitals:
        return raw.VaspData(np.array(orbital_types.split(), dtype="S"))
    else:
        return raw.VaspData(None)

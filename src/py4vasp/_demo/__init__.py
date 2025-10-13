# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import raw
from py4vasp._demo import band, cell, dispersion, kpoint, projector, stoichiometry

NUMBER_STEPS = 4


def wrap_data(data):
    return raw.VaspData(np.array(data))


def wrap_orbital_types(use_orbitals, orbital_types):
    if use_orbitals:
        return raw.VaspData(np.array(orbital_types.split(), dtype="S"))
    else:
        return raw.VaspData(None)

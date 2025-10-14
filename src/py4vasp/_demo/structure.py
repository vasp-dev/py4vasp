# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def Sr2TiO4(has_ion_types=True, number_steps=1):
    repetitions = (number_steps, 1, 1)
    positions = [
        [0.64529, 0.64529, 0.0],
        [0.35471, 0.35471, 0.0],
        [0.00000, 0.00000, 0.0],
        [0.84178, 0.84178, 0.0],
        [0.15823, 0.15823, 0.0],
        [0.50000, 0.00000, 0.5],
        [0.00000, 0.50000, 0.5],
    ]
    return raw.Structure(
        stoichiometry=_demo.stoichiometry.Sr2TiO4(has_ion_types),
        cell=_demo.cell.Sr2TiO4(),
        positions=np.tile(positions, repetitions),
    )

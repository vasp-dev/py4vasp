# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp import _demo, raw


def Sr2TiO4():
    shape = (_demo.NUMBER_EXCITONS, *_demo.GRID_DIMENSIONS)
    exciton_charge = _demo.wrap_random_data(shape)
    return raw.ExcitonDensity(
        structure=_demo.structure.Sr2TiO4(), exciton_charge=exciton_charge
    )

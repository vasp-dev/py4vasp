# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def Sr2TiO4(use_selective_dynamics):
    number_components = _demo.AXES * _demo.NUMBER_ATOMS
    shape = (number_components, number_components)
    force_constants = _demo.wrap_random_data(shape, seed=51609352)
    if use_selective_dynamics:
        mask = 3 * [True] + 5 * [False] + 5 * [True] + 6 * [False] + 2 * [True]
        force_constants = force_constants[mask][:, mask]
        selective_dynamics = np.reshape(mask, (_demo.NUMBER_ATOMS, _demo.AXES))
    else:
        selective_dynamics = _demo.wrap_random_data(None, present=False)
    return raw.ForceConstant(
        structure=_demo.structure.Sr2TiO4(),
        force_constants=0.5 * (force_constants + force_constants[:].T),
        selective_dynamics=selective_dynamics,
    )

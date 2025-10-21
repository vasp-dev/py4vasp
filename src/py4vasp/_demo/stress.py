# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def Sr2TiO4(randomize):
    shape = (_demo.NUMBER_STEPS, _demo.AXES, _demo.AXES)
    if randomize:
        stresses = np.random.random(shape)
    else:
        stresses = np.arange(np.prod(shape)).reshape(shape)
    return raw.Stress(structure=_demo.structure.Sr2TiO4(), stress=stresses)


def Fe3O4(randomize):
    shape = (_demo.NUMBER_STEPS, _demo.AXES, _demo.AXES)
    if randomize:
        stresses = np.random.random(shape)
    else:
        stresses = np.arange(np.prod(shape)).reshape(shape)
    return raw.Stress(
        structure=_demo.structure.Fe3O4(),
        stress=stresses,
    )

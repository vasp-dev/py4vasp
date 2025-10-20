# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def Sr2TiO4():
    shape = (_demo.NUMBER_ATOMS, _demo.AXES, _demo.AXES)
    return raw.BornEffectiveCharge(
        structure=_demo.structure.Sr2TiO4(),
        charge_tensors=np.arange(np.prod(shape)).reshape(shape),
    )

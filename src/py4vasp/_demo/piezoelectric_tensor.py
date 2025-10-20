# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def piezoelectric_tensor():
    shape = (2, _demo.AXES, _demo.AXES, _demo.AXES)
    data = np.arange(np.prod(shape)).reshape(shape)
    return raw.PiezoelectricTensor(electron=data[0], ion=data[1])

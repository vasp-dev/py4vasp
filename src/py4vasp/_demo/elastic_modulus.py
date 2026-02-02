# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def elastic_modulus(selection="dft"):
    structure = None if (selection != "dft with structure") else _demo.structure.Fe3O4()
    shape = (2, _demo.AXES, _demo.AXES, _demo.AXES, _demo.AXES)
    data = np.arange(np.prod(shape)).reshape(shape) + 1.0
    return raw.ElasticModulus(clamped_ion=data[0], relaxed_ion=data[1], structure=structure)

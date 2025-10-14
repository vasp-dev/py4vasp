# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def electron():
    shape = (_demo.AXES, _demo.AXES, _demo.NUMBER_POINTS, _demo.COMPLEX)
    return raw.DielectricFunction(
        energies=np.linspace(0, 1, _demo.NUMBER_POINTS),
        dielectric_function=_demo.wrap_random_data(shape),
        current_current=_demo.wrap_random_data(shape),
    )


def ionic():
    shape = (_demo.AXES, _demo.AXES, _demo.NUMBER_POINTS, _demo.COMPLEX)
    return raw.DielectricFunction(
        energies=np.linspace(0, 1, _demo.NUMBER_POINTS),
        dielectric_function=_demo.wrap_random_data(shape),
        current_current=raw.VaspData(None),
    )

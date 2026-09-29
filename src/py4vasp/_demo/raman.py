# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def Sr2TiO4():
    shape = (_demo.NUMBER_MODES, _demo.AXES, _demo.AXES, _demo.NUMBER_POINTS)
    tensor = np.array(_demo.wrap_random_data(shape + (_demo.COMPLEX,)))
    return raw.Raman(
        frequencies=_demo.wrap_data(np.linspace(0, 800, _demo.NUMBER_MODES)),
        energies=_demo.wrap_data(np.linspace(0, 1, _demo.NUMBER_POINTS)),
        raman_tensor=_demo.wrap_data(_symmetrize(tensor)),
    )


def _symmetrize(tensor):
    # VASP writes a tensor that is symmetric in its two directions and py4vasp relies on
    # it, so random data has to be symmetrized to be a valid input rather than a trap
    return 0.5 * (tensor + np.swapaxes(tensor, 1, 2))

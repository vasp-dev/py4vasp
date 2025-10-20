# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import raw


def polarization():
    return raw.Polarization(electron=np.array((1, 2, 3)), ion=np.array((4, 5, 6)))

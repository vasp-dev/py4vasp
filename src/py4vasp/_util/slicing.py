# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

INDICES = {"a": 0, "b": 1, "c": 2}


def plane(cell, cut):
    index = INDICES[cut]
    return np.delete(np.delete(cell, index, axis=0), index, axis=1)

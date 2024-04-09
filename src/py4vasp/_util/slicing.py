# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

INDICES = {"a": 0, "b": 1, "c": 2}


def plane(cell, cut):
    remaining_vectors = np.delete(cell, INDICES[cut], axis=0)
    for i in range(3):
        if np.allclose(remaining_vectors[:, i], 0):
            return np.delete(remaining_vectors, i, axis=1)
    raise NotImplementedError()

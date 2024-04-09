# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

INDICES = {"a": 0, "b": 1, "c": 2}


def plane(cell, cut):
    old_vectors = np.delete(cell, INDICES[cut], axis=0)
    for i in range(3):
        if np.allclose(old_vectors[:, i], 0):
            return np.delete(old_vectors, i, axis=1)
    U, S, _ = np.linalg.svd(old_vectors)
    new_vectors = U @ np.diag(S)
    if np.linalg.det(new_vectors) < 0:
        new_vectors = new_vectors[:, ::-1]
    return new_vectors

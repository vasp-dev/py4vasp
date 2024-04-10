# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

INDICES = {"a": 0, "b": 1, "c": 2}


def plane(cell, cut):
    # old_det = np.linalg.det(cell)
    old_vectors = np.delete(cell, INDICES[cut], axis=0)
    # expected = old_vectors @ old_vectors.T
    # print(old_vectors[0])
    # print(old_vectors[1])
    # print(expected)
    # print(np.cross(old_vectors[0], old_vectors[1]))
    # for i in range(3):
    #     if np.allclose(old_vectors[:, i], 0):
    #         return np.delete(old_vectors, i, axis=1)
    U, S, Vh = np.linalg.svd(old_vectors, full_matrices=False)
    print(old_vectors)
    print(Vh)
    print("new", old_vectors @ Vh.T)
    new_vectors = U @ np.diag(S)
    print(new_vectors)
    print(U)
    # if np.abs(U[0,0]) < np.abs(U[1,0]):
    #     new_vectors = new_vectors[:, ::-1]
    print(new_vectors)
    # actual = new_vectors @ new_vectors.T
    # print(new_vectors[0])
    # print(new_vectors[1])
    # print(np.linalg.det(new_vectors))
    # print(actual)
    # print(np.allclose(actual, expected))
    # if not np.allclose(actual, expected):
    # # print(new_vectors)
    # # print(np.cross(*new_vectors))
    # # if np.linalg.det(new_vectors) < 0:
    # print(new_vectors)
    return new_vectors

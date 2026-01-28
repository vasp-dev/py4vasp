# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np


def symmetry_reduce(tensor):
    if tensor is None:
        return None
    symmetry_reduced_tensor = [
        tensor[0, 0],
        tensor[1, 1],
        tensor[2, 2],
        0.5 * (tensor[0, 1] + tensor[1, 0]),
        0.5 * (tensor[1, 2] + tensor[2, 1]),
        0.5 * (tensor[0, 2] + tensor[2, 0]),
    ]
    return np.array(symmetry_reduced_tensor)


def tensor_constants(tensor):
    if tensor is None:
        return None, None
    eigenvalues, _ = np.linalg.eig(tensor)
    anisotropic_constants = [float(ev) for ev in eigenvalues]
    isotropic_constant = np.trace(np.diag(eigenvalues)) / float(len(eigenvalues))
    return float(isotropic_constant), anisotropic_constants

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import raw


def electronic_minimization():
    random_convergence_data = np.random.rand(9, 3)
    iteration_number = np.arange(1, 10)[:, np.newaxis]
    ncg = np.random.randint(4, 10, (9, 1))
    random_rms = np.random.rand(9, 2)
    convergence_data = np.hstack(
        [iteration_number, random_convergence_data, ncg, random_rms]
    )
    convergence_data = raw.VaspData(convergence_data)
    label = raw.VaspData([b"N", b"E", b"dE", b"deps", b"ncg", b"rms", b"rms(c)"])
    is_elmin_converged = [0]
    return raw.ElectronicMinimization(
        convergence_data=convergence_data,
        label=label,
        is_elmin_converged=is_elmin_converged,
    )

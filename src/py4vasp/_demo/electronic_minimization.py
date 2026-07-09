# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import raw


def electronic_minimization():
    random_convergence_data = np.random.rand(9, 3)
    # the total energy converges downward toward its final value, so the distance
    # E - E_final stays positive throughout the minimization
    random_convergence_data[:, 0] = -8.0 - np.linspace(0.0, 1.0, 9)
    iteration_number = np.arange(1, 10)[:, np.newaxis]
    ncg = np.random.randint(4, 10, (9, 1))
    random_rms = np.random.rand(9, 2)
    # density updates (and hence rms(c)) only start after the NELMDL delay, so the
    # first few electronic steps report a value of zero for this column
    random_rms[:5, 1] = 0.0
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

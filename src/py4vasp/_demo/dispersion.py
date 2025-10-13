# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def single_band():
    kpoints = _demo.kpoint.grid("explicit", "no_labels")
    eigenvalues = np.array([np.linspace([0], [1], len(kpoints.coordinates))])
    return raw.Dispersion(kpoints, eigenvalues)

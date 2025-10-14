# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def Sr2TiO4():
    frequencies = np.sqrt(
        np.linspace(0.1, -0.02, _demo.NUMBER_MODES, dtype=np.complex128)
    )
    return raw.PhononMode(
        structure=_demo.structure.Sr2TiO4(),
        frequencies=frequencies.view(np.float64).reshape(-1, 2),
        eigenvectors=_make_unitary_matrix(_demo.NUMBER_MODES),
    )


def _make_unitary_matrix(n, seed=None):
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((n, n))
    unitary_matrix, _ = np.linalg.qr(matrix)
    return raw.VaspData(unitary_matrix)

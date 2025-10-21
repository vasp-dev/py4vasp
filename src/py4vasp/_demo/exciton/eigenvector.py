# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def Sr2TiO4():
    dispersion = _demo.dispersion.multiple_bands()
    number_kpoints = len(dispersion.kpoints.coordinates)
    shape = (
        _demo.NONPOLARIZED,
        number_kpoints,
        _demo.NUMBER_CONDUCTION_BANDS,
        _demo.NUMBER_VALENCE_BANDS,
    )
    bse_index = np.arange(np.prod(shape)).reshape(shape)
    number_transitions = bse_index.size
    shape = (_demo.NUMBER_EIGENVECTORS, number_transitions, _demo.COMPLEX)
    eigenvectors = np.random.uniform(0, 20, shape)
    return raw.ExcitonEigenvector(
        dispersion=dispersion,
        fermi_energy=0.2,
        bse_index=raw.VaspData(bse_index),
        eigenvectors=raw.VaspData(eigenvectors),
        first_valence_band=raw.VaspData(np.array([1])),
        first_conduction_band=raw.VaspData(np.array([3])),
    )

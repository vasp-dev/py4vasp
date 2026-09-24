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


def dispersion():
    """The modes of a phonon dispersion, defined on the primitive cell.

    VASP reports these as real frequencies in THz for every **q** point of the path,
    where the modes of a linear response calculation are complex energies at the zone
    centre only.
    """
    qpoints = _demo.kpoint.qpoints()
    number_qpoints = len(qpoints.coordinates)
    shape = (
        number_qpoints,
        _demo.NUMBER_MODES,
        _demo.NUMBER_ATOMS,
        _demo.AXES,
        _demo.COMPLEX,
    )
    frequencies = np.linspace(-2, 20, number_qpoints * _demo.NUMBER_MODES)
    return raw.PhononMode(
        structure=_primitive_structure(),
        frequencies=_demo.wrap_data(frequencies.reshape(shape[:2])),
        eigenvectors=_demo.wrap_data(np.linspace(0, 1, np.prod(shape)).reshape(shape)),
        qpoints=qpoints,
    )


def _primitive_structure():
    # the primitive cell of a phonon calculation is a single frame, not a trajectory
    positions = np.linspace(0, 1, _demo.NUMBER_ATOMS * _demo.AXES)
    return raw.Structure(
        stoichiometry=_demo.stoichiometry.Sr2TiO4(),
        cell=_demo.cell.Sr2TiO4(),
        positions=_demo.wrap_data(positions.reshape(_demo.NUMBER_ATOMS, _demo.AXES)),
    )


def _make_unitary_matrix(n, seed=None):
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((n, n))
    unitary_matrix, _ = np.linalg.qr(matrix)
    return raw.VaspData(unitary_matrix)

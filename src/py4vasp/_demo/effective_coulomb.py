# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def crpa(two_center):
    positions = _setup_positions(two_center)
    shape_C = (_demo.COLLINEAR, _demo.COLLINEAR, _demo.NUMBER_WANNIER**4, _demo.COMPLEX)
    if two_center:
        shape_V = shape_C[:-1] + (len(positions), _demo.COMPLEX)
    else:
        shape_V = shape_C
    shape_U = shape_V
    cell = _demo.cell.Fe3O4()
    cell.lattice_vectors = cell.lattice_vectors[0]
    return raw.EffectiveCoulomb(
        number_wannier_states=_demo.NUMBER_WANNIER,
        frequencies=_demo.wrap_data([[0.0, 0.0]]),
        bare_potential_high_cutoff=_demo.wrap_random_data(shape_V),
        bare_potential_low_cutoff=_demo.wrap_random_data(shape_C),
        screened_potential=_demo.wrap_random_data(shape_U),
        cell=cell,
        positions=_setup_positions(two_center),
    )


def crpar(two_center):
    positions = _setup_positions(two_center)
    frequencies = np.array(
        (np.zeros(_demo.NUMBER_OMEGA), np.linspace(0, 10, _demo.NUMBER_OMEGA))
    ).T.copy()
    shape_C = (
        _demo.NONPOLARIZED,
        _demo.NONPOLARIZED,
        _demo.NUMBER_WANNIER**4,
        _demo.COMPLEX,
    )
    if two_center:
        shape_V = shape_C[:-1] + (len(positions), _demo.COMPLEX)
    else:
        shape_V = shape_C
    shape_U = (_demo.NUMBER_OMEGA,) + shape_V
    cell = _demo.cell.Sr2TiO4()
    cell.lattice_vectors = cell.lattice_vectors[0]
    return raw.EffectiveCoulomb(
        number_wannier_states=_demo.NUMBER_WANNIER,
        frequencies=_demo.wrap_data(frequencies),
        bare_potential_high_cutoff=_demo.wrap_random_data(shape_V),
        bare_potential_low_cutoff=_demo.wrap_random_data(shape_C),
        screened_potential=_demo.wrap_random_data(shape_U),
        cell=cell,
        positions=_setup_positions(two_center),
    )


def _setup_positions(two_center):
    if two_center:
        return _demo.wrap_data(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ]
        )
    else:
        return raw.VaspData(None)

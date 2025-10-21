# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools

import numpy as np

from py4vasp import _demo, raw


def self_energy(selection):
    band_start = 1
    # mock band_kpoint_spin_index array
    number_kpoints = 2
    band_kpoint_spin_index_shape = [
        _demo.NONPOLARIZED,
        number_kpoints,
        _demo.NUMBER_BANDS,
    ]
    band_kpoint_spin_index = np.full(band_kpoint_spin_index_shape, -1)
    spin = 0
    index_ = 0
    kpoints_bands = itertools.product(range(number_kpoints), range(_demo.NUMBER_BANDS))
    for kpoint, band in kpoints_bands:
        if kpoint == band:
            continue  # create a gap in the band_kpoint_spin_index
        index_ += 1
        band_kpoint_spin_index[spin, kpoint, band] = index_
    # mock fan and dw
    number_indices = np.count_nonzero(band_kpoint_spin_index != -1)
    fan_shape = [
        number_indices,
        _demo.NUMBER_FREQUENCIES,
        _demo.NUMBER_TEMPERATURES,
        _demo.COMPLEX,
    ]
    debye_waller_shape = [number_indices, _demo.NUMBER_TEMPERATURES]
    energies_shape = [number_indices, _demo.NUMBER_FREQUENCIES]
    temperatures_shape = [_demo.NUMBER_SAMPLES, _demo.NUMBER_TEMPERATURES]
    return raw.ElectronPhononSelfEnergy(
        valid_indices=range(_demo.NUMBER_SAMPLES),
        nbands_sum=_demo.electron_phonon.wrap_nbands_sum(selection),
        delta=_demo.electron_phonon.wrap_delta(selection, seed=18573411),
        scattering_approximation=_demo.electron_phonon.wrap_scattering_approximation(),
        chemical_potential=_demo.electron_phonon.chemical_potential.chemical_potential(),
        id_index=_demo.electron_phonon.wrap_id_index(),
        eigenvalues=_demo.wrap_random_data(band_kpoint_spin_index_shape),
        temperatures=_demo.wrap_random_data(temperatures_shape),
        debye_waller=[
            _demo.wrap_random_data(debye_waller_shape)
            for _ in range(_demo.NUMBER_SAMPLES)
        ],
        fan=[_demo.wrap_random_data(fan_shape) for _ in range(_demo.NUMBER_SAMPLES)],
        energies=[
            _demo.wrap_random_data(energies_shape) for _ in range(_demo.NUMBER_SAMPLES)
        ],
        band_kpoint_spin_index=[
            band_kpoint_spin_index for _ in range(_demo.NUMBER_SAMPLES)
        ],
        band_start=[band_start for _ in range(_demo.NUMBER_SAMPLES)],
    )

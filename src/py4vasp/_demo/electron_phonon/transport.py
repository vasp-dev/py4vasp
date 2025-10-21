# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def transport(selection):
    temperature_mesh = np.linspace(0, 500, _demo.NUMBER_TEMPERATURES)
    transport_shape = [
        _demo.NUMBER_SAMPLES,
        _demo.NUMBER_TEMPERATURES,
        _demo.NUMBER_FREQUENCIES,
        _demo.AXES,
        _demo.AXES,
    ]
    mobility_shape = [
        _demo.NUMBER_SAMPLES,
        _demo.NUMBER_TEMPERATURES,
        _demo.AXES,
        _demo.AXES,
    ]
    scattering_approx = _demo.electron_phonon.wrap_scattering_approximation(selection)
    return raw.ElectronPhononTransport(
        valid_indices=range(_demo.NUMBER_SAMPLES),
        nbands_sum=_demo.electron_phonon.wrap_nbands_sum(selection),
        chemical_potential=_demo.electron_phonon.chemical_potential.chemical_potential(),
        id_index=_demo.electron_phonon.wrap_id_index(),
        delta=_demo.electron_phonon.wrap_delta(selection, seed=733144842),
        temperatures=[temperature_mesh for _ in range(_demo.NUMBER_SAMPLES)],
        transport_function=_demo.wrap_random_data(transport_shape),
        mobility=_demo.wrap_random_data(mobility_shape),
        seebeck=_demo.wrap_random_data(mobility_shape),
        peltier=_demo.wrap_random_data(mobility_shape),
        electronic_conductivity=_demo.wrap_random_data(mobility_shape),
        electronic_thermal_conductivity=_demo.wrap_random_data(mobility_shape),
        scattering_approximation=scattering_approx,
    )

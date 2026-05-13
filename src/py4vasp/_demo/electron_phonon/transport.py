# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def transport(selection):
    temperature_mesh = np.linspace(0, 500, _demo.NUMBER_TEMPERATURES)
    is_spin = selection == "spin"
    spin_dims = [_demo.COLLINEAR] if is_spin else []
    transport_shape = [
        _demo.NUMBER_SAMPLES,
        _demo.NUMBER_TEMPERATURES,
        *spin_dims,
        _demo.NUMBER_FREQUENCIES,
        _demo.AXES,
        _demo.AXES,
    ]
    spin_tensor_shape = [
        _demo.NUMBER_SAMPLES,
        _demo.NUMBER_TEMPERATURES,
        *spin_dims,
        _demo.AXES,
        _demo.AXES,
    ]
    mobility_shape = [
        _demo.NUMBER_SAMPLES,
        _demo.NUMBER_TEMPERATURES,
        _demo.AXES,
        _demo.AXES,
    ]
    base_selection = "default" if is_spin else selection
    scattering_approx = _demo.electron_phonon.wrap_scattering_approximation(
        base_selection
    )
    return raw.ElectronPhononTransport(
        valid_indices=range(_demo.NUMBER_SAMPLES),
        nbands_sum=_demo.electron_phonon.wrap_nbands_sum(base_selection),
        chemical_potential=_demo.electron_phonon.chemical_potential.chemical_potential(),
        id_index=_demo.electron_phonon.wrap_id_index(),
        delta=_demo.electron_phonon.wrap_delta(base_selection, seed=733144842),
        temperatures=[temperature_mesh for _ in range(_demo.NUMBER_SAMPLES)],
        transport_function=_demo.wrap_random_data(transport_shape),
        mobility=_demo.wrap_random_data(mobility_shape),
        seebeck=_demo.wrap_random_data(spin_tensor_shape),
        peltier=_demo.wrap_random_data(spin_tensor_shape),
        electronic_conductivity=_demo.wrap_random_data(spin_tensor_shape),
        electronic_thermal_conductivity=_demo.wrap_random_data(spin_tensor_shape),
        scattering_approximation=scattering_approx,
    )

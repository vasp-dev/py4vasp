# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp import _demo, raw


def bandgap(selection):
    number_components = 3 if selection == "collinear" else 1
    shape_gap = [_demo.NUMBER_SAMPLES, number_components]
    shape_renorm = [_demo.NUMBER_SAMPLES, number_components, _demo.NUMBER_TEMPERATURES]
    shape_temperature = [_demo.NUMBER_SAMPLES, _demo.NUMBER_TEMPERATURES]
    scattering_approx = _demo.electron_phonon.wrap_scattering_approximation("bandgap")
    return raw.ElectronPhononBandgap(
        valid_indices=range(_demo.NUMBER_SAMPLES),
        nbands_sum=_demo.electron_phonon.wrap_nbands_sum(selection),
        delta=_demo.electron_phonon.wrap_delta(selection, seed=7824570),
        chemical_potential=_demo.electron_phonon.chemical_potential.chemical_potential(),
        scattering_approximation=scattering_approx,
        fundamental_renorm=_demo.wrap_random_data(shape_renorm),
        direct_renorm=_demo.wrap_random_data(shape_renorm),
        fundamental=_demo.wrap_random_data(shape_gap),
        direct=_demo.wrap_random_data(shape_gap),
        temperatures=_demo.wrap_random_data(shape_temperature),
        id_index=_demo.electron_phonon.wrap_id_index(),
    )

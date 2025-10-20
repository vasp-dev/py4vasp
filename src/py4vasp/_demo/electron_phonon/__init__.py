# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw
from py4vasp._demo.electron_phonon import (
    bandgap,
    chemical_potential,
    self_energy,
    transport,
)


def wrap_nbands_sum(selection):
    if selection == "CRTA":
        return [raw.VaspData(None) for _ in range(_demo.NUMBER_SAMPLES)]
    return _demo.wrap_data(np.linspace(10, 100, _demo.NUMBER_SAMPLES, dtype=np.int32))


def wrap_delta(selection, seed):
    if selection == "CRTA":
        return [raw.VaspData(None) for _ in range(_demo.NUMBER_SAMPLES)]
    return _demo.wrap_random_data([_demo.NUMBER_SAMPLES], seed=seed)


def wrap_id_index():
    unused = np.full(_demo.NUMBER_SAMPLES, fill_value=9999)
    index_chemical_potential = (
        np.arange(_demo.NUMBER_SAMPLES) % _demo.NUMBER_CHEMICAL_POTENTIALS
    )
    id_index = np.array([unused, unused, index_chemical_potential + 1, unused]).T
    return raw.VaspData(id_index)


def wrap_scattering_approximation(selection="default"):
    if selection == "bandgap":
        choices = ["SERTA", "SERTA", "MRTA_TAU", "SERTA", "SERTA"]
    elif selection == "CRTA":
        choices = ["CRTA" for _ in range(_demo.NUMBER_SAMPLES)]
    else:
        choices = ["SERTA", "ERTA_LAMDBA", "ERTA_TAU", "MRTA_LAMDBA", "MRTA_TAU"]
    return _demo.wrap_data(choices)

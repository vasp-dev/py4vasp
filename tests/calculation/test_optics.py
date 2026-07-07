# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._calculation import QUANTITIES, Calculation
from py4vasp._calculation.optics import Optics, OpticsHandler

HBAR_C = 1239.84  # eV·nm


def _reflectivity(eps):
    n = np.sqrt(eps)
    return np.abs((n - 1) / (n + 1)) ** 2


def _absorption(eps, energies):
    k = np.sqrt(eps).imag
    alpha = 2 * k * energies / HBAR_C
    return alpha / np.max(alpha)


def _transmission(eps, energies):
    return np.clip(1 - _reflectivity(eps) - _absorption(eps, energies), 0, 1)


def isotropic(tensor):
    return np.trace(tensor) / 3


def get_direction(tensor, direction):
    lookup = {"x": 0, "y": 1, "z": 2}
    i = lookup[direction[0]]
    j = lookup[direction[1]]
    return 0.5 * (tensor[i, j] + tensor[j, i])


@pytest.fixture
def electron(raw_data):
    raw_dielectric = raw_data.dielectric_function("electron")
    optics = Optics.from_data(raw_dielectric)
    optics.ref = types.SimpleNamespace()
    optics.ref.raw_data = raw_dielectric
    optics.ref.energies = raw_dielectric.energies
    to_complex = lambda data: data[..., 0] + 1j * data[..., 1]
    optics.ref.dielectric_function = to_complex(raw_dielectric.dielectric_function)
    return optics


def test_optics_is_registered_and_resolves():
    assert "optics" in QUANTITIES
    calc = Calculation.from_path(".")
    assert isinstance(calc.optics, Optics)


def test_from_data_creates_dispatcher(electron):
    assert isinstance(electron, Optics)

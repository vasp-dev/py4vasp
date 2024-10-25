# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp import calculation


@pytest.fixture
def self_energy(raw_data):
    raw_self_energy = raw_data.electron_phonon.self_energy("default")
    self_energy = calculation.electron_phonon.self_energy.from_data(raw_self_energy)
    self_energy.ref = types.SimpleNamespace()
    self_energy.ref.eigenvalues = raw_self_energy.eigenvalues
    self_energy.ref.debye_waller = raw_self_energy.debye_waller
    self_energy.ref.fan = raw_self_energy.fan
    return self_energy


@pytest.fixture(params=[None, 0, slice(1, 3), slice(None)])
def sample(request):
    return request.param


def test_read(self_energy, sample, Assert):
    if sample is None:
        slice_ = -1
        actual = self_energy.read()
    else:
        slice_ = sample
        actual = self_energy[sample].read()
    Assert.allclose(actual["eigenvalues"], self_energy.ref.eigenvalues)
    Assert.allclose(actual["debye_waller"], self_energy.ref.debye_waller[slice_])
    Assert.allclose(actual["fan"], self_energy.ref.fan[slice_])

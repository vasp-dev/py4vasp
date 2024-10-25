# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp import calculation


@pytest.fixture
def self_energy(raw_data):
    raw_self_energy = raw_data.electron_phonon.self_energy("default")
    self_energy = calculation.electron_phonon.self_energy.from_data(raw_self_energy)
    return self_energy


def test_read(self_energy):
    assert True

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest

from py4vasp._calculation.phonon_mode import PhononMode


@pytest.fixture
def phonon_mode(raw_data):
    raw_mode = raw_data.phonon_mode("default")
    mode = PhononMode.from_data(raw_mode)
    return mode


def test_read(phonon_mode):
    phonon_mode.read()

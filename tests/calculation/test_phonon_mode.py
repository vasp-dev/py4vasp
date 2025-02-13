# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp._calculation.phonon_mode import PhononMode
from py4vasp._calculation.structure import Structure


@pytest.fixture
def phonon_mode(raw_data):
    raw_mode = raw_data.phonon_mode("default")
    mode = PhononMode.from_data(raw_mode)
    mode.ref = types.SimpleNamespace()
    mode.ref.structure = Structure.from_data(raw_mode.structure)
    mode.ref.frequencies = raw_mode.frequencies
    mode.ref.eigenvectors = raw_mode.eigenvectors
    return mode


def test_read(phonon_mode, Assert):
    actual = phonon_mode.read()
    Assert.same_structure(actual["structure"], phonon_mode.ref.structure.read())
    Assert.allclose(actual["frequencies"], phonon_mode.ref.frequencies)
    Assert.allclose(actual["eigenvectors"], phonon_mode.ref.eigenvectors)


def test_frequencies(phonon_mode, Assert):
    Assert.allclose(phonon_mode.frequencies(), phonon_mode.ref.frequencies)


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.phonon_mode("Sr2TiO4")
    check_factory_methods(PhononMode, data)

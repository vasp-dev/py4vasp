# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp._calculation.effective_coulomb import EffectiveCoulomb


@pytest.fixture
def without_frequency(raw_data):
    raw_coulomb = raw_data.effective_coulomb("crpa")
    coulomb = EffectiveCoulomb.from_data(raw_coulomb)
    coulomb.ref = types.SimpleNamespace()
    return coulomb


@pytest.fixture
def with_frequency(raw_data):
    raw_coulomb = raw_data.effective_coulomb("crpar")
    coulomb = EffectiveCoulomb.from_data(raw_coulomb)
    coulomb.ref = types.SimpleNamespace()
    return coulomb


def test_read_without_frequency(without_frequency):
    without_frequency.read()


def test_read_with_frequency(with_frequency):
    with_frequency.read()

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp._calculation.effective_coulomb import EffectiveCoulomb


@pytest.fixture
def effective_coulomb(raw_data):
    raw_coulomb = raw_data.effective_coulomb("default")
    coulomb = EffectiveCoulomb.from_data(raw_coulomb)
    coulomb.ref = types.SimpleNamespace()
    return coulomb


def test_read(effective_coulomb):
    effective_coulomb.read()

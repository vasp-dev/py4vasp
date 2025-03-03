# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import dataclasses
import types

import numpy as np
import pytest

from py4vasp._calculation.nics import Nics


@pytest.fixture
def chemical_shift(raw_data):
    raw_nics = raw_data.nics("Sr2TiO4")
    nics = Nics.from_data(raw_nics)
    return nics


def test_read(chemical_shift, Assert):
    actual = chemical_shift.read()

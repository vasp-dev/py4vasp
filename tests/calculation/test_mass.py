# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest

import py4vasp
from py4vasp import Calculation, _calculation
from py4vasp._util import masses


def test_standard_atomic_weights_match_the_table(tmp_path):
    # the constant needs no VASP output, so an empty directory suffices
    calculation = Calculation.from_path(tmp_path)
    assert dict(calculation.mass.STANDARD_ATOMIC_WEIGHTS) == masses.TABLE


def test_default_calculation_exposes_standard_atomic_weights(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert py4vasp.calculation.mass.STANDARD_ATOMIC_WEIGHTS["O"] == 15.999


def test_standard_atomic_weights_are_read_only(tmp_path):
    # the table is the default every mass weighting falls back to
    weights = Calculation.from_path(tmp_path).mass.STANDARD_ATOMIC_WEIGHTS
    with pytest.raises(TypeError):
        weights["O"] = 18.0
    assert masses.TABLE["O"] == 15.999


def test_mass_is_not_a_registered_quantity():
    # a registered quantity has to implement read, print and selections
    assert "mass" not in _calculation.QUANTITIES

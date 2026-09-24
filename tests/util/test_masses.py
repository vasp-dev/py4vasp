# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp import exception
from py4vasp._util import masses


def test_masses_of_elements(Assert):
    actual = masses.of(["Ba", "Ti", "O", "O", "O"])
    Assert.allclose(actual, [137.327, 47.867, 15.999, 15.999, 15.999])


def test_masses_keep_the_order_of_the_elements(Assert):
    Assert.allclose(masses.of(["O", "Ti"]), np.flip(masses.of(["Ti", "O"])))


def test_every_element_has_a_plausible_mass():
    # a nucleus has at least as many nucleons as protons and at most roughly three
    # times as many, so this catches a misplaced decimal point or a shifted row
    for atomic_number, (element, mass) in enumerate(masses.TABLE.items(), start=1):
        assert atomic_number <= mass <= 2.7 * atomic_number, element


def test_table_covers_the_elements_VASP_provides_potentials_for():
    assert set(("H", "C", "O", "Fe", "Sr", "Ba", "U", "Pu")) <= set(masses.TABLE)


def test_unknown_element_raises_error():
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.of(["Sr", "Xx"])
    assert "Xx" in str(error.value)

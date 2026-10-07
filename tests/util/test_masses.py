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


ELEMENTS = ["Sr", "Ti", "O"]


def test_resolve_defaults_to_the_standard_atomic_weights(Assert):
    Assert.allclose(masses.resolve(None, ELEMENTS), masses.of(ELEMENTS))


def test_resolve_keeps_the_masses_the_user_provides(Assert):
    Assert.allclose(masses.resolve([88.0, 48.0, 16.0], ELEMENTS), [88.0, 48.0, 16.0])


def test_resolve_raises_error_if_masses_do_not_match_the_atoms():
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve([1.0, 2.0], ELEMENTS)
    assert "2" in str(error.value) and "3" in str(error.value)


def test_resolve_raises_error_if_masses_are_not_numbers():
    # a dictionary of element to mass is a plausible guess and would otherwise be
    # reported as a single mass rather than as the wrong kind of input
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve({"Sr": 87.62, "Ti": 47.867, "O": 15.999}, ELEMENTS)
    assert "numbers" in str(error.value)


@pytest.mark.parametrize("wrong_mass", (0.0, -1.0))
def test_resolve_raises_error_for_nonpositive_mass(wrong_mass):
    # dividing by the square root of the mass would fill the result with nan
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve([88.0, wrong_mass, 16.0], ELEMENTS)
    assert "positive" in str(error.value)

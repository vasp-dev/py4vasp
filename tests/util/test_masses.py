# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import fractions

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


def test_resolve_overrides_the_elements_in_a_mapping(Assert):
    expected = [87.62, 47.867, 18.0]
    Assert.allclose(masses.resolve({"O": 18.0}, ELEMENTS), expected)


def test_resolve_mapping_applies_to_every_atom_of_the_element(Assert):
    elements = ["Sr", "O", "Ti", "O"]
    expected = [87.62, 18.0, 47.867, 18.0]
    Assert.allclose(masses.resolve({"O": 18.0}, elements), expected)


def test_resolve_raises_error_if_masses_do_not_match_the_atoms():
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve([1.0, 2.0], ELEMENTS)
    assert "2" in str(error.value) and "3" in str(error.value)


def test_resolve_raises_error_if_masses_are_not_numbers():
    # passing the elements instead of their masses would otherwise be reported as the
    # wrong number of masses rather than as the wrong kind of input
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve(["Sr", "Ti", "O"], ELEMENTS)
    assert "numbers" in str(error.value)


@pytest.mark.parametrize("wrong_mass", (0.0, -1.0))
def test_resolve_raises_error_for_nonpositive_mass(wrong_mass):
    # dividing by the square root of the mass would fill the result with nan
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve([88.0, wrong_mass, 16.0], ELEMENTS)
    assert "positive" in str(error.value)


def test_resolve_error_uses_singular_for_one_mass():
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve(18.0, ELEMENTS)
    assert "1 mass but" in str(error.value)


def test_resolve_error_lists_masses_as_plain_numbers():
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve(np.array([88.0, 0.0, 16.0]), ELEMENTS)
    assert "[88.0, 0.0, 16.0]" in str(error.value)


@pytest.mark.parametrize("key", ("o", "D", "H"))
def test_resolve_mapping_raises_error_for_element_not_in_structure(key):
    # a misspelled element or an isotope label would otherwise be silently ignored
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve({key: 18.0}, ELEMENTS)
    assert repr(key) in str(error.value)
    assert all(element in str(error.value) for element in ELEMENTS)


def test_resolve_mapping_suggests_spelling_only_for_unknown_symbols():
    # hydrogen is a valid element that is merely absent, so a hint on how to spell
    # chemical symbols would send the user looking for the wrong mistake
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve({"H": 2.014}, ELEMENTS)
    assert "rather than" not in str(error.value)
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve({"o": 18.0}, ELEMENTS)
    assert "rather than" in str(error.value)


@pytest.mark.parametrize("wrong_mass", (0.0, -1.0, float("inf"), float("nan")))
def test_resolve_mapping_raises_error_for_nonpositive_mass(wrong_mass):
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve({"O": wrong_mass}, ELEMENTS)
    assert "positive" in str(error.value)
    # report what the user passed rather than the expanded per-atom list
    assert "'O'" in str(error.value) and "87.62" not in str(error.value)


@pytest.mark.parametrize("wrong_mass", (float("inf"), float("nan")))
def test_resolve_raises_error_for_infinite_mass(wrong_mass):
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve([88.0, wrong_mass, 16.0], ELEMENTS)
    assert "positive" in str(error.value)


@pytest.mark.parametrize("wrong_mass", ("heavy", True))
def test_resolve_mapping_raises_error_for_non_numeric_mass(wrong_mass):
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve({"O": wrong_mass}, ELEMENTS)
    assert "numbers" in str(error.value)


def test_resolve_mapping_converts_masses_to_float(Assert):
    actual = masses.resolve({"O": fractions.Fraction(18)}, ELEMENTS)
    assert actual.dtype == np.float64
    Assert.allclose(actual, [87.62, 47.867, 18.0])

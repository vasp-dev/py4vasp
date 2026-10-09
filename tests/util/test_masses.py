# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import fractions

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._calculation._stoichiometry import (
    StoichiometryHandler,
    raw_stoichiometry_from_elements,
)
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


def stoichiometry(elements=ELEMENTS):
    return StoichiometryHandler.from_data(raw_stoichiometry_from_elements(elements))


def test_resolve_defaults_to_the_standard_atomic_weights(Assert):
    Assert.allclose(masses.resolve(None, stoichiometry()), masses.of(ELEMENTS))


def test_resolve_overrides_the_elements_in_a_mapping(Assert):
    expected = [87.62, 47.867, 18.0]
    Assert.allclose(masses.resolve({"O": 18.0}, stoichiometry()), expected)


def test_resolve_mapping_applies_to_every_atom_of_the_element(Assert):
    elements = ["Sr", "O", "Ti", "O"]
    expected = [87.62, 18.0, 47.867, 18.0]
    Assert.allclose(masses.resolve({"O": 18.0}, stoichiometry(elements)), expected)


@pytest.mark.parametrize(
    "elements, expected",
    (
        (["Sr", "Ti", "O", "O"], [87.62, 47.867, 15.999, 18.0]),
        (["O", "Sr", "O", "O"], [15.999, 87.62, 15.999, 18.0]),
    ),
)
def test_resolve_mapping_overrides_a_single_atom(elements, expected, Assert):
    # the keys count atoms from 1 like the DOS selection, and the second structure
    # spreads the oxygen atoms so their indices cannot be merged into a slice
    Assert.allclose(masses.resolve({"4": 18.0}, stoichiometry(elements)), expected)


@pytest.mark.parametrize("key", ("2:3", "2 : 3"))
def test_resolve_mapping_overrides_a_range_of_atoms(key, Assert):
    elements = ["Sr", "Ti", "O", "O"]
    expected = [87.62, 18.0, 18.0, 15.999]
    Assert.allclose(masses.resolve({key: 18.0}, stoichiometry(elements)), expected)


@pytest.mark.parametrize(
    "masses_",
    (
        {"O": 17.0, "3:4": 18.0, "4": 19.0},
        {"4": 19.0, "3:4": 18.0, "O": 17.0},
        {"3:4": 18.0, "4": 19.0, "O": 17.0},
    ),
)
def test_resolve_mapping_the_more_specific_key_wins(masses_, Assert):
    # an atom is more specific than a range, which is more specific than an element,
    # whatever order the dictionary lists them in
    elements = ["Sr", "O", "O", "O"]
    expected = [87.62, 17.0, 18.0, 19.0]
    Assert.allclose(masses.resolve(masses_, stoichiometry(elements)), expected)


def test_resolve_mapping_raises_error_for_overlapping_ranges():
    elements = ["Sr", "O", "O", "O"]
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve({"2:3": 18.0, "3:4": 17.0}, stoichiometry(elements))
    assert "'2:3'" in str(error.value) and "'3:4'" in str(error.value)


@pytest.mark.parametrize("keys", (("O", " O"), ("4", " 4")))
def test_resolve_mapping_raises_error_for_the_same_key_written_twice(keys):
    # otherwise the order of the dictionary would decide which mass wins
    elements = ["Sr", "O", "O", "O"]
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve({keys[0]: 18.0, keys[1]: 17.0}, stoichiometry(elements))
    assert repr(keys[0]) in str(error.value) and repr(keys[1]) in str(error.value)


def test_resolve_mapping_accepts_overlapping_ranges_with_the_same_mass(Assert):
    elements = ["Sr", "O", "O", "O"]
    actual = masses.resolve({"2:3": 18.0, "3:4": 18.0}, stoichiometry(elements))
    Assert.allclose(actual, [87.62, 18.0, 18.0, 18.0])


@pytest.mark.parametrize(
    "masses_", ([88.0, 48.0, 16.0], np.ones(3), 18.0, "O", ["Sr", "Ti", "O"])
)
def test_resolve_raises_error_for_anything_but_a_mapping(masses_):
    # one mass per atom used to be accepted; point to the mapping that replaced it
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve(masses_, stoichiometry())
    assert '{"4": 17.999}' in str(error.value)


@pytest.mark.parametrize("key", ("o", "D", "H"))
def test_resolve_mapping_raises_error_for_element_not_in_structure(key):
    # a misspelled element or an isotope label would otherwise be silently ignored
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve({key: 18.0}, stoichiometry())
    assert repr(key) in str(error.value)
    assert all(element in str(error.value) for element in ELEMENTS)


def test_resolve_mapping_suggests_spelling_only_for_unknown_symbols():
    # hydrogen is a valid element that is merely absent, so a hint on how to spell
    # chemical symbols would send the user looking for the wrong mistake
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve({"H": 2.014}, stoichiometry())
    assert "rather than" not in str(error.value)
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve({"o": 18.0}, stoichiometry())
    assert "rather than" in str(error.value)


def test_resolve_mapping_suggests_a_close_key():
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve({"o": 18.0}, stoichiometry())
    assert 'Did you mean "O"?' in str(error.value)


def test_resolve_mapping_rejects_the_internal_key_for_all_atoms():
    # it is an implementation detail of the selection, not a key the docs offer
    with pytest.raises(exception.IncorrectUsage):
        masses.resolve({"__all__": 18.0}, stoichiometry())


def test_resolve_mapping_raises_error_for_atom_outside_the_structure():
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve({"9": 18.0}, stoichiometry())
    assert "'9'" in str(error.value) and "1 to 3" in str(error.value)
    assert "counted from 1" in str(error.value)
    # the key is an atom index, so a hint on chemical symbols would mislead
    assert "rather than" not in str(error.value)


@pytest.mark.parametrize("key", ("1 + 2", "Sr, 3", "1(2)", "", "1:5:2"))
def test_resolve_mapping_raises_error_for_more_than_one_selection(key):
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve({key: 18.0}, stoichiometry())
    assert repr(key) in str(error.value) and "own entry" in str(error.value)


@pytest.mark.parametrize("key", (4, 4.0, None))
def test_resolve_mapping_raises_error_for_keys_that_are_not_strings(key):
    # an integer key reads like a 0-based Python index, so ask for the string
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve({key: 18.0}, stoichiometry())
    assert '"4"' in str(error.value) and "counted from 1" in str(error.value)


@pytest.mark.parametrize("key", ("O:3", "Sr:Ti", "3:1"))
def test_resolve_mapping_raises_error_for_invalid_range(key):
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve({key: 18.0}, stoichiometry())
    assert repr(key) in str(error.value) and "range" in str(error.value)


@pytest.mark.parametrize("wrong_mass", (0.0, -1.0, float("inf"), float("nan")))
def test_resolve_mapping_raises_error_for_nonpositive_mass(wrong_mass):
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve({"O": wrong_mass}, stoichiometry())
    assert "positive" in str(error.value)
    # report what the user passed rather than the expanded per-atom list
    assert "'O'" in str(error.value) and "87.62" not in str(error.value)


@pytest.mark.parametrize("wrong_mass", ("heavy", True))
def test_resolve_mapping_raises_error_for_non_numeric_mass(wrong_mass):
    with pytest.raises(exception.IncorrectUsage) as error:
        masses.resolve({"O": wrong_mass}, stoichiometry())
    assert "numbers" in str(error.value)


def test_resolve_mapping_converts_masses_to_float(Assert):
    actual = masses.resolve({"O": fractions.Fraction(18)}, stoichiometry())
    assert actual.dtype == np.float64
    Assert.allclose(actual, [87.62, 47.867, 18.0])

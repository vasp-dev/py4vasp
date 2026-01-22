from pathlib import Path

import pytest

from py4vasp import demo
from py4vasp._raw.data import CalculationMetaData, _DatabaseData
from py4vasp._raw.definition import DEFAULT_SOURCE
from py4vasp._util import database


@pytest.mark.parametrize(
    ["key", "db_key_suffix", "group_name", "expected"],
    [
        ("group.quantity", ":selection", None, "group.quantity:selection"),
        ("group.quantity", None, None, "group.quantity"),
        ("group_quantity", None, "group", "group.quantity"),
        ("group_other_quantity", None, "group", "group.other_quantity"),
        (
            "group_other_quantity",
            ":selection",
            "group",
            "group.other_quantity:selection",
        ),
        (
            "group_other_quantity:selection",
            ":other_selection",
            "group",
            "group.other_quantity:selection",
        ),
        ("group.quantity:selection", None, None, "group.quantity:selection"),
        (
            "group.quantity:selection",
            ":other_selection",
            None,
            "group.quantity:selection",
        ),
    ],
)
def test_clean_db_key(key, db_key_suffix, group_name, expected):
    assert (
        database.clean_db_key(key, db_key_suffix=db_key_suffix, group_name=group_name)
        == expected
    )


def test_combine_db_dicts():
    dict1 = {
        "a": 1,
        "b": {"c": 2, "d": 3},
        "e": 4,
    }
    dict2 = {
        "b": {"c": 2, "f": 5},
        "g": 6,
    }
    combined = database.combine_db_dicts(dict1, dict2)
    expected = {
        "a": 1,
        "b": {"c": 2, "d": 3, "f": 5},
        "e": 4,
        "g": 6,
    }
    assert combined == expected


def test_construct_database_data_key():
    assert database.construct_database_data_key("group", "quantity", "selection") == (
        "group.quantity:selection",
        True,
    )
    assert database.construct_database_data_key(None, "quantity", None) == (
        "quantity",
        False,
    )
    assert database.construct_database_data_key("group", "quantity", None) == (
        "group.quantity",
        False,
    )
    assert database.construct_database_data_key(None, "quantity", "selection") == (
        "quantity:selection",
        True,
    )
    assert database.construct_database_data_key(
        "group", "quantity", DEFAULT_SOURCE
    ) == ("group.quantity", False)


@pytest.mark.parametrize(
    ["ion_numbers", "expected"],
    [
        ([4, 8, 4], [1, 2, 1]),
        ([8, 32, 8], [1, 4, 1]),
        ([10], [1]),
        ([2, 2, 4], [1, 1, 2]),
        ([4, 6, 8], [2, 3, 4]),
        ([3, 5, 7], [3, 5, 7]),
        ([3, 5, 6], [3, 5, 6]),
    ],
)
def test_get_primitive_ion_numbers(ion_numbers, expected):
    assert database.get_primitive_ion_numbers(ion_numbers) == expected


@pytest.mark.parametrize(
    ["ion_types", "ion_numbers", "expected_formula", "expected_compound"],
    [
        (["Si", "O"], [1, 2], "O2Si", "O-Si"),
        (["H", "O"], [4, 2], "H2O", "H-O"),
        (["Na", "Cl"], [1, 1], "ClNa", "Cl-Na"),
        (["C"], [1], "C", "C"),
        (["Fe", "O"], [2, 3], "Fe2O3", "Fe-O"),
        (["Al", "O", "Mg"], [2, 3, 6], "Al2Mg6O3", "Al-Mg-O"),
    ],
)
def test_get_formula_and_compound(
    ion_types, ion_numbers, expected_formula, expected_compound
):
    formula, compound = database.get_formula_and_compound(ion_types, ion_numbers)
    assert formula == expected_formula
    assert compound == expected_compound


def basic_db_checks(demo_calc_db: _DatabaseData, minimum_counter=1):
    assert demo_calc_db is not None
    assert isinstance(demo_calc_db, _DatabaseData)
    assert demo_calc_db.metadata is not None
    assert isinstance(demo_calc_db.metadata, CalculationMetaData)
    assert isinstance(demo_calc_db.available_quantities, dict)
    assert isinstance(demo_calc_db.additional_properties, dict)

    # Check metadata fields
    assert isinstance(demo_calc_db.metadata.hdf5_original_path, Path)

    # Check that available_quantities has correct structure
    # and that the loaded data is non-trivial
    true_counter = 0
    has_non_default_selections = False
    for key, value in demo_calc_db.available_quantities.items():
        available, aliases = value
        assert isinstance(available, bool)
        assert isinstance(aliases, list)
        assert len(aliases) >= 1
        assert isinstance(aliases[0], str)
        true_counter += int(available)
        if ":" in key and not key.endswith(f":{DEFAULT_SOURCE}"):
            has_non_default_selections = True
    assert has_non_default_selections
    assert true_counter > minimum_counter

    # Check that additional_properties has correct structure and
    # Check that additional_properties has only entries that are listed in available_quantities
    non_empty_counter = 0
    for key in demo_calc_db.additional_properties:
        if not (key in demo_calc_db.available_quantities):
            if not (key.startswith("cell")):
                raise AssertionError(
                    f"Key {key} in additional_properties missing from available_quantities"
                )
        elif not demo_calc_db.available_quantities[key][0]:
            raise AssertionError(
                f"Key {key} in additional_properties marked as unavailable in available_quantities"
            )

        if demo_calc_db.additional_properties[key] not in (None, {}, []):
            non_empty_counter += 1
    assert non_empty_counter > minimum_counter


@pytest.mark.parametrize(
    ["selection", "minimum_counter"],
    [(None, 5), ("collinear", 1), ("noncollinear", 5), ("spin_texture", 2)],
)
def test_demo_db(tmp_path, selection, minimum_counter):
    """Check basic _to_database functionality on demo calculation."""
    actual_path = tmp_path / "demo_calculation"
    demo_calc = demo.calculation(actual_path, selection=selection)
    demo_calc_db = demo_calc._to_database()
    basic_db_checks(demo_calc_db, minimum_counter=minimum_counter)


@pytest.mark.parametrize("tags", [None, "test", ["test", "demo"]])
def test_demo_db_with_tags(tags, tmp_path):
    """Check _to_database functionality with tags on demo calculation."""
    actual_path = tmp_path / "demo_calculation"
    demo_calc = demo.calculation(actual_path)
    demo_calc_db = demo_calc._to_database(tags=tags)
    basic_db_checks(demo_calc_db)
    assert demo_calc_db.metadata.tags == tags

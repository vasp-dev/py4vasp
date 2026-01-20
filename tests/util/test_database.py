from pathlib import Path

import pytest

from py4vasp import Calculation, demo
from py4vasp._raw.data import CalculationMetaData, _DatabaseData
from py4vasp._raw.definition import DEFAULT_SOURCE
from py4vasp._util import database


def test_clean_db_key():
    assert (
        database.clean_db_key("group.quantity", db_key_suffix=":selection")
        == "group.quantity:selection"
    )
    assert (
        database.clean_db_key("group.quantity", db_key_suffix=None) == "group.quantity"
    )
    assert (
        database.clean_db_key("group.quantity:selection", db_key_suffix=None)
        == "group.quantity:selection"
    )
    assert (
        database.clean_db_key(
            "group.quantity:selection", db_key_suffix=":other_selection"
        )
        == "group.quantity:selection"
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


def basic_db_checks(demo_calc_db: _DatabaseData, minimum_counter=1):
    assert demo_calc_db is not None
    assert isinstance(demo_calc_db, _DatabaseData)
    assert demo_calc_db.metadata is not None
    assert isinstance(demo_calc_db.metadata, CalculationMetaData)
    assert isinstance(demo_calc_db.available_quantities, dict)
    assert isinstance(demo_calc_db.additional_properties, dict)

    # Check metadata fields
    assert isinstance(demo_calc_db.metadata.hdf5, Path)

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
    ["selection", "minimum_counter"], [(None, 5), ("collinear", 1), ("noncollinear", 5), ("spin_texture", 2)]
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

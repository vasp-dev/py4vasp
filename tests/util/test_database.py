# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
from pathlib import Path

import pytest

from py4vasp import demo
from py4vasp._calculation import GROUPS, QUANTITIES
from py4vasp._raw.data import CalculationMetaData, _DatabaseData
from py4vasp._raw.data_db import _DBDataMixin
from py4vasp._raw.data_wrapper import VaspData
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
    ["ion_types", "ion_numbers", "expectations"],
    [
        (["Si", "O"], [1, 2], ("O2Si", "O-Si", ["O", "Si"], [2, 1], [2, 1])),
        (["H", "O"], [4, 2], ("H2O", "H-O", ["H", "O"], [4, 2], [2, 1])),
        (["Na", "Cl"], [1, 1], ("ClNa", "Cl-Na", ["Cl", "Na"], [1, 1], [1, 1])),
        (["C"], [1], ("C", "C", ["C"], [1], [1])),
        (["Fe", "O"], [2, 3], ("Fe2O3", "Fe-O", ["Fe", "O"], [2, 3], [2, 3])),
        (
            ["Al", "O", "Mg"],
            [2, 3, 6],
            ("Al2Mg6O3", "Al-Mg-O", ["Al", "Mg", "O"], [2, 6, 3], [2, 6, 3]),
        ),
        (
            ["Ca", "As", "Br", "Ca", "Br"],
            [2, 1, 1, 1, 2],
            ("AsBr3Ca3", "As-Br-Ca", ["As", "Br", "Ca"], [1, 3, 3], [1, 3, 3]),
        ),
        (
            ["Ca", "As", "Br", "Ca", "Br"],
            [2, 3, 1, 1, 2],
            ("AsBrCa", "As-Br-Ca", ["As", "Br", "Ca"], [3, 3, 3], [1, 1, 1]),
        ),
    ],
)
def test_get_formula_and_compound(ion_types, ion_numbers, expectations):
    (
        expected_formula,
        expected_compound,
        expected_simple_types,
        expected_simple_nums,
        expected_primitive_nums,
    ) = expectations
    formula, compound, simple_types, simple_nums, primitive_nums = (
        database.get_formula_and_compound(ion_types, ion_numbers)
    )
    assert formula == expected_formula
    assert compound == expected_compound
    assert simple_types == expected_simple_types
    assert simple_nums == expected_simple_nums
    assert primitive_nums == expected_primitive_nums


def test_get_all_possible_keys():
    """Test that get_all_possible_keys runs without error and returns a non-empty dict."""
    all_keys, output_type_dict = database.get_all_possible_keys(
        to_print=False, debug=False
    )
    assert isinstance(all_keys, dict)
    assert isinstance(output_type_dict, dict)
    assert len(all_keys) > 0
    assert len(output_type_dict) > 0
    for k in QUANTITIES:
        assert k in output_type_dict
        dataclass_name = output_type_dict[k]
        if dataclass_name is None:
            continue
        if dataclass_name.endswith("_DB"):
            assert dataclass_name in all_keys
            assert isinstance(all_keys[dataclass_name], list)
            if all_keys[dataclass_name]:
                name, dtype = all_keys[dataclass_name][0]
                assert isinstance(name, str)
                assert isinstance(dtype, str)
        else:
            assert isinstance(dataclass_name, str)
    for group, quantities in GROUPS.items():
        for quantity in quantities:
            grouped_key = f"{group}.{quantity}"
            assert grouped_key in output_type_dict
            dataclass_name = output_type_dict[grouped_key]
            if dataclass_name is None:
                continue
            if dataclass_name.endswith("_DB"):
                assert dataclass_name in all_keys
            else:
                assert isinstance(dataclass_name, str)

    # Validate explicit entries for non-default selections.
    assert output_type_dict["band"] == output_type_dict["band:kpoints_opt"]
    assert output_type_dict["band"] == output_type_dict["band:kpoints_wan"]
    assert "band:default" not in output_type_dict
    assert output_type_dict["current_density"] is None

    assert (
        sum([1 for v in all_keys.values() if len(v) > 0 and isinstance(v[0], tuple)])
        > 10
    )


def basic_db_checks(demo_calc_db: _DatabaseData, minimum_counter=1):
    assert demo_calc_db is not None
    assert isinstance(demo_calc_db, _DatabaseData)
    assert demo_calc_db.metadata is not None
    assert isinstance(demo_calc_db.metadata, CalculationMetaData)
    assert isinstance(demo_calc_db.properties, dict)

    # Check metadata fields
    assert isinstance(demo_calc_db.metadata.path, Path)

    # Check that properties has enough non-empty entries
    non_empty_counter = sum(
        1 for v in demo_calc_db.properties.values() if v not in (None, {}, [])
    )
    assert non_empty_counter > minimum_counter
    assert "run_info" in demo_calc_db.properties


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


def test_demo_db_no_longer_accepts_tags(tmp_path):
    """_to_database() no longer accepts tags or fermi_energy arguments."""
    import inspect

    from py4vasp._calculation import Calculation

    sig = inspect.signature(Calculation._to_database)
    params = [name for name in sig.parameters if name != "self"]
    assert params == []


def test_no_vaspdata_in_db():
    """Check that VaspData objects are converted to their underlying data under the _DBDataMixin for the database wrapper classes."""

    @dataclasses.dataclass
    class DummyClassDB(_DBDataMixin):
        field1: int = None
        field2: str = None
        field3: bool = None

    dummyClass = DummyClassDB(
        field1=VaspData(None), field2=VaspData("test"), field3=True
    )
    assert dummyClass.field1 is None
    assert dummyClass.field2 == "test"
    assert dummyClass.field3 is True

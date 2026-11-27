# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
import datetime
from types import MappingProxyType

import numpy as np
import pytest

from py4vasp._util.tree import to_tree


@dataclasses.dataclass
class Record:
    name: str
    values: list


def test_mapping_becomes_a_dictionary_with_text_keys():
    actual = to_tree(MappingProxyType({1: {"a": 2}}))
    assert actual == {"1": {"a": 2}}
    assert type(actual) is dict


def test_dataclass_becomes_a_dictionary_of_its_fields():
    actual = to_tree(Record("x", [1.5, 2.5]))
    assert list(actual) == ["name", "values"]
    assert actual["name"] == "x"
    assert np.array_equal(actual["values"], [1.5, 2.5])


@pytest.mark.parametrize(
    "value",
    (
        np.zeros((2, 3)),
        np.arange(4, dtype=np.int32),
        np.array([True, False]),
        np.array([1 + 2j]),
        np.array(["a", "b"]),
    ),
)
def test_array_of_numbers_booleans_or_text_passes_unchanged(value):
    assert to_tree(value) is value


@pytest.mark.parametrize(
    "value, expected",
    (
        ([1, 2], np.array([1, 2])),
        ((0.5, 1.5), np.array([0.5, 1.5])),
        ([["a", "b"], ["c", "d"]], np.array([["a", "b"], ["c", "d"]])),
        ([], np.array([])),
    ),
)
def test_list_or_tuple_becomes_an_array(value, expected):
    actual = to_tree(value)
    assert isinstance(actual, np.ndarray)
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected)


def test_bytes_become_text():
    assert to_tree(np.array([b"X", b"M"])).tolist() == ["X", "M"]
    assert to_tree([b"X", b"M"]).tolist() == ["X", "M"]
    assert to_tree(b"G") == "G"


def test_array_of_objects_is_retyped_from_its_values():
    actual = to_tree(np.array([b"up", b"down"], dtype=object))
    assert actual.dtype.kind == "U"
    assert actual.tolist() == ["up", "down"]


def test_elements_of_unequal_length_are_kept_by_index():
    actual = to_tree([[1, 2], [3]])
    assert list(actual) == ["0", "1"]
    assert np.array_equal(actual["0"], [1, 2])
    assert np.array_equal(actual["1"], [3])


@pytest.mark.parametrize("value", ([1, "a"], ("a", 2.5), [None, 1.0], [{"a": 1}]))
def test_elements_of_different_kinds_are_kept_by_index(value):
    actual = to_tree(value)
    assert list(actual) == [str(index) for index in range(len(value))]
    assert [to_tree(item) for item in value] == list(actual.values())


def test_array_of_objects_of_different_kinds_is_kept_by_index():
    actual = to_tree(np.array([1, "a"], dtype=object))
    assert actual == {"0": 1, "1": "a"}


@pytest.mark.parametrize(
    "value, expected",
    (
        (np.float64(1.5), 1.5),
        (np.int32(3), 3),
        (np.bool_(True), True),
        (np.array(2.0), 2.0),
        # a range of indices is written as the selections write atoms: from 1 and
        # including its end; a single index is the index alone
        (slice(0, 2), "1:2"),
        (slice(3, 7), "4:7"),
        (slice(0, 1), "1"),
        (slice(2, 3, 1), "3"),
        # without the length, other slices cannot be written that way, so they keep
        # the notation of Python, in brackets to tell them apart
        (slice(None, -1, 2), "[:-1:2]"),
        (slice(3, None), "[3:]"),
        (slice(-2, None), "[-2:]"),
        (slice(0, 6, 2), "[0:6:2]"),
        (slice(2, 2), "[2:2]"),
        (None, None),
        ("text", "text"),
    ),
)
def test_scalar_becomes_a_python_scalar(value, expected):
    actual = to_tree(value)
    assert actual == expected
    assert type(actual) is type(expected)


def test_any_other_object_becomes_its_text():
    assert to_tree(datetime.date(2026, 1, 2)) == "2026-01-02"
    assert to_tree(np.array(["2026-01-02"], dtype="datetime64[D]")).tolist() == [
        "2026-01-02"
    ]

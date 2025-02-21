# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest
from util import Mapping

from py4vasp import exception


@pytest.fixture
def range_mapping():
    return Mapping(range(2), "common", ["variable_1", "variable_2"])


@pytest.fixture
def dict_mapping():
    return Mapping(("a", "b", "c"), "common", ["foo", "bar", "baz"])


def test_access_range_mapping(range_mapping):
    assert range_mapping[0] == Mapping([0], "common", "variable_1")
    assert range_mapping[1] == Mapping([1], "common", "variable_2")


def test_access_dict_mapping(dict_mapping):
    assert dict_mapping["a"] == Mapping(["a"], "common", "foo")
    assert dict_mapping["b"] == Mapping(["b"], "common", "bar")
    assert dict_mapping["c"] == Mapping(["c"], "common", "baz")


@pytest.mark.parametrize("incorrect_index", (-1, 3, "foo", None, slice(None)))
def test_incorrect_range_index(range_mapping, incorrect_index):
    with pytest.raises(exception.IncorrectUsage):
        range_mapping[incorrect_index]


@pytest.mark.parametrize("incorrect_index", (1, "foo", None, slice(None)))
def test_incorrect_dict_index(dict_mapping, incorrect_index):
    with pytest.raises(exception.IncorrectUsage):
        dict_mapping[incorrect_index]

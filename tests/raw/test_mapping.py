# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest
from util import Mapping


@pytest.fixture
def range_mapping():
    return Mapping(range(2), "common", ["variable_1", "variable_2"])


@pytest.fixture
def dict_mapping():
    return Mapping(("a", "b", "c"), "common", ["foo", "bar", "baz"])


def test_access_range_mapping(range_mapping):
    pass


# def test_incorrect_index():
#     range_mapping[]

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp import exception
from py4vasp._util import index


@pytest.mark.parametrize("selection, indices", [("Sr", 1), ("Ti", 2), ("O", 3)])
def test_one_component(selection, indices):
    values = np.arange(10) ** 2
    map_ = {0: {"Sr": 1, "Ti": 2, "O": 3}}
    selector = index.Selector(map_, values)
    assert selector[(selection,)] == values[indices]


@pytest.mark.parametrize("selection, indices", [("A", [1, 2]), ("B", [3, 5, 7])])
def test_sum_over_selection(selection, indices):
    values = np.arange(10) ** 2
    map_ = {0: {"A": slice(1, 3), "B": slice(3, 8, 2)}}
    selector = index.Selector(map_, values)
    assert selector[(selection,)] == np.sum(values[indices])


@pytest.mark.parametrize("selection, indices", [("x", [7]), ("y", [2, 5])])
def test_select_one_of_two_components(selection, indices):
    values = np.arange(30).reshape((3, 10))
    map_ = {1: {"x": 7, "y": slice(2, 6, 3)}}
    selector = index.Selector(map_, values)
    assert np.all(selector[(selection,)] == np.sum(values[:, indices], axis=1))


@pytest.mark.parametrize(
    "selection, indices",
    [
        (("A",), (1, slice(None))),
        (("x",), (slice(None), 1)),
        (("B", "z"), (2, 3)),
        (("y", "C"), (3, 2)),
    ],
)
def test_select_two_of_two_components(selection, indices):
    values = np.arange(30).reshape((6, 5))
    map_ = {0: {"A": 1, "B": 2, "C": 3}, 1: {"x": 1, "y": 2, "z": 3}}
    selector = index.Selector(map_, values)
    assert selector[selection] == np.sum(values[indices])


def test_error_when_indices_are_not_int_or_slice():
    with pytest.raises(exception.NotImplemented):
        index.Selector({0: {"A": [1, 2]}}, None)

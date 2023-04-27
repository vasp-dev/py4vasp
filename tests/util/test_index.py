# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp import exception
from py4vasp._util import index, select


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


@pytest.mark.parametrize(
    "selection, expected",
    [
        (("A", "x"), np.array([[30, 34, 38, 42, 46], [270, 274, 278, 282, 286]])),
        (("A", "y"), np.array([[15, 17, 19, 21, 23], [135, 137, 139, 141, 143]])),
        (("B", "x"), np.array([[110, 114, 118, 122, 126], [350, 354, 358, 362, 366]])),
        (("B", "y"), np.array([[55, 57, 59, 61, 63], [175, 177, 179, 181, 183]])),
    ],
)
def test_select_two_of_four_components(selection, expected):
    values = np.arange(120).reshape((2, 3, 4, 5))
    map_ = {1: {"A": 0, "B": 1}, 2: {"x": slice(None), "y": slice(1, 3)}}
    selector = index.Selector(map_, values)
    assert np.all(selector[selection] == expected)


@pytest.mark.parametrize(
    "selection, indices",
    [
        ((select.Group(["1", "3"], select.range_separator),), slice(0, 3)),
        ((select.Group(["2", "6"], select.range_separator),), slice(1, 6)),
        ((select.Group(["4", "5"], select.range_separator),), slice(3, 5)),
    ],
)
def test_select_range(selection, indices):
    values = np.arange(10) ** 2
    map_ = {0: {"1": 0, "2": 1, "3": 2, "4": 3, "5": slice(4, 5), "6": slice(5, 6, 1)}}
    selector = index.Selector(map_, values)
    assert selector[selection] == np.sum(values[indices])


@pytest.mark.parametrize(
    "selection, indices",
    [
        (("total",), 0),
        ((select.Group(["A", "B"], select.pair_separator),), 1),
        ((select.Group(["B", "A"], select.pair_separator),), 1),
        ((select.Group(["C", "A"], select.pair_separator),), 2),
        ((select.Group(["B", "C"], select.pair_separator),), 3),
    ],
)
def test_select_pair(selection, indices):
    values = np.arange(10) ** 2
    map_ = {0: {"total": 0, "A~B": 1, "A~C": 2, "B~C": 3}}
    selector = index.Selector(map_, values)
    assert selector[selection] == np.sum(values[indices])


def test_error_when_duplicate_key():
    with pytest.raises(exception._Py4VaspInternalError):
        index.Selector({0: {"A": 1}, 1: {"A": 2}}, None)


def test_error_when_indices_are_not_int_or_slice():
    with pytest.raises(exception._Py4VaspInternalError):
        index.Selector({0: {"A": [1, 2]}}, None)


@pytest.mark.parametrize(
    "selection",
    [
        ("A",),
        (select.Group(["A", "B"], select.range_separator),),
        (select.Group(["A", "B"], select.pair_separator),),
    ],
)
def test_error_when_key_is_not_present(selection):
    map_ = {0: {"B": 1}}
    with pytest.raises(exception.IncorrectUsage):
        index.Selector(map_, np.zeros(10))[selection]


def test_error_when_range_belongs_to_different_dimensions():
    map_ = {0: {"A": 1}, 1: {"x": 2}}
    selector = index.Selector(map_, np.zeros(10))
    with pytest.raises(exception.IncorrectUsage):
        selector[(select.Group(["A", "x"], select.range_separator),)]


@pytest.mark.parametrize("range_", [("A", "B"), ("B", "A")])
def test_error_when_slice_has_step(range_):
    map_ = {0: {"A": slice(1, 5, -1), "B": 2}}
    selector = index.Selector(map_, np.zeros(10))
    group = select.Group(range_, select.range_separator)
    with pytest.raises(exception.IncorrectUsage):
        selector[(group,)]


def test_error_when_two_selections_for_the_same_dimension():
    map_ = {0: {"A": 1, "B": 2}}
    with pytest.raises(exception.IncorrectUsage):
        index.Selector(map_, np.zeros(10))[("A", "B")]

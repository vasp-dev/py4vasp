# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._util import index, select


def make_range(left, right):
    return select.Group([left, right], select.range_separator)


def make_pair(left, right):
    return select.Group([left, right], select.pair_separator)


def make_operation(left, operator, right):
    return select.Operation((left,), operator, (right,))


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
        ((make_range("1", "3"),), slice(0, 3)),
        ((make_range("2", "6"),), slice(1, 6)),
        ((make_range("4", "5"),), slice(3, 5)),
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
        ((make_pair("A", "B"),), 1),
        ((make_pair("B", "A"),), 1),
        ((make_pair("C", "A"),), 2),
        ((make_pair("B", "C"),), 3),
    ],
)
def test_select_pair(selection, indices):
    values = np.arange(10) ** 2
    map_ = {0: {"total": 0, "A~B": 1, "A~C": 2, "B~C": 3}}
    selector = index.Selector(map_, values)
    assert selector[selection] == np.sum(values[indices])


@pytest.mark.parametrize(
    "selection, expected",
    [
        ((make_operation("A", "+", "B"),), 5),
        ((make_operation("C", "-", "D"),), -7),
        ((make_operation("E", "+", "E"),), 50),
        ((make_operation("A", "+", make_operation("B", "-", "C")),), -4),
        ((make_operation("A", "-", make_operation("B", "+", "C")),), 6),
        ((make_operation(make_operation("D", "-", "E"), "+", "F"),), 27),
        ((make_operation(make_operation("D", "+", "E"), "-", "F"),), 5),
    ],
)
def test_select_operation(selection, expected, Assert):
    values = np.arange(10) ** 2
    map_ = {0: {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}}
    selector = index.Selector(map_, values)
    Assert.allclose(selector[selection], expected)


@pytest.mark.parametrize(
    "selection, expected",
    [
        ((make_operation("A", "+", "x"),), [669, 7029, 20589]),
        ((make_operation("y", "-", "B"),), [-244, -1524, -3604]),
        ((select.Operation(("A", "y"), "-", ("x", "B")),), [-72, -232, -392]),
        (("A", make_operation("y", "-", "x")), [13, 53, 93]),
        ((make_operation(make_range("A", "B"), "-", "x"),), [571, 5411, 15051]),
        ((make_operation("y", "-", make_pair("z", "z")),), [-80, -240, -400]),
    ],
)
def test_mix_indices(selection, expected, Assert):
    values = np.arange(60).reshape((3, 4, 5)) ** 2
    map_ = {1: {"A": 1, "B": 2}, 2: {"x": 1, "y": 2, "z~z": 3}}
    selector = index.Selector(map_, values)
    Assert.allclose(selector[selection], expected)


@pytest.mark.parametrize(
    "first_text, second_text",
    [
        ("A - B(x + y)", "A - x(B) - y(B)"),
        ("-A + B", "B - A"),
        ("-A(x - y)", "A(y) - A(x)"),
        ("A(1:3) + A(4:5)", "A"),
        ("A(x + y(2)) - B(z(1 + 2))", "A(x) + A(y(2)) - B(z(1:2))"),
        ("A - B(1 - x(2 + 3) + y(4 - 5))", "A - B(1) + B(x(2:3)) - B(y(4)) + B(y(5))"),
    ],
)
def test_equivalent_operation(first_text, second_text, Assert):
    values = np.log(np.arange(120).reshape([2, 3, 4, 5]) + 1)
    map_ = {
        3: {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4},
        0: {"A": 0, "B": 1},
        1: {"x": 0, "y": 1, "z": 2},
    }
    selector = index.Selector(map_, values)
    first_selections = select.Tree.from_selection(first_text).selections()
    second_selections = select.Tree.from_selection(second_text).selections()
    for first, second in itertools.zip_longest(first_selections, second_selections):
        assert first is not None
        assert second is not None
        Assert.allclose(selector[first], selector[second])


def test_complex_operation(Assert):
    values = np.sqrt(np.arange(120).reshape([5, 4, 3, 2]))
    map_ = {
        1: {"A": 1, "B": 2, "C~D": 3},
        0: {"x": 1, "y": 2, "u": 3, "v": 4},
        2: {"1": 0, "z": 1, "3": 2},
    }
    selector = index.Selector(map_, values)
    selection = "A(x + y(z)) + B(1:3 u - v) - C~D"
    selections = select.Tree.from_selection(selection).selections()
    expected_results = [
        [17.923642676146642, 18.351800332750983],
        [-98.3178573582437, -99.00269818706242],
    ]
    for selection, expected in zip(selections, expected_results):
        Assert.allclose(selector[selection], expected)


@pytest.mark.parametrize(
    "selection, label",
    [
        (("up",), "up"),
        (("A",), "A"),
        (("x",), "x"),
        (("down", "B"), "B_down"),
        (("B", "y"), "B_y"),
        (("z", "up"), "z_up"),
        (("1",), "A_1"),
        (("5",), "B_2"),
    ],
)
def test_label(selection, label):
    numbers = {str(i + 1): i for i in range(7)}
    map_ = {
        1: {"A": slice(0, 3), "B": slice(3, 7), **numbers},
        2: {"x": 0, "y": 1, "z": 2},
        0: {"total": slice(0, 2), "up": 0, "down": 1},
    }
    selector = index.Selector(map_, np.zeros((2, 7, 3, 0)))
    assert selector.label(selection) == label


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

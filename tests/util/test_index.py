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


def make_assignment(left, right):
    return select.Assignment(left, right)


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


@pytest.mark.parametrize("selection, expected", [("A", 98), ("B", 45)])
def test_map_contains_slice_or_step(selection, expected):
    values = np.arange(10) ** 2
    map_ = {0: {"A": [1, 4, 9], "B": slice(0, 8, 3)}}
    selector = index.Selector(map_, values)
    assert selector[(selection,)] == expected


@pytest.mark.parametrize("selection, indices", [("x", [7]), ("y", [2, 5])])
def test_select_one_of_two_components(selection, indices):
    values = np.arange(30).reshape((3, 10))
    map_ = {1: {"x": 7, "y": slice(2, 6, 3)}}
    selector = index.Selector(map_, values)
    assert np.all(selector[(selection,)] == np.sum(values[:, indices], axis=1))


@pytest.mark.parametrize(
    "selection, expected",
    [
        ("x", [1.732050807568877, 2.0, 2.23606797749979]),
        ("y", [2.917980236721544, 3.182861868757633, 3.366007993209797]),
        ("z", [3.464101615137755, 3.632351299197724, 3.787630057136079]),
    ],
)
def test_custom_function(selection, expected, Assert):
    values = np.sqrt(np.arange(30)).reshape(10, 3)
    map_ = {0: {"x": 1, "y": slice(0, 8), "z": [1, 4, 9]}}
    selector = index.Selector(map_, values, reduction=np.average)
    Assert.allclose(selector[(selection,)], expected)


@pytest.mark.parametrize("selection, expected", [("A", 4), ("B", 11), ("C", -1)])
def test_select_from_list(selection, expected):
    values = [11, 9, 7, 4, 2, -1]
    map_ = {0: {"A": 3, "B": 0, "C": -1}}
    selector = index.Selector(map_, values)
    assert selector[(selection,)] == expected
    assert selector.label(selection) == selection


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
    "selection, expected",
    [(("A",), [26, 34]), (("x",), [14, 22]), ((), [10, 18]), (("x", "A"), [30, 38])],
)
def test_select_with_default(selection, expected):
    values = np.arange(24).reshape(3, 2, 4)
    map_ = {0: {"A": [1, 2], None: slice(0, 2)}, 2: {None: 1, "x": 3}}
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
    "selection, indices",
    [
        ((make_assignment("int", "1"),), 0),
        ((make_assignment("int", "2"),), slice(1, 4)),
        ((make_assignment("str", "x"),), slice(2, 5)),
        ((make_assignment("str", "y"),), 3),
        ((make_assignment("float", "-0.5"),), slice(5, 10)),
        ((make_assignment("float", "0"),), 4),
    ],
)
def test_select_assignment(selection, indices):
    values = np.arange(10) ** 2
    map_ = {
        0: {
            "int": {1: 0, 2: slice(1, 4)},
            "str": {"x": slice(2, 5), "y": 3},
            "float": {-0.5: slice(5, 10), 1e-8: 4},
        },
    }
    selector = index.Selector(map_, values)
    assert selector[selection] == np.sum(values[indices])
    assert selector.label(selection) == str(selection[0])


def test_error_if_assignment_key_is_invalid():
    data = np.zeros((3, 2))
    map_ = {0: {"x": {}}}
    selector = index.Selector(map_, data)
    with pytest.raises(exception.IncorrectUsage):
        selector[(select.Assignment("y", 1),)]


def test_error_if_assignment_value_is_not_present():
    data = np.zeros((3, 2))
    map_ = {0: {"x": {1: 0}}}
    selector = index.Selector(map_, data)
    with pytest.raises(exception.IncorrectUsage):
        selector[(select.Assignment("x", 2),)]


def test_error_if_assignment_used_on_non_mapping():
    data = np.zeros((3, 2))
    map_ = {0: {"x": 1}}
    selector = index.Selector(map_, data)
    with pytest.raises(exception.IncorrectUsage):
        selector[(select.Assignment("x", 1),)]


@pytest.mark.parametrize(
    "selection",
    [
        (make_assignment("x", "invalid"),),
        (make_assignment("x", "1.5"),),
        (make_assignment("y", "invalid"),),
    ],
)
def test_error_if_typecasting_fails(selection):
    data = np.zeros((3, 2))
    map_ = {0: {"x": {1: 0}, "y": {1.5: 1}}}
    selector = index.Selector(map_, data)
    with pytest.raises(exception.IncorrectUsage):
        selector[selection]


@pytest.mark.parametrize(
    "selection", [("x",), (make_range("x", "y"),), (make_pair("x", "y"),)]
)
def test_error_if_nonassignment_used_on_mapping(selection):
    data = np.zeros((3, 2))
    map_ = {0: {"x": {1: 0}}}
    selector = index.Selector(map_, data)
    with pytest.raises(exception.IncorrectUsage):
        selector[selection]


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


@pytest.mark.parametrize(
    "selection, expected",
    [
        (("A",), [0.909297426825682, 0.956375928404503]),
        (("x",), [-1.566975662971037, -3.801910387458235]),
        (("B", "y"), [0.823172422671085, 0.073694517959313]),
        (("z", "C"), [0.660919112573909, 0.485842070293535]),
        ((make_range("B", "D"), "z"), [0.66432711338294, 0.499714623309142]),
        ((make_operation("x", "+", "y"), "C"), [0.168814278260321, -1.165363630766923]),
        ((make_operation("D", "-", "z"),), [0.273708667728343, 0.339540822873806]),
    ],
)
def test_dynamic_reduction(selection, expected, Assert):
    values = np.sin(np.arange(48)).reshape(2, 4, 6)
    map_ = {
        1: {"A": 0, "B": 1, "C": 2, "D": 3},
        2: {"x": 0, "y": slice(1, 3), "z": slice(3, 6)},
    }
    selector = index.Selector(map_, values, reduction=ExampleReduction)
    Assert.allclose(selector[selection], expected)


class ExampleReduction(index.Reduction):
    def __init__(self, keys):
        if keys[-1] == "x":
            self._reduction = np.sum
        elif keys[-1] == "y":
            self._reduction = np.average
        elif keys[-1] == "z":
            self._reduction = np.std
        else:
            self._reduction = np.max

    def __call__(self, array, axis):
        return self._reduction(array, axis=axis)


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
    "selection, expected",
    [
        ("A + x", [0.435144515066183, 1.278152404958349, 2.037388074302085]),
        ("x(A) - x", [0.2516487639954, 0.270462229948895, 0.251648763995401]),
        (
            "y - x + A",
            [1.0351275429316753e-01, 1.3523111497444718e-01, 1.4813600970223307e-01],
        ),
    ],
)
def test_operation_with_default_selection(selection, expected, Assert):
    values = np.tanh(np.linspace(-2, 2, 60)).reshape(5, 3, 4)
    map_ = {2: {"A": slice(2, 4), None: slice(0, 2)}, 0: {"x": 2, "y": 1, None: 3}}
    selector = index.Selector(map_, values)
    selection, *_ = select.Tree.from_selection(selection).selections()
    Assert.allclose(selector[selection], expected)


@pytest.mark.parametrize(
    "selection, expected",
    [
        ("A + x", [4.069394429954547e-02, 3.866352549793870e-09]),
        ("B - y", [-1.732435589575556e-02, 3.400533929941882e-04]),
        ("A(z) + B(x - y)", [-3.599130760612862e-01, -5.358270586524438e-02]),
    ],
)
def test_operation_with_custom_reduction(selection, expected, Assert):
    values = np.log(np.linspace(0.1, 1.9, 48)).reshape(2, 4, 6)
    map_ = {
        1: {"A": slice(0, 3), "B": slice(2, 4)},
        2: {"x": slice(1, 5), "y": slice(None, None, 3), "z": slice(0, 5, 2)},
    }
    selector = index.Selector(map_, values, reduction=np.prod)
    selection, *_ = select.Tree.from_selection(selection).selections()
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
        (("8",), "8"),  # use number of no matching label
        (("up", "z", "3"), "A_3_z_up"),
        ((make_pair("u", "v"),), "u~v"),
        ((make_pair("v", "u"),), "u~v"),
        ((make_range("2", "4"),), "2:4"),
        (("A", make_range("x", "z")), "A_x:z"),
        (("z", make_range("1", "5")), "1:5_z"),
        ((make_operation("A", "+", "B"),), "A + B"),
        ((make_operation("x", "-", "up"),), "x - up"),
        ((select.Operation((), "+", "1"),), "A_1"),
        ((select.Operation((), "-", "x"),), "-x"),
    ],
)
def test_label(selection, label):
    numbers = {str(i + 1): i for i in range(8)}
    map_ = {
        1: {"A": slice(0, 3), "B": slice(3, 7), **numbers},
        2: {"x": 0, "y": 1, "z": 2, "u~v": 3},
        0: {"total": slice(0, 2), "up": 0, "down": 1},
    }
    selector = index.Selector(map_, np.zeros((2, 8, 4, 0)), use_number_labels=True)
    assert selector.label(selection) == label


@pytest.mark.parametrize(
    "selection, label",
    [
        ("A(y) - x(B)", "A_y - B_x"),
        ("A - B(x + y)", "A - B_x - B_y"),
        ("z - A:B", "z - A:B"),
        ("x(1) + y(2) - z(4)", "A_1_x + A_2_y - B_1_z"),
        ("up(x + y(2)) - down(z(1 + 2))", "x_up + A_2_y_up - A_1_z_down - A_2_z_down"),
    ],
)
def test_label_operations(selection, label):
    tree = select.Tree.from_selection(selection)
    selection, *_ = tree.selections()
    numbers = {str(i + 1): i for i in range(8)}
    map_ = {
        1: {"A": slice(0, 3), "B": slice(3, 7), **numbers},
        2: {"x": 0, "y": 1, "z": 2, "u~v": 3},
        0: {"total": slice(0, 2), "up": 0, "down": 1},
    }
    selector = index.Selector(map_, np.zeros((2, 8, 4, 0)), use_number_labels=True)
    assert selector.label(selection) == label


def test_labels_without_number_labels():
    map_ = {1: {"A": 0, "0": 0}}
    selector = index.Selector(map_, np.zeros((10, 10)))
    assert selector.label(("0",)) == "0"


def test_error_when_duplicate_key():
    with pytest.raises(exception._Py4VaspInternalError):
        index.Selector({0: {"A": 1}, 1: {"A": 2}}, None)


def test_error_when_numbers_are_longer_than_one():
    map_ = {0: {"A": 1, "1": slice(0, 2)}}
    with pytest.raises(exception._Py4VaspInternalError):
        index.Selector(map_, np.zeros(3), use_number_labels=True)


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
    selector = index.Selector(map_, np.zeros((3, 4)))
    with pytest.raises(exception.IncorrectUsage):
        selector[(select.Group(["A", "x"], select.range_separator),)]


@pytest.mark.parametrize("range_", [("A", "B"), ("B", "A"), ("C", "B")])
def test_error_when_slice_has_step(range_):
    map_ = {0: {"A": slice(1, 5, -1), "B": 2, "C": [1, 4]}}
    selector = index.Selector(map_, np.zeros(10))
    group = select.Group(range_, select.range_separator)
    with pytest.raises(exception.IncorrectUsage):
        selector[(group,)]


def test_error_when_two_selections_for_the_same_dimension():
    map_ = {0: {"A": 1, "B": 2}}
    with pytest.raises(exception.IncorrectUsage):
        index.Selector(map_, np.zeros(10))[("A", "B")]


def test_error_when_out_of_bounds_access():
    data = np.zeros((5, 4))
    map_ = {2: {"u": 1}}
    with pytest.raises(exception._Py4VaspInternalError):
        index.Selector(map_, data)


def test_error_if_indices_are_not_convertible():
    data = np.zeros((3, 2))
    map_ = {0: {"x": "y"}}
    with pytest.raises(exception._Py4VaspInternalError):
        index.Selector(map_, data)

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp import exception
from py4vasp._util import merge


@pytest.mark.parametrize(
    "left, right",
    [
        ({0.0: "Γ", 0.5: "X"}, {0.5: "X", 0.0: "Γ"}),
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}),
        (np.array([b"a", b"b"]), np.array([b"a", b"b"])),
        (np.array([{1: 2}], dtype=object), np.array([{1: 2}], dtype=object)),
        (
            np.array(["2020-01-01"], "datetime64[D]"),
            np.array(["2020-01-01"], "datetime64[D]"),
        ),
        ("label", "label"),
        (None, None),
        ([1, 2], (1, 2)),
    ],
)
def test_values_close_accepts_whatever_values_equal_accepts(left, right):
    assert merge.values_equal(left, right)
    assert merge.values_close(left, right)


@pytest.mark.parametrize(
    "left, right",
    [
        (1.0, 1.0 + 1e-14),
        ({0.0: "Γ", 0.5: "X"}, {0.0: "Γ", 0.5 + 1e-15: "X"}),
        ({0.5: "X", 0.0: "Γ"}, {0.0: "Γ", 0.5 + 1e-15: "X"}),
        ((0.0, 1.0), (0.0, 1.0 + 1e-15)),
        (np.array([1.0, 2.0]), np.array([1.0, 2.0 + 1e-14])),
        (1 + 1j, 1 + 1j + 1e-14),
    ],
)
def test_values_close_ignores_rounding(left, right):
    assert not merge.values_equal(left, right)
    assert merge.values_close(left, right)


@pytest.mark.parametrize(
    "left, right",
    [
        (np.array([True, False]), np.array([1.0, 1e-13])),
        (True, 1.0000001),
        ({0.0: "Γ", 0.5: "X"}, {0.0: "Γ", 0.6: "X"}),
        ({0.0: "Γ", 0.5: "X"}, {0.0: "Γ", 0.5: "K"}),
        ({0.0: "Γ", 0.5: "X"}, {0.0: "Γ"}),
        ({0.0: "Γ"}, [0.0]),
        ((0.0, 1.0), (0.0, 1.0, 2.0)),
        (np.array([1.0, 1.0, 1.0]), np.array(1.0)),
        (np.array([1.0, 2.0]), np.array([1.0, 3.0])),
    ],
)
def test_values_close_rejects_a_real_difference(left, right):
    assert not merge.values_close(left, right)


def test_merge_field_uses_the_comparison_it_is_given():
    left, right = {0.0: "Γ"}, {1e-15: "Γ"}
    assert (
        merge.merge_field_or_raise(left, right, "xticks", "graphs", merge.values_close)
        is left
    )
    with pytest.raises(exception.IncorrectUsage):
        merge.merge_field_or_raise(left, right, "xticks", "graphs")


def test_values_close_pairs_keys_it_cannot_sort_in_order():
    # a dict whose keys cannot be ordered falls back to the order they come in
    left = {1: 1.0, "b": 2.0}
    assert merge.values_close(left, {1: 1.0 + 1e-15, "b": 2.0})
    assert not merge.values_close(left, {1: 1.0, "b": 3.0})


def test_merge_unique_sequences_requires_sequences():
    with pytest.raises(exception.IncorrectUsage):
        merge.merge_unique_sequences([1], 2, lambda x, y: x == y)


def test_merge_unique_sequences_keeps_the_type_of_the_left_side():
    equal = lambda left, right: left == right
    assert merge.merge_unique_sequences([1, 2], [2, 3], equal) == [1, 2, 3]
    assert merge.merge_unique_sequences((1, 2), (2, 3), equal) == (1, 2, 3)
    # a type that cannot be rebuilt from a list comes back as a tuple
    assert merge.merge_unique_sequences(range(3), [3], equal) == (0, 1, 2, 3)
    assert merge.merge_unique_sequences([], [1], equal) == [1]
    assert merge.merge_unique_sequences([1], [], equal) == [1]


class _Ambiguous:
    """Behaves like an array that refuses to be reduced to a single truth value."""

    def __bool__(self):
        raise ValueError("truth value is ambiguous")

    def __array__(self, dtype=None, copy=None):
        raise ValueError("cannot be converted to an array")


def test_is_unset_treats_an_ambiguous_value_as_set():
    assert not merge.is_unset(_Ambiguous())


def test_values_equal_is_false_when_an_array_cannot_be_compared():
    assert not merge.values_equal(np.array([1.0]), _Ambiguous())

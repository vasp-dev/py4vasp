# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools
from collections.abc import Sequence

import numpy as np

from py4vasp import exception

RELATIVE_TOLERANCE = 1e-10
"""Relative difference under which two numbers are considered the same."""
ABSOLUTE_TOLERANCE = 1e-12
"""Absolute difference under which two numbers are considered the same."""


def is_unset(value):
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0
    try:
        return not value
    except ValueError:
        return False


def values_equal(left_value, right_value):
    if isinstance(left_value, np.ndarray) or isinstance(right_value, np.ndarray):
        try:
            return np.array_equal(np.asarray(left_value), np.asarray(right_value))
        except Exception:
            return False
    sequence_type = (list, tuple)
    if isinstance(left_value, sequence_type) and isinstance(right_value, sequence_type):
        if len(left_value) != len(right_value):
            return False
        return all(
            values_equal(left_entry, right_entry)
            for left_entry, right_entry in zip(left_value, right_value)
        )
    return left_value == right_value


def values_close(left_value, right_value):
    """Like :func:`values_equal` but tolerant of floating-point noise.

    Two calculations that evaluate the same quantity rarely produce bit-identical
    numbers -- summing the same distances in a different order is enough to change
    the last digits. Comparing those exactly rejects data that is the same, so
    numbers compare within a tolerance here while everything else still has to
    match exactly. The tolerance is far below any difference that carries meaning,
    so genuinely different data is still rejected.
    """
    if isinstance(left_value, dict) or isinstance(right_value, dict):
        if not (isinstance(left_value, dict) and isinstance(right_value, dict)):
            return False
        if len(left_value) != len(right_value):
            return False
        return all(
            values_close(left_key, right_key) and values_close(left_entry, right_entry)
            for (left_key, left_entry), (right_key, right_entry) in zip(
                left_value.items(), right_value.items()
            )
        )
    sequence_type = (list, tuple)
    if isinstance(left_value, sequence_type) and isinstance(right_value, sequence_type):
        if len(left_value) != len(right_value):
            return False
        return all(
            values_close(left_entry, right_entry)
            for left_entry, right_entry in zip(left_value, right_value)
        )
    if _both_numeric(left_value, right_value):
        return _numbers_close(left_value, right_value)
    return values_equal(left_value, right_value)


def _both_numeric(left_value, right_value):
    numeric = (int, float, np.number, np.ndarray)
    if isinstance(left_value, bool) or isinstance(right_value, bool):
        return False
    if not (isinstance(left_value, numeric) and isinstance(right_value, numeric)):
        return False
    return not any(
        np.issubdtype(np.asarray(value).dtype, np.str_)
        for value in (left_value, right_value)
    )


def _numbers_close(left_value, right_value):
    left_array, right_array = np.asarray(left_value), np.asarray(right_value)
    if left_array.shape != right_array.shape:
        return False
    try:
        return bool(
            np.allclose(
                left_array,
                right_array,
                rtol=RELATIVE_TOLERANCE,
                atol=ABSOLUTE_TOLERANCE,
            )
        )
    except (TypeError, ValueError):
        return False


def merge_field_or_raise(
    left_field, right_field, field_name, object_name, equal=values_equal
):
    if is_unset(left_field):
        return right_field
    if is_unset(right_field):
        return left_field
    if not equal(left_field, right_field):
        message = f"""Cannot combine two {object_name} with incompatible {field_name}:
    left: {left_field}
    right: {right_field}"""
        raise exception.IncorrectUsage(message)
    return left_field


def merge_unique_sequences(left_values, right_values, entries_equal):
    if is_unset(left_values):
        return right_values
    if is_unset(right_values):
        return left_values
    if not isinstance(left_values, Sequence) or not isinstance(right_values, Sequence):
        message = "Special merge expected sequence inputs on both sides."
        raise exception.IncorrectUsage(message)

    merged = []
    for value in itertools.chain(left_values, right_values):
        if any(entries_equal(value, seen) for seen in merged):
            continue
        merged.append(value)
    return _as_left_sequence_type(left_values, merged)


def _as_left_sequence_type(left_values, merged):
    if isinstance(left_values, list):
        return merged
    if isinstance(left_values, tuple):
        return tuple(merged)
    try:
        return type(left_values)(merged)
    except Exception:
        return tuple(merged)

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools
from collections.abc import Sequence

import numpy as np

from py4vasp import exception


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


def merge_field_or_raise(left_field, right_field, field_name, object_name):
    if is_unset(left_field):
        return right_field
    if is_unset(right_field):
        return left_field
    if not values_equal(left_field, right_field):
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

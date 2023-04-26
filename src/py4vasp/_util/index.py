# Copyright © VASP Software GmbH,G
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
import string

import numpy as np

from py4vasp import exception
from py4vasp._util import select


class Selector:
    def __init__(self, maps, data):
        self._data = data
        self._axes = tuple(maps.keys())
        _raise_error_if_duplicate_keys(maps)
        self._map = {
            key: (dim, _make_slice(indices))
            for dim, map_ in maps.items()
            for key, indices in map_.items()
        }
        self._weights = [
            np.ones(data.shape[dim]) if dim in maps else None
            for dim in range(data.ndim)
        ]
        dims = {string.ascii_letters[dim] for dim in maps}
        self._einsum = ",".join([string.ascii_letters[: data.ndim], *sorted(dims)])

    def __getitem__(self, selection):
        weights = self._get_weights(selection)
        relevant_weights = filter(lambda x: x is not None, weights)
        return np.einsum(self._einsum, self._data, *relevant_weights)

    def _get_weights(self, selection):
        if isinstance(selection[0], select.Operation):
            return self._evaluate_operation(selection[0])
        weights = self._weights.copy()
        keys = [""] * self._data.ndim
        for key in selection:
            dimension, weight = self._get_dimension_and_weight(key)
            _raise_error_if_index_already_set(keys[dimension], key)
            weights[dimension] = weight
            keys[dimension] = key
        return weights

    def _get_dimension_and_weight(self, key):
        try:
            if isinstance(key, str):
                return self._weight_from_slice(*self._map[key])
            elif _is_range(key):
                return self._read_range(key)
            elif _is_pair(key):
                return self._read_pair(key)
            else:
                assert False, f"Reading {key} is not implemented."
        except KeyError as error:
            message = (
                f"Could not read {key}, please check the spelling and capitalization."
            )
            raise exception.IncorrectUsage(message) from error

    def _weight_from_slice(self, dimension, slice_):
        weight = np.zeros_like(self._weights[dimension])
        weight[slice_] = 1
        return dimension, weight

    def _read_range(self, range_):
        dimension = self._read_dimension(range_)
        slice_ = self._merge_slice(range_)
        return self._weight_from_slice(dimension, slice_)

    def _read_dimension(self, range_):
        dim1, _ = self._map[range_.group[0]]
        dim2, _ = self._map[range_.group[1]]
        if dim1 == dim2:
            return dim1
        message = f"The range {range_} could not be read, because the components correspond to different dimensions."
        raise exception.IncorrectUsage(message)

    def _merge_slice(self, range_):
        allowed_steps = (None, 1)
        _, left = self._map[range_.group[0]]
        _, right = self._map[range_.group[1]]
        if left.step in allowed_steps and right.step in allowed_steps:
            return slice(left.start, right.stop)
        message = f"Cannot read range {range_} because the data is not contiguous."
        raise exception.IncorrectUsage(message)

    def _read_pair(self, pair):
        key = str(pair)
        if key not in self._map:
            pair = dataclasses.replace(pair, group=reversed(pair.group))
            key = str(pair)
        return self._weight_from_slice(*self._map[key])

    def _evaluate_operation(self, operation):
        left_weights = self._get_weights(operation.left_operand)
        right_weights = self._get_weights(operation.right_operand)
        return [
            _combine(*operands, operation.operator)
            for operands in zip(left_weights, right_weights)
        ]


def _make_slice(indices):
    if isinstance(indices, int):
        return slice(indices, indices + 1)
    if isinstance(indices, slice):
        return indices
    message = f"A conversion of {indices} to slice is not implemented."
    raise exception._Py4VaspInternalError(message)


def _raise_error_if_index_already_set(previous_key, key):
    if not previous_key:
        return
    message = f"Conflicting keys '{previous_key}' and '{key}' act on the same index."
    raise exception.IncorrectUsage(message)


def _raise_error_if_duplicate_keys(maps):
    duplicates = _find_duplicates(maps)
    if not duplicates:
        return
    raise exception._Py4VaspInternalError(_format_error_message(duplicates))


def _find_duplicates(maps):
    keys = set()
    duplicates = set()
    for map_ in maps.values():
        new_keys = set(map_.keys())
        duplicates.update(keys.intersection(new_keys))
        keys.update(new_keys)
    return duplicates


def _format_error_message(duplicates):
    text = "', '".join(duplicates)
    occur = "occurs" if len(duplicates) == 1 else "occur"
    return f"The maps may not have duplicate keys, but '{text}' {occur} more than once."


def _is_range(key):
    return isinstance(key, select.Group) and key.separator == select.range_separator


def _is_pair(key):
    return isinstance(key, select.Group) and key.separator == select.pair_separator


def _combine(left, right, operator):
    if operator == "+":
        return left + right
    elif operator == "-":
        return left - right

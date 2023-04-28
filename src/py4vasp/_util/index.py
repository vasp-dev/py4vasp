# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
import itertools

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

    def __getitem__(self, selection):
        return sum(
            slices.factor * np.sum(self._data[slices.indices], axis=self._axes)
            for slices in self._get_all_slices(selection)
        )

    def _get_all_slices(self, selection):
        if len(selection) == 0:
            yield _Slices(self._data.ndim)
        elif len(selection) == 1:
            yield from self._get_slices_from_single_selection(*selection)
        else:
            left_slices = self._get_all_slices(selection[::2])
            right_slices = self._get_all_slices(selection[1::2])
            for left, right in itertools.product(left_slices, right_slices):
                yield _Slices.from_merge(left, right)

    def _get_slices_from_single_selection(self, selection):
        if isinstance(selection, str):
            yield self._read_key(selection)
        elif _is_range(selection):
            yield self._read_range(selection)
        elif _is_pair(selection):
            yield self._read_pair(selection)
        elif isinstance(selection, select.Operation):
            yield from self._evaluate_operation(selection)
        else:
            assert False, f"Reading {key} is not implemented."

    def _read_key(self, key):
        try:
            dimension, slice_ = self._map[key]
        except KeyError as error:
            _raise_key_not_found_error(key, error)
        return _Slices(self._data.ndim).set(dimension, slice_, key)

    def _read_range(self, range_):
        try:
            dimension = self._read_dimension(range_)
            slice_ = self._merge_slice(range_)
        except KeyError as error:
            _raise_key_not_found_error(range_, error)
        return _Slices(self._data.ndim).set(dimension, slice_, range_)

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
        return self._read_key(key)

    def _evaluate_operation(self, operation):
        yield from self._get_all_slices(operation.left_operand)
        generator = self._get_all_slices(operation.right_operand)
        yield next(generator).set_operator(operation.operator)
        yield from generator


def _make_slice(indices):
    if isinstance(indices, int):
        return slice(indices, indices + 1)
    if isinstance(indices, slice):
        return indices
    message = f"A conversion of {indices} to slice is not implemented."
    raise exception._Py4VaspInternalError(message)


def _is_range(key):
    return isinstance(key, select.Group) and key.separator == select.range_separator


def _is_pair(key):
    return isinstance(key, select.Group) and key.separator == select.pair_separator


class _Slices:
    def __init__(self, ndim):
        self._indices = [slice(None)] * ndim
        self._keys = [""] * ndim
        self.factor = 1

    @classmethod
    def from_merge(cls, left, right):
        slices = cls(len(left._indices))
        slices._keys = _merge_keys(left._keys, right._keys)
        slices._indices = _merge_indices(left._indices, right._indices)
        slices.factor = left.factor * right.factor
        return slices

    def set(self, dimension, slice_, key):
        self._indices[dimension] = slice_
        self._keys[dimension] = key
        return self

    def set_operator(self, operator):
        self.factor = 1 if operator == "+" else -1
        return self

    @property
    def indices(self):
        return tuple(self._indices)

    def print(self, label):
        print(label, self._indices, self._keys, self.factor)


def _merge_keys(left_keys, right_keys):
    result = []
    for left_key, right_key in zip(left_keys, right_keys):
        _raise_error_if_index_used_twice(left_key, right_key)
        result.append(left_key or right_key)
    return result


def _merge_indices(left_indices, right_indices):
    return [
        left_index if right_index == slice(None) else right_index
        for left_index, right_index in zip(left_indices, right_indices)
    ]


def _raise_error_if_index_used_twice(left_key, right_key):
    if not left_key or not right_key:
        return
    message = f"Conflicting keys '{left_key}' and '{right_key}' act on the same index."
    raise exception.IncorrectUsage(message)


def _raise_error_if_duplicate_keys(maps):
    duplicates = _find_duplicates(maps)
    if not duplicates:
        return
    raise exception._Py4VaspInternalError(_format_error_message(duplicates))


def _raise_key_not_found_error(key, error):
    message = f"Could not read {key}, please check the spelling and capitalization."
    raise exception.IncorrectUsage(message) from error


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

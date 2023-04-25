# Copyright © VASP Software GmbH,G
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
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
        indices = [slice(None)] * self._data.ndim
        keys = [""] * self._data.ndim
        for key in selection:
            dimension, slice_ = self._get_dimension_and_slice(key)
            _raise_error_if_index_already_set(keys[dimension], key)
            indices[dimension] = slice_
            keys[dimension] = key
        return np.sum(self._data[tuple(indices)], axis=self._axes)

    def _get_dimension_and_slice(self, key):
        try:
            if isinstance(key, str):
                return self._map[key]
            elif _is_range(key):
                return self._read_range(key)
            elif _is_pair(key):
                return self._read_pair(key)
            else:
                assert False, f"Reading {key} is not implemented."
        except KeyError as error:
            message = (
                "Could not read {key}, please check the spelling and capitalization."
            )
            raise exception.IncorrectUsage(message) from error

    def _read_range(self, range_):
        dimension = self._read_dimension(range_)
        slice_ = self._merge_slice(range_)
        return dimension, slice_

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
        if key in self._map:
            return self._map[key]
        pair.group = reversed(pair.group)
        return self._map[str(pair)]


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

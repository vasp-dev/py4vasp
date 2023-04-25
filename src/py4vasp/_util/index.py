# Copyright © VASP Software GmbH,
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
        for key in selection:
            dimension, slice_ = self._get_dimension_and_slice(key)
            indices[dimension] = slice_
        return np.sum(self._data[tuple(indices)], axis=self._axes)

    def _get_dimension_and_slice(self, key):
        if isinstance(key, str):
            return self._map[key]
        elif isinstance(key, select.Group):
            return self._read_group(key)

    def _read_group(self, group):
        dim1, left = self._map[group.group[0]]
        dim2, right = self._map[group.group[1]]
        if dim1 != dim2:
            message = f"The range {group} could not be read, because the components correspond to different dimensions."
            raise exception.IncorrectUsage(message)
        return dim1, slice(left.start, right.stop)


def _make_slice(indices):
    if isinstance(indices, int):
        return slice(indices, indices + 1)
    if isinstance(indices, slice):
        return indices
    message = f"A conversion of {indices} to slice is not implemented."
    raise exception._Py4VaspInternalError(message)


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

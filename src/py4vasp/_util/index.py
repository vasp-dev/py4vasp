# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import exception


class Selector:
    def __init__(self, maps, data):
        self._data = data
        self._axes = tuple(maps.keys())
        self._map = {
            key: (dim, _make_slice(indices))
            for dim, map_ in maps.items()
            for key, indices in map_.items()
        }

    def __getitem__(self, selection):
        indices = [slice(None)] * self._data.ndim
        for key in selection:
            dimension, slice_ = self._map[key]
            indices[dimension] = slice_
        return np.sum(self._data[tuple(indices)], axis=self._axes)


def _make_slice(indices):
    if isinstance(indices, int):
        return slice(indices, indices + 1)
    if isinstance(indices, slice):
        return indices
    message = f"A conversion of {indices} to slice is not implemented."
    raise exception._Py4VaspInternalError(message)

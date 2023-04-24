# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import annotations

import dataclasses

import numpy as np


class Selector:
    def __init__(self, maps, data):
        self._data = data
        self._map = {
            key: (dim, _make_1d(indices))
            for dim, map in maps.items()
            for key, indices in map.items()
        }

    def __getitem__(self, key):
        selection = [slice(None)] * self._data.ndim
        dim, indices = self._map[key]
        selection[dim] = indices
        return np.sum(self._data[tuple(selection)], axis=-1)


def _make_1d(indices):
    if isinstance(indices, int):
        return [indices]
    else:
        return indices

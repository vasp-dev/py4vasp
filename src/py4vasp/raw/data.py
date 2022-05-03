import numpy as np


class VaspData(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, data):
        self._data = data

    def __array__(self):
        return np.array(self._data)

    def __getitem__(self, key):
        return self._data[key]

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def size(self):
        return self._data.size

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

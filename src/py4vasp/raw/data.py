import numpy as np


class VaspData(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, data):
        self._data = data

    def __array__(self):
        return np.array(self._data)

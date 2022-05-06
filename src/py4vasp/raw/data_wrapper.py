import numpy as np


class VaspData(np.lib.mixins.NDArrayOperatorsMixin):
    """Wraps the data produced by the VASP calculation.

    Instead of exposing the underlying file structure directly, the data is wrapped in
    this container. This allows changing the way the data is internally represented
    without affecting the user. In particular, the data is possibly only lazily loaded
    when it is actually necessary.

    By inheriting from NDArrayOperatorsMixin most numpy functionality except for the
    class attributes should work. If any other feature is needed any instance of this
    class can be passed into a numpy array. Please be aware that using the data in
    this way will access the file. If performance is an issue, make sure that this
    file I/O is reduced as much as possible.

    Parameters
    ----------
    data
        The data wrapped by this container.
    """

    def __init__(self, data):
        self._data = data

    def __array__(self):
        return np.array(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self._data)})"

    def __len__(self):
        return len(self._data)

    @property
    def ndim(self):
        "The number of dimensions of the data."
        return self._data.ndim

    @property
    def size(self):
        "The total number of elements of the data."
        return self._data.size

    @property
    def shape(self):
        "The shape of the data. Empty tuple for scalar data."
        return self._data.shape

    @property
    def dtype(self):
        "Describes the type of the contained data."
        return self._data.dtype

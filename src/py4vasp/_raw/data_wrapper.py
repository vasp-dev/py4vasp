# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import textwrap

import numpy as np

from py4vasp import exception


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
        if data is None:
            self._data = None
            self._repr_data = "None"
            return
        if isinstance(data, VaspData):
            self._data = data._data
            self._repr_data = data._repr_data
            return
        self._repr_data = repr(data)
        if not hasattr(data, "__array__"):
            data = np.array(data)
        if getattr(data, "attrs", {}).get("dtype") == "complex":
            self._data = _ComplexWrapper(data)
        elif data.ndim == 0:
            self._data = _parse_scalar(data)
        else:
            self._data = data

    def __array__(self, *args, **kwargs):
        return np.array(self._get_data(), *args, **kwargs)

    def __getitem__(self, key):
        return self._get_data()[key]

    def __repr__(self):
        return f"{self.__class__.__name__}({self._repr_data})"

    def __len__(self):
        return len(self._get_data())

    def is_none(self):
        return self._data is None

    def _get_data(self):
        if self.is_none():
            message = """\
                Could not find data in output, please make sure that the provided input
                should produce this data and that the VASP calculation already finished.
                Also check that VASP did not exit with an error."""
            raise exception.NoData(textwrap.dedent(message))
        else:
            return self._data

    @property
    def ndim(self):
        "The number of dimensions of the data."
        return self._get_data().ndim

    @property
    def size(self):
        "The total number of elements of the data."
        return self._get_data().size

    @property
    def shape(self):
        "The shape of the data. Empty tuple for scalar data."
        return self._get_data().shape

    @property
    def dtype(self):
        "Describes the type of the contained data."
        return self._get_data().dtype

    def astype(self, *args, **kwargs):
        "Copy of the array, cast to a specified type."
        return self._get_data().astype(*args, **kwargs)


def _parse_scalar(data):
    if data.dtype.type == np.bytes_:
        data = data[()].decode()
    return np.array(data)


class _ComplexWrapper(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, data):
        self._data = data

    def __array__(self, *args, **kwargs):
        return self._cast_complex(np.array(self._data, *args, **kwargs))

    def __getitem__(self, key):
        keys = *np.atleast_1d(key), slice(None)
        return self._cast_complex(self._data[keys])

    def _cast_complex(self, array):
        return array.view(np.complex128).reshape(array.shape[:-1])

    @property
    def ndim(self):
        return self._data.ndim - 1

    @property
    def size(self):
        return self._data.size // 2

    @property
    def shape(self):
        return self._data.shape[:-1]

    @property
    def dtype(self):
        return np.complex128

    def astype(self, *args, **kwargs):
        return np.array(self).astype(*args, **kwargs)

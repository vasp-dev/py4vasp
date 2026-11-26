# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Turn the data a routine reads into a tree a tool can send on as it is.

A tool that shows the data of a quantity sends its arrays as raw bytes and everything
else as JSON. :func:`to_tree` brings what py4vasp reads into the three kinds of values
that allows:

- a dictionary with text keys, whose values are again of these kinds;
- a numpy array of booleans, integers, floating-point or complex numbers, or text;
- a scalar: None, bool, int, float, complex, or str.

A float may still be NaN or infinite; how to write it is left to the tool.

The tool shows indices counted from 1, as py4vasp's selections do. So a slice of
indices, e.g. the atoms of an element, is written the way a selection writes them:
counted from 1 and including its end, e.g. "1:2" for the first two atoms and "3" for
the third one. A slice that cannot be written that way without the length of what it
indexes, e.g. with an open end, a negative bound, or a step, keeps the notation of
Python in brackets, e.g. "[3:]".
"""

import dataclasses
from collections.abc import Mapping

import numpy as np

# The kinds of numpy arrays that pass unchanged: boolean, signed and unsigned integer,
# floating point, complex, and text.
_KEPT_KINDS = "biufcU"


def to_tree(value):
    """Convert data read by py4vasp into dictionaries of arrays and scalars.

    Parameters
    ----------
    value
        What a routine read, e.g. the dictionary of :meth:`read` or a database record.

    Returns
    -------
    dict or np.ndarray or None or bool or int or float or complex or str
        A dictionary for a mapping or a dataclass instance, keyed by text. An array for a
        list, a tuple, or an array whose elements form one array of numbers, booleans,
        or text; bytes become text. A dictionary keyed by the index for one whose
        elements do not, e.g. lists of unequal length or numbers mixed with text. A
        scalar for anything else: numpy scalars become Python ones, a slice its text
        as described in the module, and any other object its ``str``.
    """
    if isinstance(value, Mapping):
        return {str(key): to_tree(item) for key, item in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = dataclasses.fields(value)
        return {field.name: to_tree(getattr(value, field.name)) for field in fields}
    if isinstance(value, (np.ndarray, list, tuple)):
        return _array(value)
    return _scalar(value)


def _array(value):
    try:
        array = np.asarray(value)
    except ValueError:  # nested sequences of unequal length
        return _by_index(value)
    if array.ndim == 0:
        return _scalar(array.item())
    if array.dtype.kind == "S":
        return np.char.decode(array, errors="replace")
    if array.dtype.kind == "O":
        # e.g. the bytes HDF5 stores text as; a list that is still made of objects
        # after its elements are converted holds values of different kinds.
        if isinstance(value, np.ndarray):
            return _array(_converted(value.tolist()))
        return _by_index(value)
    if array.dtype.kind == "U" and not isinstance(value, np.ndarray):
        # numpy writes numbers mixed with text as text
        if not all(isinstance(item, str) for item in _leaves(value)):
            return _by_index(value)
    if array.dtype.kind in _KEPT_KINDS:
        return array
    return array.astype(str)  # e.g. dates and time spans


def _by_index(value):
    return {str(index): to_tree(item) for index, item in enumerate(value)}


def _converted(values):
    if isinstance(values, list):
        return [_converted(item) for item in values]
    return to_tree(values)


def _leaves(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaves(item)
    elif isinstance(value, np.ndarray):
        yield from value.ravel()
    else:
        yield value


def _scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, float, complex, str)):
        return value
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, slice):
        return _slice_text(value)
    return str(value)


def _slice_text(value):
    start, stop, step = value.start, value.stop, value.step
    counted = isinstance(start, int) and isinstance(stop, int) and step in (None, 1)
    if counted and 0 <= start < stop:
        first, last = start + 1, stop  # counted from 1, including the end
        return str(first) if first == last else f"{first}:{last}"
    parts = ["" if part is None else str(part) for part in (start, stop)]
    if step is not None:
        parts.append(str(step))
    return f"[{':'.join(parts)}]"

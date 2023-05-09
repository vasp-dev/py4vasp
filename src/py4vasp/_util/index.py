# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
""" Select indices from an array based on a map.

In multiple cases, VASP produces multiple outputs and the user wants to select one of
its components e.g. plotting the p DOS. This module provides the Selector class that
can take the selections of a `py4vasp._util.selection.Tree` and extract the relevant
components from the array. The defaults for each index is to select all components and
we sum over all selected components.

Example
-------
>>> from py4vasp._util import index
>>> selections = [("A", "x"), ("B", "y"), ("z",)]
>>> maps = {1: {"A": 0, "B": 1}, 2: {"x": 0, "y": 1, "z": 2}}
>>> data = np.random.random((10, 2, 3))
>>> selector = index.Selector(maps, data)
>>> result = [selector[selection] for selection in selections]

Here, we generated the selections manually; normally you would obtain them from a
`py4vasp._util.selection.Tree`. The maps define which labels correspond to which indices
in the data array. The key identifies which dimension of the data array to access. The
value is a dictionary of labels onto sections of the array. You can use single indices
like in this example or a `slice`. The first two selections will return `data[:,0,0]`
and `data[:,1,1]`, respectively. The last selection is equivalent to
`np.sum(data[:,:,2], axis=-1)` because we sum over all dimensions mentioned as keys
in `maps`.
"""
import dataclasses
import itertools

import numpy as np

from py4vasp import exception
from py4vasp._util import select


class Selector:
    """Manages the logic to read a user selection.

    Parameters
    ----------
    maps : dict
        The keys of the dictionary should be integer values indicating the dimension
        of the array described by its values. The values are dictionaries that map
        labels of the dimension onto corresponding slices of the array. Instead of a
        slice a single index is allowed as well.
    data : VaspData
        An array read from the VASP calculation. The indices in the maps should be
        compatible with the dimension of this array.
    """

    def __init__(self, maps, data):
        self._data = data
        self._axes = tuple(maps.keys())
        _raise_error_if_duplicate_keys(maps)
        self._map = self._make_map(maps)
        self._number_labels = self._make_number_labels(maps)

    def _make_map(self, maps):
        return {
            key: (dim, _make_slice(indices))
            for dim, map_ in maps.items()
            for key, indices in map_.items()
        }

    def _make_number_labels(self, maps):
        return {
            key: self._make_label(map_, key, index, self._data.shape[dim])
            for dim, map_ in maps.items()
            for key, index in map_.items()
            if key.isdecimal()
        }

    def _make_label(self, map_, number, index, size):
        indices = range(size)
        index, *rest = list(indices[_make_slice(index)])
        message = f"Integer label {number} maps to more than a single index."
        _raise_error_if_list_not_empty(rest, message)
        for key, value in map_.items():
            if key.isdecimal():
                continue
            range_ = indices[_make_slice(value)]
            if index in range_:
                return f"{key}_{range_.index(index) + 1}"
        return number

    def __getitem__(self, selection):
        """Main functionality provided by the class.

        Parameters
        ----------
        selection : tuple
            A selection ideally produced by `py4vasp._util.selection.Tree`. The elements
            of the tuple should correspond to labels in the maps used to initialize this
            class or mathematical operations of them.

        Returns
        -------
        np.ndarray
            The subsection of the array corresponding to the selection. Note that this
            will sum over all dimensions provided in the initialization of the class,
            i.e., ndim of the result = ndim of data - len(maps).
        """
        return sum(
            slices.factor * np.sum(self._data[slices.indices], axis=self._axes)
            for slices in self._get_all_slices(selection)
        )

    def label(self, selection):
        return " ".join(
            slices.label(i, self._axes, self._number_labels)
            for i, slices in enumerate(self._get_all_slices(selection))
        )
        for slices in self._get_all_slices(selection):
            print(slices.label(self._axes, self._number_labels))
            result = slices.label(self._axes, self._number_labels)
        return result

    def _get_all_slices(self, selection, operator="+"):
        if len(selection) == 0:
            yield _Slices(self._data.ndim)
        elif len(selection) == 1:
            yield from self._get_slices_from_single_selection(*selection, operator)
        else:
            left_slices = self._get_all_slices(selection[::2], operator)
            right_slices = self._get_all_slices(selection[1::2])
            for left, right in itertools.product(left_slices, right_slices):
                yield _Slices.from_merge(left, right)

    def _get_slices_from_single_selection(self, selection, operator):
        if isinstance(selection, str):
            yield self._read_key(selection).set_operator(operator)
        elif _is_range(selection):
            yield self._read_range(selection).set_operator(operator)
        elif _is_pair(selection):
            yield self._read_pair(selection).set_operator(operator)
        elif isinstance(selection, select.Operation):
            yield from self._evaluate_operation(selection, operator)
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

    def _evaluate_operation(self, operation, operator):
        if not operation.unary():
            yield from self._get_all_slices(operation.left_operand, operator)
        operator = _merge_operator(operator, operation.operator)
        yield from self._get_all_slices(operation.right_operand, operation.operator)


def _merge_operator(first_operator, second_operator):
    if first_operator == second_operator:
        return "+"
    else:
        return "-"


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
        self.factor *= 1 if operator == "+" else -1
        return self

    @property
    def indices(self):
        return tuple(self._indices)

    def label(self, index, axes, number_labels):
        if index == 0:
            factor = "" if self.factor == 1 else "-"
        else:
            factor = "+ " if self.factor == 1 else "- "
        return factor + "_".join(self._parse_keys(axes, number_labels))

    def _parse_keys(self, axes, number_labels):
        for axis in axes:
            if key := self._keys[axis]:
                yield self._parse_key(key, number_labels)

    def _parse_key(self, key, number_labels):
        if _is_range(key):
            return str(key)
        if key.isdecimal():
            return number_labels[key]
        return key


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


def _raise_key_not_found_error(key, error):
    message = f"Could not read {key}, please check the spelling and capitalization."
    raise exception.IncorrectUsage(message) from error


def _raise_error_if_list_not_empty(list_, message):
    if list_:
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

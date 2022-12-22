# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import inspect

import pytest

from py4vasp import exception
from py4vasp._data import slice_
from py4vasp._util import documentation


class Other:
    def __init__(self, *args, **kwargs):
        self._range = range(10)
        self._args = args
        self._kwargs = kwargs
        print("other", args, kwargs)


@documentation.format(examples=slice_.examples("example"))
class ExampleSlice(slice_.Mixin, Other):
    "{examples}"

    def steps(self):
        return self._steps

    def slice(self):
        return self._slice

    def is_slice(self):
        return self._is_slice

    def last_step_in_slice(self):
        return self._last_step_in_slice

    def range_steps(self):
        return self._range[self._steps]

    def range_slice(self):
        return self._range[self._slice]


@pytest.fixture
def single_step():
    return ExampleSlice()[0]


@pytest.fixture
def last_step():
    return ExampleSlice()


@pytest.fixture
def all_steps():
    return ExampleSlice()[:]


@pytest.fixture
def subset_of_steps():
    return ExampleSlice()[1:4]


@pytest.fixture
def stride():
    return ExampleSlice()[::3]


def test_pass_arguments_to_other():
    example = ExampleSlice("positional", key="word")
    assert example._args == ("positional",)
    assert example._kwargs == {"key": "word"}


def test_access_single_step(single_step):
    assert single_step.steps() == 0
    assert single_step.slice() == slice(0, 1)


def test_access_last_step(last_step):
    assert last_step.steps() == -1
    assert last_step.slice() == slice(-1, None)


def test_access_all_steps(all_steps):
    assert all_steps.steps() == slice(None)
    assert all_steps.slice() == slice(None)


def test_access_subset_of_steps(subset_of_steps):
    assert subset_of_steps.steps() == slice(1, 4)
    assert subset_of_steps.slice() == slice(1, 4)


def test_access_stride(stride):
    assert stride.steps() == slice(None, None, 3)
    assert stride.slice() == slice(None, None, 3)


def test_range_single_step(single_step):
    assert single_step.range_steps() == 0
    assert single_step.range_slice() == range(0, 1)


def test_range_last_step(last_step):
    assert last_step.range_steps() == 9
    assert last_step.range_slice() == range(9, 10)


def test_range_all_steps(all_steps):
    assert all_steps.range_steps() == range(0, 10)
    assert all_steps.range_slice() == range(0, 10)


def test_range_subset_of_steps(subset_of_steps):
    assert subset_of_steps.range_steps() == range(1, 4)
    assert subset_of_steps.range_slice() == range(1, 4)


def test_range_stride(stride):
    assert stride.range_steps() == range(0, 10, 3)
    assert stride.range_slice() == range(0, 10, 3)


def test_copy_created():
    example = ExampleSlice()
    assert example.steps() == -1
    first_step = example[0]
    assert first_step.steps() == 0
    assert example.steps() == -1
    assert first_step.steps() == 0


def test_is_slice_single_step(single_step):
    assert not single_step.is_slice()


def test_is_slice_last_step(last_step):
    assert not last_step.is_slice()


def test_is_slice_all_steps(all_steps):
    assert all_steps.is_slice()


def test_is_slice_subset_of_steps(subset_of_steps):
    assert subset_of_steps.is_slice()


def test_is_slice_stride(stride):
    assert stride.is_slice()


def test_last_step_in_slice_single_step(single_step):
    assert single_step.last_step_in_slice() == 0


def test_last_step_in_slice_last_step(last_step):
    assert last_step.last_step_in_slice() == -1


def test_last_step_in_slice_all_steps(all_steps):
    assert all_steps.last_step_in_slice() == -1


def test_last_step_in_slice_subset_of_steps(subset_of_steps):
    assert subset_of_steps.last_step_in_slice() == 3


def test_last_step_in_slice_stride(stride):
    assert stride.last_step_in_slice() == -1


def test_incorrect_argument():
    with pytest.raises(exception.IncorrectUsage):
        ExampleSlice()["step not an integer"]


def test_documentation(single_step, last_step):
    reference = slice_.examples("example")
    assert inspect.getdoc(single_step) == reference
    assert inspect.getdoc(last_step) == reference


def test_nested_slices(subset_of_steps):
    with pytest.raises(exception.NotImplemented):
        subset_of_steps[:]

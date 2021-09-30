from py4vasp.data._base import RefinementDescriptor
from py4vasp.data._trajectory import DataTrajectory, trajectory_examples
import py4vasp.exceptions as exception
import py4vasp._util.documentation as _documentation
import inspect
import pytest


@_documentation.add(trajectory_examples("dataimpl"))
class DataImpl(DataTrajectory):
    def steps(self):
        return self._steps

    def slice(self):
        return self._slice

    range_steps = RefinementDescriptor("_range_steps")
    range_slice = RefinementDescriptor("_range_slice")

    def _range_steps(self):
        return self._raw_data[self._steps]

    def _range_slice(self):
        return self._raw_data[self._slice]


@pytest.fixture
def single_step():
    return DataImpl(range(10))[0]


@pytest.fixture
def last_step():
    return DataImpl(range(10))


@pytest.fixture
def all_steps():
    return DataImpl(range(10))[:]


@pytest.fixture
def subset_of_steps():
    return DataImpl(range(10))[1:4]


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


def test_copy_created():
    trajectory = DataImpl(range(10))
    assert trajectory.steps() == -1
    first_step = trajectory[0]
    assert first_step.steps() == 0
    assert trajectory.steps() == -1
    assert first_step.steps() == 0


def test_is_slice_single_step(single_step):
    assert not single_step._is_slice


def test_is_slice_last_step(last_step):
    assert not last_step._is_slice


def test_is_slice_all_steps(all_steps):
    assert all_steps._is_slice


def test_is_slice_subset_of_steps(subset_of_steps):
    assert subset_of_steps._is_slice


def test_last_step_in_slice_single_step(single_step):
    assert single_step._last_step_in_slice == 0


def test_last_step_in_slice_last_step(last_step):
    assert last_step._last_step_in_slice == -1


def test_last_step_in_slice_all_steps(all_steps):
    assert all_steps._last_step_in_slice == -1


def test_last_step_in_slice_subset_of_steps(subset_of_steps):
    assert subset_of_steps._last_step_in_slice == 3


def test_incorrect_argument(all_steps):
    with pytest.raises(exception.IncorrectUsage):
        all_steps["step not an integer"]


def test_documentation(single_step, last_step):
    reference = trajectory_examples("dataimpl")
    assert inspect.getdoc(single_step) == reference
    assert inspect.getdoc(last_step) == reference

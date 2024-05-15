# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import tempfile
from unittest.mock import MagicMock

import hypothesis.extra.numpy as np_strat
import hypothesis.strategies as strategy
import numpy as np
import pytest
from hypothesis import assume, given

from py4vasp import exception
from py4vasp.raw import VaspData

threshold = 100.0


@strategy.composite
def operands(draw):
    (shape_x, shape_y), _ = draw(np_strat.mutually_broadcastable_shapes(num_shapes=2))
    x = draw_test_data(draw, shape_x)
    y = draw_test_data(draw, shape_y, positive=True)

    return x, y


@strategy.composite
def array_or_scalar(draw):
    (shape,), _ = draw(np_strat.mutually_broadcastable_shapes(num_shapes=1))
    return draw_test_data(draw, shape)


@strategy.composite
def array_and_slice(draw):
    shape = draw(np_strat.array_shapes())
    array = draw_test_data(draw, shape)
    slice = draw(np_strat.basic_indices(shape))
    return array, slice


@strategy.composite
def complex_array_or_scalar(draw):
    (complex_shape,), _ = draw(np_strat.mutually_broadcastable_shapes(num_shapes=1))
    real_shape = (*complex_shape, 2)
    return draw_test_data(draw, real_shape)


def draw_test_data(draw, shape, positive=False):
    min_value = 1 / threshold if positive else -threshold
    elements = strategy.floats(min_value=min_value, max_value=threshold)
    if len(shape) == 0:
        result = draw(elements)
    else:
        result = draw(np_strat.arrays(float, shape, elements=elements))
    return np.sign(result) * np.maximum(np.abs(result), 1 / threshold)


@given(ops=operands())
def test_operators(ops, Assert):
    ref_x, ref_y = ops
    vasp_x, vasp_y = VaspData(ref_x), VaspData(ref_y)
    Assert.allclose(vasp_x + vasp_y, ref_x + ref_y)
    Assert.allclose(vasp_x - vasp_y, ref_x - ref_y)
    Assert.allclose(vasp_x * vasp_y, ref_x * ref_y)
    Assert.allclose(np.abs(vasp_x) ** vasp_y, np.abs(ref_x) ** ref_y)
    assert np.all((vasp_x == vasp_y) == (ref_x == ref_y))
    assert np.all((vasp_x != vasp_y) == (ref_x != ref_y))
    assert np.all((vasp_x > vasp_y) == (ref_x > ref_y))
    assert np.all((vasp_x < vasp_y) == (ref_x < ref_y))
    assert np.all((vasp_x >= vasp_y) == (ref_x >= ref_y))
    assert np.all((vasp_x <= vasp_y) == (ref_x <= ref_y))
    assume(np.all(ref_y) != 0)
    Assert.allclose(vasp_x / vasp_y, ref_x / ref_y)
    Assert.allclose(vasp_x % vasp_y, ref_x % ref_y)
    Assert.allclose(vasp_x // vasp_y, ref_x // ref_y)


@given(ops=operands())
def test_functions(ops, Assert):
    ref_x, ref_y = ops
    vasp_x, vasp_y = VaspData(ref_x), VaspData(ref_y)
    Assert.allclose(np.sin(vasp_x), np.sin(ref_x))
    Assert.allclose(np.cos(vasp_x), np.cos(ref_x))
    Assert.allclose(np.tan(vasp_x), np.tan(ref_x))
    Assert.allclose(np.arcsin(vasp_x / threshold), np.arcsin(ref_x / threshold))
    Assert.allclose(np.arccos(vasp_x / threshold), np.arccos(ref_x / threshold))
    Assert.allclose(np.arctan(vasp_x), np.arctan(ref_x))
    Assert.allclose(np.abs(vasp_x), np.abs(ref_x))
    Assert.allclose(np.linalg.norm(vasp_x), np.linalg.norm(ref_x))
    Assert.allclose(np.outer(vasp_x, vasp_y), np.outer(ref_x, ref_y))
    Assert.allclose(np.maximum(vasp_x, vasp_y), np.maximum(ref_x, ref_y))
    Assert.allclose(np.minimum(vasp_x, vasp_y), np.minimum(ref_x, ref_y))


@given(data=array_or_scalar())
def test_attributes(data):
    # wrap a small amount of properties common to hdf5 and ndarray
    vasp = VaspData(data)
    assert vasp.ndim == data.ndim
    assert vasp.size == data.size
    assert vasp.shape == data.shape
    assert vasp.dtype == data.dtype
    assert repr(vasp) == f"VaspData({repr(data)})"
    assume(data.ndim > 0)
    assert len(vasp) == len(data)
    assert not vasp.is_none()


@given(data=array_or_scalar())
def test_conversion(data, Assert):
    vasp = VaspData(data)
    Assert.allclose(vasp.astype(np.float32), data.astype(np.float32))
    Assert.allclose(np.array(vasp), data)
    Assert.allclose(np.asarray(vasp, np.float16), np.asarray(vasp, np.float16))


@given(array_slice=array_and_slice())
def test_slices(array_slice, Assert):
    array, slice = array_slice
    vasp = VaspData(array)
    Assert.allclose(vasp[slice], array[slice])


@pytest.mark.parametrize(
    "function",
    [
        lambda vasp: np.array(vasp),
        lambda vasp: vasp[:],
        lambda vasp: len(vasp),
        lambda vasp: vasp.ndim,
        lambda vasp: vasp.size,
        lambda vasp: vasp.shape,
        lambda vasp: vasp.dtype,
    ],
)
def test_missing_data(function):
    vasp = VaspData(None)
    with pytest.raises(exception.NoData):
        function(vasp)
    assert vasp.is_none()


def test_scalar_data():
    reference = 1
    mock = MagicMock()
    mock.ndim = 0
    mock.__array__ = lambda: np.array(reference)
    vasp = VaspData(mock)
    assert vasp == reference
    assert np.array(vasp) == reference
    assert vasp[()] == reference
    assert vasp.ndim == 0
    assert vasp.size == 1
    assert vasp.shape == ()
    assert vasp.dtype == np.array(reference).dtype
    assert repr(vasp) == f"VaspData({repr(mock)})"


def test_scalar_string():
    reference = "text stored in file"
    vasp = VaspData(np.array(reference.encode()))
    assert vasp[()] == reference


def test_list_data(Assert):
    data = [[1, 2, 3], [4, 5, 6]]
    reference = np.array(data)
    actual = VaspData(data)
    Assert.allclose(actual, reference)
    Assert.allclose(np.array(actual), reference)
    Assert.allclose(actual[()], reference)
    assert actual.ndim == reference.ndim
    assert actual.size == reference.size
    assert actual.shape == reference.shape
    assert actual.dtype == reference.dtype
    assert repr(actual) == f"VaspData({repr(data)})"


def test_nested_data():
    data = VaspData(VaspData(None))
    assert data.is_none()
    assert repr(data) == "VaspData(None)"
    zeros = np.zeros(10)
    data = VaspData(VaspData(zeros))
    assert repr(data) == f"VaspData({repr(zeros)})"


@given(data=complex_array_or_scalar())
def test_complex_from_numpy(data, Assert):
    data = data.view(np.complex128).reshape(data.shape[:-1])
    vasp = VaspData(data)
    assert vasp.ndim == data.ndim
    assert vasp.size == data.size
    assert vasp.shape == data.shape
    assert vasp.dtype == data.dtype
    assert not vasp.is_none()
    Assert.allclose(vasp + vasp, data + data)
    Assert.allclose(3 * vasp, 3 * data)
    Assert.allclose(np.sin(vasp), np.sin(data))
    Assert.allclose(np.cos(vasp), np.cos(data))
    Assert.allclose(np.tan(vasp), np.tan(data))


class MockComplex(np.ndarray):
    attrs = {"dtype": "complex"}


@given(data=complex_array_or_scalar())
def test_complex_from_dataset(data, Assert):
    view = data.view(np.complex128).reshape(data.shape[:-1])
    mock_complex = MockComplex(shape=data.shape, buffer=data.data)
    vasp = VaspData(mock_complex)
    assert vasp.ndim == view.ndim
    assert vasp.size == view.size
    assert vasp.shape == view.shape
    assert vasp.dtype == view.dtype
    assert not vasp.is_none()
    Assert.allclose(vasp + vasp, view + view)
    Assert.allclose(3 * vasp, 3 * view)
    Assert.allclose(np.sin(vasp), np.sin(view))
    Assert.allclose(vasp.astype(np.complex64), view.astype(np.complex64))
    assert repr(vasp) == f"VaspData({repr(mock_complex)})"
    # make sure no copies are created, i.e., when I modify the original data, the
    # VaspData changes accordingly
    copy = view.copy()
    mock_complex += 1.0
    Assert.allclose(vasp, copy + (1 + 1j))

from py4vasp.raw import VaspData
from hypothesis import given, assume
import hypothesis.strategies as strategy
import hypothesis.extra.numpy as np_strat
import numpy as np


threshold = 100.0


@strategy.composite
def operands(draw):
    (shape_x, shape_y), _ = draw(np_strat.mutually_broadcastable_shapes(num_shapes=2))
    x = draw_test_data(draw, shape_x)
    y = draw_test_data(draw, shape_y)
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


def draw_test_data(draw, shape):
    elements = strategy.floats(min_value=-threshold, max_value=threshold)
    if len(shape) == 0:
        result = draw(elements)
    else:
        result = draw(np_strat.arrays(np.float, shape, elements=elements))
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


@given(array_slice=array_and_slice())
def test_slices(array_slice, Assert):
    array, slice = array_slice
    vasp = VaspData(array)
    Assert.allclose(vasp[slice], array[slice])

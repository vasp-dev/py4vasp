from py4vasp.raw import VaspData
from hypothesis import given, assume
import hypothesis.strategies as strategy
from hypothesis.extra.numpy import mutually_broadcastable_shapes, array_shapes, arrays
import numpy as np


threshold = 100.0


@strategy.composite
def operands(draw):
    (shape_x, shape_y), _ = draw(mutually_broadcastable_shapes(num_shapes=2))
    x = array_or_scalar(draw, shape_x)
    y = array_or_scalar(draw, shape_y)
    return x, y


def array_or_scalar(draw, shape):
    elements = strategy.floats(min_value=-threshold, max_value=threshold)
    if len(shape) == 0:
        result = draw(elements)
    else:
        result = draw(arrays(np.float, shape, elements=elements))
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

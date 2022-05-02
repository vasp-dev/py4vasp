from py4vasp.raw import VaspData
from hypothesis import given
import hypothesis.strategies as strategy
from hypothesis.extra.numpy import mutually_broadcastable_shapes, arrays
import numpy as np
import warnings


@strategy.composite
def operands(draw):
    (shape_x, shape_y), _ = draw(mutually_broadcastable_shapes(num_shapes=2))
    x = array_or_scalar(draw, shape_x)
    y = array_or_scalar(draw, shape_y)
    return x, y


def array_or_scalar(draw, shape):
    if len(shape) == 0:
        return draw(strategy.floats())
    else:
        return draw(arrays(np.float, shape))


@given(ops=operands())
def test_operators(ops, Assert):
    ref_x, ref_y = ops
    vasp_x, vasp_y = VaspData(ref_x), VaspData(ref_y)
    with warnings.catch_warnings():
        # NaN of infinite may be tested, so we ignore the warnings
        warnings.simplefilter("ignore")
        Assert.allclose(vasp_x + vasp_y, ref_x + ref_y)
        Assert.allclose(vasp_x - vasp_y, ref_x - ref_y)
        Assert.allclose(vasp_x * vasp_y, ref_x * ref_y)
        assert np.all((vasp_x == vasp_y) == (ref_x == ref_y))
        assert np.all((vasp_x != vasp_y) == (ref_x != ref_y))
        assert np.all((vasp_x > vasp_y) == (ref_x > ref_y))
        assert np.all((vasp_x < vasp_y) == (ref_x < ref_y))
        assert np.all((vasp_x >= vasp_y) == (ref_x >= ref_y))
        assert np.all((vasp_x <= vasp_y) == (ref_x <= ref_y))

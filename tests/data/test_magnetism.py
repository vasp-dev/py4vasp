from py4vasp.data import Magnetism, _util
from .test_topology import raw_topology
import py4vasp.raw as raw
import numpy as np
import pytest


@pytest.fixture
def raw_magnetism(raw_topology):
    number_steps = 4
    number_atoms = len(raw_topology.elements)
    lmax = 3
    number_components = 2
    shape = (number_steps, number_components, number_atoms, lmax)
    magnetism = raw.Magnetism(moments=np.arange(np.prod(shape)).reshape(shape))
    magnetism.charges = magnetism.moments[:, 0, :, :]
    magnetism.total_charges = np.sum(magnetism.charges, axis=2)
    magnetism.magnetic_moments = magnetism.moments[:, 1, :, :]
    magnetism.total_moments = np.sum(magnetism.magnetic_moments, axis=2)
    return magnetism


@pytest.fixture
def noncollinear_magnetism(raw_magnetism):
    shape = raw_magnetism.moments.shape
    shape = (shape[0] // 2, shape[1] * 2, shape[2], shape[3])
    raw_magnetism.moments = raw_magnetism.moments.reshape(shape)
    return raw_magnetism


@pytest.fixture
def charge_only(raw_magnetism):
    shape = raw_magnetism.moments.shape
    shape = (shape[0] * shape[1], 1, shape[2], shape[3])
    raw_magnetism.moments = raw_magnetism.moments.reshape(shape)
    return raw_magnetism


def test_from_file(raw_magnetism, mock_file, check_read):
    with mock_file("magnetism", raw_magnetism) as mocks:
        check_read(Magnetism, mocks, raw_magnetism)


def test_read(raw_magnetism, Assert):
    actual = Magnetism(raw_magnetism).read()
    Assert.allclose(actual["charges"], raw_magnetism.charges)
    Assert.allclose(actual["moments"], raw_magnetism.magnetic_moments)
    actual = Magnetism(raw_magnetism).read(-1)
    Assert.allclose(actual["charges"], raw_magnetism.charges[-1])
    Assert.allclose(actual["moments"], raw_magnetism.magnetic_moments[-1])


def test_charges(raw_magnetism, Assert):
    actual = Magnetism(raw_magnetism).charges()
    Assert.allclose(actual, raw_magnetism.charges)


def test_moments(raw_magnetism, Assert):
    actual = Magnetism(raw_magnetism).moments()
    Assert.allclose(actual, raw_magnetism.magnetic_moments)


def test_total_charges(raw_magnetism, Assert):
    actual = Magnetism(raw_magnetism).total_charges()
    Assert.allclose(actual, raw_magnetism.total_charges)
    actual = Magnetism(raw_magnetism).total_charges(range(2))
    Assert.allclose(actual, raw_magnetism.total_charges[0:2])


def test_total_moments(raw_magnetism, Assert):
    actual = Magnetism(raw_magnetism).total_moments()
    Assert.allclose(actual, raw_magnetism.total_moments)
    actual = Magnetism(raw_magnetism).total_moments(3)
    Assert.allclose(actual, raw_magnetism.total_moments[3])


def test_print_magnetism(raw_magnetism):
    actual, _ = _util.format_(Magnetism(raw_magnetism))
    reference = "MAGMOM = 444.00 453.00 462.00 471.00 480.00 489.00 498.00"
    assert actual == {"text/plain": reference}


def test_noncollinear(noncollinear_magnetism, Assert):
    actual = Magnetism(noncollinear_magnetism)
    Assert.allclose(actual.charges(), noncollinear_magnetism.moments[:, 0])
    step = 0
    moments = actual.moments(step)
    for new_order in np.ndindex(moments.shape):
        atom, orbital, component = new_order
        old_order = (step, component + 1, atom, orbital)  # 0 component is charge
        Assert.allclose(moments[new_order], noncollinear_magnetism.moments[old_order])
    total_moments = actual.total_moments()
    for new_order in np.ndindex(total_moments.shape):
        step, atom, component = new_order
        old_order = (step, component + 1, atom)  # 0 component is charge
        expected_total_moment = np.sum(noncollinear_magnetism.moments[old_order])
        Assert.allclose(total_moments[new_order], expected_total_moment)


def test_print_noncollinear(noncollinear_magnetism):
    actual, _ = _util.format_(Magnetism(noncollinear_magnetism))
    reference = """
MAGMOM = 318.00 381.00 444.00 \\
         327.00 390.00 453.00 \\
         336.00 399.00 462.00 \\
         345.00 408.00 471.00 \\
         354.00 417.00 480.00 \\
         363.00 426.00 489.00 \\
         372.00 435.00 498.00
    """.strip()
    assert actual == {"text/plain": reference}


def test_charge_only(charge_only, Assert):
    actual = Magnetism(charge_only)
    Assert.allclose(actual.charges(), charge_only.moments[:, 0])
    assert actual.moments() is None
    assert actual.total_moments() is None
    assert "moments" not in actual.read()


def test_print_charge(charge_only):
    actual, _ = _util.format_(Magnetism(charge_only))
    assert actual == {"text/plain": "not available"}

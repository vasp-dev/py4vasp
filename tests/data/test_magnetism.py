from py4vasp.data import Magnetism
import py4vasp.exceptions as exception
import numpy as np
import pytest
import types


@pytest.fixture
def collinear_magnetism(raw_data):
    raw_magnetism = raw_data.magnetism("collinear")
    magnetism = Magnetism(raw_magnetism)
    magnetism.ref = types.SimpleNamespace()
    magnetism.ref.charges = raw_magnetism.moments[:, 0, :, :]
    magnetism.ref.moments = raw_magnetism.moments[:, 1, :, :]
    return magnetism


@pytest.fixture
def noncollinear_magnetism(raw_data):
    raw_magnetism = raw_data.magnetism("noncollinear")
    magnetism = Magnetism(raw_magnetism)
    magnetism.ref = types.SimpleNamespace()
    magnetism.ref.charges = raw_magnetism.moments[:, 0, :, :]
    magnetism.ref.moments = np.moveaxis(raw_magnetism.moments[:, 1:4, :, :], 1, 3)
    return magnetism


@pytest.fixture
def charge_only(raw_data):
    raw_magnetism = raw_data.magnetism("charge_only")
    magnetism = Magnetism(raw_magnetism)
    magnetism.ref = types.SimpleNamespace()
    magnetism.ref.charges = raw_magnetism.moments[:, 0, :, :]
    magnetism.ref.moments = None
    return magnetism


@pytest.fixture
def all_magnetism(collinear_magnetism, noncollinear_magnetism, charge_only):
    magnetism = types.SimpleNamespace()
    magnetism.collinear = collinear_magnetism
    magnetism.noncollinear = noncollinear_magnetism
    magnetism.charge_only = charge_only
    return magnetism


def test_read(all_magnetism, Assert):
    check_read(all_magnetism.collinear, Assert)
    check_read(all_magnetism.noncollinear, Assert)
    check_read(all_magnetism.charge_only, Assert)


def check_read(magnetism, Assert):
    actual = magnetism.read()
    print(actual["charges"])
    Assert.allclose(actual["charges"], magnetism.ref.charges)
    Assert.allclose(actual["moments"], magnetism.ref.moments)
    actual = magnetism.read(-1)
    Assert.allclose(actual["charges"], magnetism.ref.charges[-1])
    if magnetism.ref.moments is not None:
        Assert.allclose(actual["moments"], magnetism.ref.moments[-1])


def test_charges(all_magnetism, Assert):
    check_charges(all_magnetism.collinear, Assert)
    check_charges(all_magnetism.noncollinear, Assert)
    check_charges(all_magnetism.charge_only, Assert)


def check_charges(magnetism, Assert):
    Assert.allclose(magnetism.charges(), magnetism.ref.charges)


def test_moments(all_magnetism, Assert):
    check_moments(all_magnetism.collinear, Assert)
    check_moments(all_magnetism.noncollinear, Assert)
    check_moments(all_magnetism.charge_only, Assert)


def check_moments(magnetism, Assert):
    Assert.allclose(magnetism.moments(), magnetism.ref.moments)


def test_total_charges(all_magnetism, Assert):
    check_total_charges(all_magnetism.collinear, Assert)
    check_total_charges(all_magnetism.noncollinear, Assert)
    check_total_charges(all_magnetism.charge_only, Assert)


def check_total_charges(magnetism, Assert):
    total_charges = np.sum(magnetism.ref.charges, axis=2)
    Assert.allclose(magnetism.total_charges(), total_charges)
    Assert.allclose(magnetism.total_charges(range(1, 3)), total_charges[1:3])


def test_total_moments(collinear_magnetism, noncollinear_magnetism, Assert):
    check_total_moments(collinear_magnetism, Assert)
    check_total_moments(noncollinear_magnetism, Assert)


def check_total_moments(magnetism, Assert):
    total_moments = np.sum(magnetism.ref.moments, axis=2)
    Assert.allclose(magnetism.total_moments(), total_moments)
    Assert.allclose(magnetism.total_moments(3), total_moments[3])


def test_charge_only_total_moments(charge_only):
    assert charge_only.total_moments() is None
    assert charge_only.total_moments(3) is None


def test_collinear_print(collinear_magnetism, format_):
    actual, _ = format_(collinear_magnetism)
    reference = "MAGMOM = 444.00 453.00 462.00 471.00 480.00 489.00 498.00"
    assert actual == {"text/plain": reference}


def test_noncollinear_print(noncollinear_magnetism, format_):
    actual, _ = format_(noncollinear_magnetism)
    print(actual["text/plain"])
    reference = """
MAGMOM = 822.00 885.00 948.00 \\
         831.00 894.00 957.00 \\
         840.00 903.00 966.00 \\
         849.00 912.00 975.00 \\
         858.00 921.00 984.00 \\
         867.00 930.00 993.00 \\
         876.00 939.00 1002.00
    """.strip()
    assert actual == {"text/plain": reference}


def test_charge_only_print(charge_only, format_):
    actual, _ = format_(charge_only)
    assert actual == {"text/plain": "not spin polarized"}


def test_nonexisting_magnetism():
    with pytest.raises(exception.NoData):
        magnetism = Magnetism(None).read()


def test_incorrect_argument(all_magnetism):
    check_incorrect_argument(all_magnetism.collinear)
    check_incorrect_argument(all_magnetism.noncollinear)
    check_incorrect_argument(all_magnetism.charge_only)


def check_incorrect_argument(magnetism):
    with pytest.raises(exception.IncorrectUsage):
        magnetism.read("index not an integer")
    out_of_bounds = 999
    with pytest.raises(exception.IncorrectUsage):
        magnetism.moments(out_of_bounds)
    with pytest.raises(exception.IncorrectUsage):
        magnetism.total_moments(out_of_bounds)
    with pytest.raises(exception.IncorrectUsage):
        magnetism.charges(out_of_bounds)
    with pytest.raises(exception.IncorrectUsage):
        magnetism.total_charges(out_of_bounds)


def test_descriptor(collinear_magnetism, check_descriptors):
    descriptors = {
        "_to_dict": ["to_dict", "read"],
        "_charges": ["charges"],
        "_moments": ["moments"],
        "_total_charges": ["total_charges"],
        "_total_moments": ["total_moments"],
    }
    check_descriptors(collinear_magnetism, descriptors)


def test_from_file(raw_data, mock_file, check_read):
    raw_magnetism = raw_data.magnetism("collinear")
    with mock_file("magnetism", raw_magnetism) as mocks:
        check_read(Magnetism, mocks, raw_magnetism)

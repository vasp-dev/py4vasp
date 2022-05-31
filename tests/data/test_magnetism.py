# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp.data import Magnetism
import py4vasp.exceptions as exception
import numpy as np
import pytest
import types
from unittest.mock import patch


@pytest.fixture
def collinear_magnetism(raw_data):
    raw_magnetism = raw_data.magnetism("collinear")
    magnetism = Magnetism.from_data(raw_magnetism)
    magnetism.ref = types.SimpleNamespace()
    magnetism.ref.charges = raw_magnetism.moments[:, 0, :, :]
    magnetism.ref.moments = raw_magnetism.moments[:, 1, :, :]
    return magnetism


@pytest.fixture
def noncollinear_magnetism(raw_data):
    raw_magnetism = raw_data.magnetism("noncollinear")
    magnetism = Magnetism.from_data(raw_magnetism)
    magnetism.ref = types.SimpleNamespace()
    magnetism.ref.charges = raw_magnetism.moments[:, 0, :, :]
    magnetism.ref.moments = np.moveaxis(raw_magnetism.moments[:, 1:4, :, :], 1, 3)
    return magnetism


@pytest.fixture
def charge_only(raw_data):
    class GetItemNone:
        def __getitem__(self, step):
            return None

    raw_magnetism = raw_data.magnetism("charge_only")
    magnetism = Magnetism.from_data(raw_magnetism)
    magnetism.ref = types.SimpleNamespace()
    magnetism.ref.charges = raw_magnetism.moments[:, 0, :, :]
    magnetism.ref.moments = GetItemNone()
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
    check_read_all_steps(magnetism, Assert)
    check_read_subset_of_steps(magnetism, Assert)
    check_read_specific_step(magnetism, Assert)
    check_read_last_step(magnetism, Assert)


def check_read_all_steps(magnetism, Assert):
    actual = magnetism[:].read()
    Assert.allclose(actual["charges"], magnetism.ref.charges[:])
    Assert.allclose(actual["moments"], magnetism.ref.moments[:])


def check_read_subset_of_steps(magnetism, Assert):
    subset = slice(1, 3)
    actual = magnetism[subset].read()
    Assert.allclose(actual["charges"], magnetism.ref.charges[subset])
    Assert.allclose(actual["moments"], magnetism.ref.moments[subset])


def check_read_specific_step(magnetism, Assert):
    actual = magnetism[0].read()
    Assert.allclose(actual["charges"], magnetism.ref.charges[0])
    Assert.allclose(actual["moments"], magnetism.ref.moments[0])


def check_read_last_step(magnetism, Assert):
    actual = magnetism.read()
    Assert.allclose(actual["charges"], magnetism.ref.charges[-1])
    Assert.allclose(actual["moments"], magnetism.ref.moments[-1])


def test_charges(all_magnetism, Assert):
    check_charges(all_magnetism.collinear, Assert)
    check_charges(all_magnetism.noncollinear, Assert)
    check_charges(all_magnetism.charge_only, Assert)


def check_charges(magnetism, Assert):
    Assert.allclose(magnetism[:].charges(), magnetism.ref.charges[:])
    Assert.allclose(magnetism[1:3].charges(), magnetism.ref.charges[1:3])
    Assert.allclose(magnetism[0].charges(), magnetism.ref.charges[0])
    Assert.allclose(magnetism.charges(), magnetism.ref.charges[-1])


def test_moments(all_magnetism, Assert):
    check_moments(all_magnetism.collinear, Assert)
    check_moments(all_magnetism.noncollinear, Assert)
    check_moments(all_magnetism.charge_only, Assert)


def check_moments(magnetism, Assert):
    Assert.allclose(magnetism[:].moments(), magnetism.ref.moments[:])
    Assert.allclose(magnetism[1:3].moments(), magnetism.ref.moments[1:3])
    Assert.allclose(magnetism[0].moments(), magnetism.ref.moments[0])
    Assert.allclose(magnetism.moments(), magnetism.ref.moments[-1])


def test_total_charges(all_magnetism, Assert):
    check_total_charges(all_magnetism.collinear, Assert)
    check_total_charges(all_magnetism.noncollinear, Assert)
    check_total_charges(all_magnetism.charge_only, Assert)


def check_total_charges(magnetism, Assert):
    total_charges = np.sum(magnetism.ref.charges, axis=2)
    Assert.allclose(magnetism[:].total_charges(), total_charges[:])
    Assert.allclose(magnetism[1:3].total_charges(), total_charges[1:3])
    Assert.allclose(magnetism[0].total_charges(), total_charges[0])
    Assert.allclose(magnetism.total_charges(), total_charges[-1])


def test_total_moments(collinear_magnetism, noncollinear_magnetism, Assert):
    check_total_moments(collinear_magnetism, Assert)
    check_total_moments(noncollinear_magnetism, Assert)


def check_total_moments(magnetism, Assert):
    total_moments = np.sum(magnetism.ref.moments, axis=2)
    Assert.allclose(magnetism[:].total_moments(), total_moments[:])
    Assert.allclose(magnetism[1:3].total_moments(), total_moments[1:3])
    Assert.allclose(magnetism[0].total_moments(), total_moments[0])
    Assert.allclose(magnetism.total_moments(), total_moments[-1])


def test_charge_only_total_moments(charge_only):
    assert charge_only[:].total_moments() is None
    assert charge_only[1:3].total_moments() is None
    assert charge_only[0].total_moments() is None
    assert charge_only.total_moments() is None


def test_plot_collinear_magnetism(collinear_magnetism, Assert):
    total_moments = np.sum(collinear_magnetism.ref.moments, axis=2)
    for step in (0, -1):
        actual_moments = get_show_arrow_args(collinear_magnetism, step)
        reference_moments = total_moments[step]
        check_plot_collinear_magnetism(actual_moments, reference_moments, Assert)
    check_slices_not_implemented(collinear_magnetism)


def test_plot_noncollinear_magnetism(noncollinear_magnetism, Assert):
    total_moments = np.sum(noncollinear_magnetism.ref.moments, axis=2)
    for step in (0, -1):
        actual_moments = get_show_arrow_args(noncollinear_magnetism, step)
        reference_moments = total_moments[step]
        check_plot_noncollinear_magnetism(actual_moments, reference_moments, Assert)
    check_slices_not_implemented(noncollinear_magnetism)


def get_show_arrow_args(magnetism, step):
    with patch("py4vasp.data.Structure._to_viewer3d") as plot:
        if step == -1:
            magnetism.plot()
        else:
            magnetism[step].plot()
        plot.assert_called_once()
        viewer = plot.return_value
        viewer.show_arrows_at_atoms.assert_called_once()
        args, _ = viewer.show_arrows_at_atoms.call_args
    return args[0]


def check_plot_collinear_magnetism(actual_moments, reference_moments, Assert):
    rescale_moments = Magnetism.length_moments / np.max(reference_moments)
    for actual, reference in zip(actual_moments, reference_moments):
        Assert.allclose(actual, rescale_moments * np.array([0.0, 0.0, reference]))


def check_plot_noncollinear_magnetism(actual_moments, reference_moments, Assert):
    largest_moment = np.max(np.linalg.norm(reference_moments, axis=1))
    rescale_moments = Magnetism.length_moments / largest_moment
    Assert.allclose(actual_moments, rescale_moments * reference_moments)


def check_slices_not_implemented(magnetism):
    for steps in (slice(None), slice(1, 3)):
        with pytest.raises(exception.NotImplemented):
            magnetism[steps].plot()


def test_plot_charge_only(charge_only):
    with patch("py4vasp.data.Structure._to_viewer3d") as plot:
        charge_only.plot()
        plot.assert_called_once()
        viewer = plot.return_value
        viewer.show_arrows_at_atoms.assert_not_called()
    check_slices_not_implemented(charge_only)


def test_collinear_print(collinear_magnetism, format_):
    actual, _ = format_(collinear_magnetism)
    reference = "MAGMOM = 444.00 453.00 462.00 471.00 480.00 489.00 498.00"
    assert actual == {"text/plain": reference}


def test_noncollinear_print(noncollinear_magnetism, format_):
    actual, _ = format_(noncollinear_magnetism)
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


def test_incorrect_argument(all_magnetism):
    check_incorrect_argument(all_magnetism.collinear)
    check_incorrect_argument(all_magnetism.noncollinear)
    check_incorrect_argument(all_magnetism.charge_only)


def check_incorrect_argument(magnetism):
    with pytest.raises(exception.IncorrectUsage):
        magnetism["step not an integer"].read()
    out_of_bounds = 999
    with pytest.raises(exception.IncorrectUsage):
        magnetism[out_of_bounds].moments()
    with pytest.raises(exception.IncorrectUsage):
        magnetism[out_of_bounds].total_moments()
    with pytest.raises(exception.IncorrectUsage):
        magnetism[out_of_bounds].charges()
    with pytest.raises(exception.IncorrectUsage):
        magnetism[out_of_bounds].total_charges()


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.magnetism("collinear")
    check_factory_methods(Magnetism, data)

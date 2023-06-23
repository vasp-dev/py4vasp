# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp import exception
from py4vasp.data import Magnetism


@pytest.fixture(params=[slice(None), slice(1, 3), 0, -1])
def slice_(request):
    return request.param


@pytest.fixture(params=["collinear", "noncollinear", "orbital_moments", "charge_only"])
def example_magnetism(raw_data, request):
    return setup_magnetism(raw_data, kind=request.param)


@pytest.fixture
def charge_only(raw_data):
    return setup_magnetism(raw_data, "charge_only")


@pytest.fixture
def collinear_magnetism(raw_data):
    return setup_magnetism(raw_data, "collinear")


@pytest.fixture
def noncollinear_magnetism(raw_data):
    return setup_magnetism(raw_data, "noncollinear")


@pytest.fixture
def orbital_moments(raw_data):
    return setup_magnetism(raw_data, "orbital_moments")


def setup_magnetism(raw_data, kind):
    raw_magnetism = raw_data.magnetism(kind)
    magnetism = Magnetism.from_data(raw_magnetism)
    magnetism.ref = types.SimpleNamespace()
    magnetism.ref.kind = kind
    magnetism.ref.charges = raw_magnetism.spin_moments[:, 0]
    set_moments(raw_magnetism, magnetism.ref)
    return magnetism


def set_moments(raw_magnetism, reference):
    class GetItemNone:
        def __getitem__(self, step):
            return None

    if reference.kind == "charge_only":
        reference.moments = GetItemNone()
    elif reference.kind == "collinear":
        reference.moments = raw_magnetism.spin_moments[:, 1]
    elif reference.kind == "noncollinear":
        reference.moments = np.moveaxis(raw_magnetism.spin_moments[:, 1:4], 1, 3)
    else:
        spin_moments = np.moveaxis(raw_magnetism.spin_moments[:, 1:4], 1, 3)
        orbital_moments = np.moveaxis(raw_magnetism.orbital_moments[:, 1:4], 1, 3)
        reference.moments = spin_moments + orbital_moments
        reference.spin_moments = spin_moments
        reference.orbital_moments = orbital_moments


@pytest.fixture
def all_magnetism(collinear_magnetism, noncollinear_magnetism, charge_only):
    magnetism = types.SimpleNamespace()
    magnetism.collinear = collinear_magnetism
    magnetism.noncollinear = noncollinear_magnetism
    magnetism.charge_only = charge_only
    return magnetism


def test_read(example_magnetism, slice_, Assert):
    magnetism = example_magnetism[slice_] if slice_ != -1 else example_magnetism
    actual = magnetism.read()
    Assert.allclose(actual["charges"], example_magnetism.ref.charges[slice_])
    Assert.allclose(actual["moments"], example_magnetism.ref.moments[slice_])


def test_read_spin_and_orbital_moments(orbital_moments, slice_, Assert):
    magnetism = orbital_moments[slice_] if slice_ != -1 else orbital_moments
    actual = magnetism.read()
    reference = orbital_moments.ref
    Assert.allclose(actual["spin_moments"], reference.spin_moments[slice_])
    Assert.allclose(actual["orbital_moments"], reference.orbital_moments[slice_])


def test_charges(example_magnetism, slice_, Assert):
    magnetism = example_magnetism[slice_] if slice_ != -1 else example_magnetism
    Assert.allclose(magnetism.charges(), example_magnetism.ref.charges[slice_])


def test_moments(example_magnetism, slice_, Assert):
    magnetism = example_magnetism[slice_] if slice_ != -1 else example_magnetism
    Assert.allclose(magnetism.moments(), example_magnetism.ref.moments[slice_])


def test_total_charges(example_magnetism, slice_, Assert):
    magnetism = example_magnetism[slice_] if slice_ != -1 else example_magnetism
    total_charges = np.sum(magnetism.ref.charges, axis=2)
    Assert.allclose(magnetism.total_charges(), total_charges[slice_])


def test_total_moments(example_magnetism, slice_, Assert):
    magnetism = example_magnetism[slice_] if slice_ != -1 else example_magnetism
    if example_magnetism.ref.kind == "charge_only":
        assert magnetism.total_moments() is None
    else:
        total_moments = np.sum(magnetism.ref.moments, axis=2)
        Assert.allclose(magnetism.total_moments(), total_moments[slice_])


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
    with patch("py4vasp.data.Structure.plot") as plot:
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
    with patch("py4vasp.data.Structure.plot") as plot:
        charge_only.plot()
        plot.assert_called_once()
        viewer = plot.return_value
        viewer.show_arrows_at_atoms.assert_not_called()
    check_slices_not_implemented(charge_only)


def test_plot_supercell(collinear_magnetism):
    supercell = (3, 2, 1)
    with patch("py4vasp.data.Structure.plot") as plot:
        collinear_magnetism.plot(supercell)
        plot.assert_called_once_with(supercell)
        viewer = plot.return_value
        viewer.show_arrows_at_atoms.assert_called_once()


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


def test_incorrect_argument(example_magnetism):
    with pytest.raises(exception.IncorrectUsage):
        example_magnetism["step not an integer"].read()
    out_of_bounds = 999
    with pytest.raises(exception.IncorrectUsage):
        example_magnetism[out_of_bounds].moments()
    with pytest.raises(exception.IncorrectUsage):
        example_magnetism[out_of_bounds].total_moments()
    with pytest.raises(exception.IncorrectUsage):
        example_magnetism[out_of_bounds].charges()
    with pytest.raises(exception.IncorrectUsage):
        example_magnetism[out_of_bounds].total_charges()


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.magnetism("collinear")
    check_factory_methods(Magnetism, data)

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import functools
import types

import numpy as np
import pytest

from py4vasp import _config, calculation, exception


@pytest.fixture(params=[slice(None), slice(1, 3), 0, -1])
def steps(request):
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
    magnetism = calculation.magnetism.from_data(raw_magnetism)
    magnetism.ref = types.SimpleNamespace()
    magnetism.ref.kind = kind
    magnetism.ref.charges = raw_magnetism.spin_moments[:, 0]
    magnetism.ref.structure = calculation.structure.from_data(raw_magnetism.structure)
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
        orbital_moments = np.moveaxis(raw_magnetism.orbital_moments, 1, 3)
        reference.moments = spin_moments + orbital_moments
        reference.spin_moments = spin_moments
        reference.orbital_moments = orbital_moments


def test_read(example_magnetism, steps, Assert):
    magnetism = example_magnetism[steps] if steps != -1 else example_magnetism
    actual = magnetism.read()
    Assert.allclose(actual["charges"], example_magnetism.ref.charges[steps])
    Assert.allclose(actual["moments"], example_magnetism.ref.moments[steps])


def test_read_spin_and_orbital_moments(orbital_moments, steps, Assert):
    magnetism = orbital_moments[steps] if steps != -1 else orbital_moments
    actual = magnetism.read()
    reference = orbital_moments.ref
    Assert.allclose(actual["spin_moments"], reference.spin_moments[steps])
    Assert.allclose(actual["orbital_moments"], reference.orbital_moments[steps])


def test_charges(example_magnetism, steps, Assert):
    magnetism = example_magnetism[steps] if steps != -1 else example_magnetism
    Assert.allclose(magnetism.charges(), example_magnetism.ref.charges[steps])


def test_moments(example_magnetism, steps, Assert):
    magnetism = example_magnetism[steps] if steps != -1 else example_magnetism
    Assert.allclose(magnetism.moments(), example_magnetism.ref.moments[steps])


def test_moments_selection(example_magnetism, Assert):
    magnetism = example_magnetism
    Assert.allclose(magnetism.moments("total"), magnetism.ref.moments[-1])
    if magnetism.ref.kind == "orbital_moments":
        Assert.allclose(magnetism.moments("spin"), magnetism.ref.spin_moments[-1])
        Assert.allclose(magnetism.moments("orbital"), magnetism.ref.orbital_moments[-1])
    else:
        Assert.allclose(magnetism.moments("spin"), magnetism.ref.moments[-1])
        with pytest.raises(exception.NoData):
            magnetism.moments("orbital")
    with pytest.raises(exception.IncorrectUsage):
        magnetism.moments("unknown_option")


def test_total_charges(example_magnetism, steps, Assert):
    magnetism = example_magnetism[steps] if steps != -1 else example_magnetism
    total_charges = np.sum(magnetism.ref.charges, axis=2)
    Assert.allclose(magnetism.total_charges(), total_charges[steps])


def test_total_moments(example_magnetism, steps, Assert):
    magnetism = example_magnetism[steps] if steps != -1 else example_magnetism
    if example_magnetism.ref.kind == "charge_only":
        assert magnetism.total_moments() is None
    else:
        total_moments = np.sum(magnetism.ref.moments, axis=2)
        Assert.allclose(magnetism.total_moments(), total_moments[steps])


@pytest.mark.parametrize("selection", ["total", "spin", "orbital"])
def test_total_moments_selection(example_magnetism, selection, Assert):
    if expect_exception(example_magnetism.ref.kind, selection):
        with pytest.raises(exception.NoData):
            example_magnetism.total_moments(selection)
    else:
        moments = example_magnetism.moments(selection)
        total_moments = np.sum(moments, axis=1) if moments is not None else None
        Assert.allclose(example_magnetism.total_moments(selection), total_moments)


@pytest.mark.parametrize("selection", [None, "total", "spin", "orbital"])
def test_plot(example_magnetism, steps, selection, Assert):
    magnetism = example_magnetism[steps] if steps != -1 else example_magnetism
    plot_selection = functools.partial(magnetism.plot, selection)
    plot_under_test = plot_selection if selection else magnetism.plot
    if expect_exception(example_magnetism.ref.kind, selection):
        with pytest.raises(exception.NoData):
            plot_under_test()
        return
    view = plot_under_test()
    structure_view = example_magnetism.ref.structure[steps].plot()
    Assert.same_structure_view(view, structure_view)
    reference_moments = expected_moments(example_magnetism.ref, steps, selection)
    check_view(view, reference_moments, selection, Assert)


def expect_exception(kind, selection):
    return kind != "orbital_moments" and selection == "orbital"


def expected_moments(reference, steps=-1, selection=None):
    if reference.kind == "charge_only":
        return None
    if reference.kind == "orbital_moments" and selection == "spin":
        moments = np.sum(reference.spin_moments[steps], axis=-2)
    elif reference.kind == "orbital_moments" and selection == "orbital":
        moments = np.sum(reference.orbital_moments[steps], axis=-2)
    elif reference.kind != "collinear":
        moments = np.sum(reference.moments[steps], axis=-2)
    else:
        list_ = [
            [[0, 0, m] for m in np.sum(reference.moments[step], axis=-1)]
            for step in np.atleast_1d(np.arange(4)[steps])
        ]
        moments = np.array(list_)
    largest_moment = np.max(np.linalg.norm(moments, axis=-1))
    rescale_moments = calculation.magnetism.length_moments / largest_moment
    return rescale_moments * moments


def check_view(view, reference_moments, selection, Assert):
    if reference_moments is None:
        assert view.ion_arrows is None
    else:
        assert len(view.ion_arrows) == 1
        arrows = view.ion_arrows[0]
        assert arrows.quantity.ndim == 3
        Assert.allclose(arrows.quantity, reference_moments)
        expected_label = f"{selection} moments" if selection else "total moments"
        assert arrows.label == expected_label
        assert arrows.color == _config.VASP_BLUE
        assert arrows.radius == 0.2


@pytest.mark.parametrize("supercell", [None, 2, (3, 2, 1)])
def test_plot_supercell(example_magnetism, supercell, Assert):
    if example_magnetism.ref.kind == "charge_only":
        return
    plot_method = example_magnetism.plot
    view = plot_method(supercell=supercell) if supercell else plot_method()
    structure_view = example_magnetism.ref.structure.plot(supercell)
    Assert.same_structure_view(view, structure_view)
    reference_moments = expected_moments(example_magnetism.ref)
    check_view(view, reference_moments, selection="total", Assert=Assert)


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
    check_factory_methods(calculation.magnetism, data)

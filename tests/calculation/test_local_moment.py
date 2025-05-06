# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import functools
import types

import numpy as np
import pytest

from py4vasp import _config, exception
from py4vasp._calculation.local_moment import LocalMoment
from py4vasp._calculation.structure import Structure


@pytest.fixture(params=[slice(None), slice(1, 3), 0, -1])
def steps(request):
    return request.param


@pytest.fixture(params=["collinear", "noncollinear", "orbital_moments", "charge_only"])
def example_moments(raw_data, request):
    return setup_moments(raw_data, kind=request.param)


@pytest.fixture(params=["collinear", "noncollinear", "orbital_moments"])
def example_magnetic(raw_data, request):
    return setup_moments(raw_data, kind=request.param)


@pytest.fixture
def charge_only(raw_data):
    return setup_moments(raw_data, "charge_only")


@pytest.fixture
def collinear_moments(raw_data):
    return setup_moments(raw_data, "collinear")


@pytest.fixture
def noncollinear_moments(raw_data):
    return setup_moments(raw_data, "noncollinear")


@pytest.fixture
def orbital_moments(raw_data):
    return setup_moments(raw_data, "orbital_moments")


def setup_moments(raw_data, kind):
    raw_moment = raw_data.local_moment(kind)
    local_moment = LocalMoment.from_data(raw_moment)
    local_moment.ref = types.SimpleNamespace()
    local_moment.ref.charge = raw_moment.spin_moments[:, 0]
    local_moment.ref.structure = Structure.from_data(raw_moment.structure)
    lmax = raw_moment.spin_moments.shape[-1]
    local_moment.ref.projections = ["s", "p", "d", "f"][:lmax]
    set_moments(raw_moment, kind, local_moment.ref)
    return local_moment


def set_moments(raw_moment, kind, reference):
    if kind == "collinear":
        reference.magnetic = raw_moment.spin_moments[:, 1]
        reference.spin_moments = reference.magnetic
        reference.components = ["charge", "total", "spin"]
    elif kind == "noncollinear":
        reference.magnetic = np.moveaxis(raw_moment.spin_moments[:, 1:4], 1, 3)
        reference.spin_moments = reference.magnetic
        reference.components = ["charge", "total", "spin"]
    elif kind == "orbital_moments":
        spin_moments = np.moveaxis(raw_moment.spin_moments[:, 1:4], 1, 3)
        orbital_moments = np.zeros_like(spin_moments).astype(np.float64)
        orbital_moments[:, :, 1:] += np.moveaxis(raw_moment.orbital_moments, 1, 3)
        reference.magnetic = spin_moments + orbital_moments
        reference.spin_moments = spin_moments
        reference.orbital_moments = orbital_moments
        reference.components = ["charge", "total", "spin", "orbital"]
    else:  # charge-only
        reference.components = ["charge"]


def test_read_charge(example_moments, steps, Assert):
    moments = example_moments[steps] if steps != -1 else example_moments
    actual = moments.read()
    assert actual["orbital_projection"] == example_moments.ref.projections
    Assert.allclose(actual["charge"], example_moments.ref.charge[steps])


def test_read_magnetic(example_magnetic, steps, Assert):
    moments = example_magnetic[steps] if steps != -1 else example_magnetic
    actual = moments.read()
    Assert.allclose(actual["total"], example_magnetic.ref.magnetic[steps])


def test_read_spin_and_orbital_moments(orbital_moments, steps, Assert):
    moments = orbital_moments[steps] if steps != -1 else orbital_moments
    actual = moments.read()
    Assert.allclose(actual["spin"], orbital_moments.ref.spin_moments[steps])
    Assert.allclose(actual["orbital"], orbital_moments.ref.orbital_moments[steps])


def test_read_charge_does_not_contain_magnetic(charge_only):
    actual = charge_only.read()
    assert "total" not in actual
    assert "spin" not in actual
    assert "orbital" not in actual


def test_read_noncollinear_does_not_contain_spin_and_orbital(noncollinear_moments):
    actual = noncollinear_moments.read()
    assert "spin" not in actual
    assert "orbital" not in actual


def test_projected_charge(example_moments, steps, Assert):
    moments = example_moments[steps] if steps != -1 else example_moments
    Assert.allclose(moments.projected_charge(), example_moments.ref.charge[steps])


def test_projected_magnetic(example_magnetic, steps, Assert):
    moments = example_magnetic[steps] if steps != -1 else example_magnetic
    Assert.allclose(moments.projected_magnetic(), example_magnetic.ref.magnetic[steps])


def test_projected_magnetic_selection(example_magnetic, Assert):
    moments = example_magnetic
    Assert.allclose(moments.projected_magnetic("total"), moments.ref.magnetic[-1])
    Assert.allclose(moments.projected_magnetic("spin"), moments.ref.spin_moments[-1])


def test_projected_magnetic_orbital_selection(orbital_moments, Assert):
    moments = orbital_moments.projected_magnetic("orbital")
    Assert.allclose(moments, orbital_moments.ref.orbital_moments[-1])


@pytest.mark.parametrize("selection", [None, "total", "spin", "orbital"])
def test_projected_magnetic_on_charge_raises_error(charge_only, selection):
    kwargs = {"selection": selection} if selection else {}
    with pytest.raises(exception.NoData):
        charge_only.projected_magnetic(**kwargs)


def test_projected_magnetic_with_incorrect_selection(example_magnetic):
    with pytest.raises(exception.IncorrectUsage):
        example_magnetic.projected_magnetic("unknown_option")


def test_projected_magnetic_without_orbital_moments(
    collinear_moments, noncollinear_moments
):
    for moments in (collinear_moments, noncollinear_moments):
        with pytest.raises(exception.NoData):
            moments.projected_magnetic("orbital")


def test_charge(example_moments, steps, Assert):
    moments = example_moments[steps] if steps != -1 else example_moments
    total_charge = np.sum(moments.ref.charge, axis=2)
    Assert.allclose(moments.charge(), total_charge[steps])


def test_magnetic(example_magnetic, steps, Assert):
    moments = example_magnetic[steps] if steps != -1 else example_magnetic
    total_magnetic = np.sum(moments.ref.magnetic, axis=2)
    Assert.allclose(moments.magnetic(), total_magnetic[steps])


@pytest.mark.parametrize("selection", ["total", "spin"])
def test_magnetic_selection(example_magnetic, selection, Assert):
    total_moments = np.sum(example_magnetic.projected_magnetic(selection), axis=1)
    Assert.allclose(example_magnetic.magnetic(selection), total_moments)


def test_magnetic_orbital_moments(orbital_moments, Assert):
    total_moments = np.sum(orbital_moments.projected_magnetic("orbital"), axis=1)
    Assert.allclose(orbital_moments.magnetic("orbital"), total_moments)


def test_magnetic_select_multiple(orbital_moments, Assert):
    spin_moments, orbital_mom = orbital_moments[:].projected_magnetic("spin, orbital")
    total_moments = orbital_moments[:].magnetic("spin, orbital")
    Assert.allclose(spin_moments, orbital_moments.ref.spin_moments)
    Assert.allclose(orbital_mom, orbital_moments.ref.orbital_moments)
    total_expected = np.sum([spin_moments, orbital_mom], axis=3)
    Assert.allclose(total_moments, total_expected)


def test_magnetic_on_charge_raises_error(charge_only, Assert):
    with pytest.raises(exception.NoData):
        charge_only.magnetic()


def test_plot_charge(charge_only, Assert):
    view = charge_only.plot()
    structure_view = charge_only.ref.structure.plot()
    Assert.same_structure_view(view, structure_view)
    assert view.ion_arrows is None


@pytest.mark.parametrize("selection", [None, "total", "spin"])
def test_plot_magnetic(example_magnetic, steps, selection, Assert):
    moments = example_magnetic[steps] if steps != -1 else example_magnetic
    kwargs = {"selection": selection} if selection else {}
    view = moments.plot(**kwargs)
    structure_view = example_magnetic.ref.structure[steps].plot()
    Assert.same_structure_view(view, structure_view)
    reference_moments = expected_moments(example_magnetic.ref, steps, selection)
    assert len(view.ion_arrows) == 1
    check_arrows(view.ion_arrows[0], reference_moments, selection, Assert)


@pytest.mark.parametrize("supercell", [None, 2, (3, 2, 1)])
def test_plot_supercell(example_magnetic, supercell, Assert):
    kwargs = {"supercell": supercell} if supercell else {}
    view = example_magnetic.plot(**kwargs)
    structure_view = example_magnetic.ref.structure.plot(supercell)
    Assert.same_structure_view(view, structure_view)
    reference_moments = expected_moments(example_magnetic.ref)
    assert len(view.ion_arrows) == 1
    check_arrows(view.ion_arrows[0], reference_moments, "total", Assert)


def test_plot_multiple(orbital_moments, Assert):
    view = orbital_moments.plot("spin, orbital")
    spin_moments = expected_moments(orbital_moments.ref, selection="spin")
    orbital_moments = expected_moments(orbital_moments.ref, selection="orbital")
    assert len(view.ion_arrows) == 2
    check_arrows(view.ion_arrows[0], spin_moments, "spin", Assert)
    check_arrows(view.ion_arrows[1], orbital_moments, "orbital", Assert)


def expected_moments(reference, steps=-1, selection=None):
    if reference.magnetic.ndim == 3:  # collinear
        # make last dimension direction dimension with vectors parallel to z
        list_ = [
            [[0, 0, m] for m in np.sum(reference.magnetic[step], axis=-1)]
            for step in np.atleast_1d(np.arange(4)[steps])
        ]
        moments = np.array(list_)
    elif selection == "spin":
        moments = np.sum(reference.spin_moments[steps], axis=-2)
    elif selection == "orbital":
        moments = np.sum(reference.orbital_moments[steps], axis=-2)
    else:  # total
        moments = np.sum(reference.magnetic[steps], axis=-2)
    largest_moment = np.max(np.linalg.norm(moments, axis=-1))
    rescale_moments = LocalMoment.length_moments / largest_moment
    return rescale_moments * moments


def check_arrows(arrows, reference_moments, selection, Assert):
    assert arrows.quantity.ndim == 3
    Assert.allclose(arrows.quantity, reference_moments)
    expected_label = f"{selection} moments" if selection else "total moments"
    assert arrows.label == expected_label
    assert arrows.color == _config.VASP_COLORS[expected_color(selection)]
    assert arrows.radius == 0.2


def expected_color(selection):
    if selection == "total" or selection is None:
        return "blue"
    if selection == "spin":
        return "purple"
    if selection == "orbital":
        return "red"
    raise NotImplemented(selection + " not implemented")


def test_selections(example_moments):
    actual = example_moments.selections()
    actual.pop("local_moment")
    assert actual == {
        "orbital_projection": example_moments.ref.projections,
        "component": example_moments.ref.components,
    }


def test_collinear_print(collinear_moments, format_):
    actual, _ = format_(collinear_moments)
    reference = "MAGMOM = 444.00 453.00 462.00 471.00 480.00 489.00 498.00"
    assert actual == {"text/plain": reference}


def test_noncollinear_print(noncollinear_moments, format_):
    actual, _ = format_(noncollinear_moments)
    reference = """\
MAGMOM = 1462.00 1574.00 1686.00 \\
         1478.00 1590.00 1702.00 \\
         1494.00 1606.00 1718.00 \\
         1510.00 1622.00 1734.00 \\
         1526.00 1638.00 1750.00 \\
         1542.00 1654.00 1766.00 \\
         1558.00 1670.00 1782.00"""
    assert actual == {"text/plain": reference}


def test_charge_only_print(charge_only, format_):
    actual, _ = format_(charge_only)
    assert actual == {"text/plain": "not spin polarized"}


def test_incorrect_step(example_moments):
    with pytest.raises(exception.IncorrectUsage):
        example_moments["step not an integer"].read()
    out_of_bounds = 999
    with pytest.raises(exception.IncorrectUsage):
        example_moments[out_of_bounds].projected_magnetic()
    with pytest.raises(exception.IncorrectUsage):
        example_moments[out_of_bounds].magnetic()
    with pytest.raises(exception.IncorrectUsage):
        example_moments[out_of_bounds].projected_charge()
    with pytest.raises(exception.IncorrectUsage):
        example_moments[out_of_bounds].charge()


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.local_moment("collinear")
    check_factory_methods(LocalMoment, data)

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
import types

import numpy as np
import pytest

from py4vasp import calculation
from py4vasp._util.slicing import plane
from py4vasp.calculation._partial_charge import STM_settings
from py4vasp.exception import IncorrectUsage, NoData, NotImplemented


@pytest.fixture(
    params=[
        "no splitting no spin",
        "no splitting no spin Ca3AsBr3",
        "no splitting no spin Sr2TiO4",
        "no splitting no spin CaAs3_110",
        "split_bands",
        "split_bands and spin_polarized",
        "split_bands and spin_polarized Ca3AsBr3",
        "split_bands and spin_polarized Sr2TiO4",
        "split_kpoints",
        "split_kpoints and spin_polarized",
        "split_kpoints and spin_polarized Ca3AsBr3",
        "split_kpoints and spin_polarized Sr2TiO4",
        "spin_polarized",
        "spin_polarized Ca3AsBr3",
        "spin_polarized Sr2TiO4",
        "split_bands and split_kpoints",
        "split_bands and split_kpoints and spin_polarized",
        "split_bands and split_kpoints and spin_polarized Ca3AsBr3",
        "split_bands and split_kpoints and spin_polarized Sr2TiO4",
    ]
)
def PartialCharge(raw_data, request):
    return make_reference_partial_charge(raw_data, request.param)


@pytest.fixture
def NonSplitPartialCharge(raw_data):
    return make_reference_partial_charge(raw_data, "no splitting no spin")


@pytest.fixture
def PolarizedNonSplitPartialCharge(raw_data):
    return make_reference_partial_charge(raw_data, "spin_polarized")


@pytest.fixture
def PolarizedNonSplitPartialChargeCa3AsBr3(raw_data):
    return make_reference_partial_charge(raw_data, "spin_polarized Ca3AsBr3")


@pytest.fixture
def NonSplitPartialChargeCaAs3_110(raw_data):
    return make_reference_partial_charge(raw_data, "CaAs3_110")


@pytest.fixture
def NonSplitPartialChargeNi_100(raw_data):
    return make_reference_partial_charge(raw_data, "Ni100")


@pytest.fixture
def PolarizedNonSplitPartialChargeSr2TiO4(raw_data):
    return make_reference_partial_charge(raw_data, "spin_polarized Sr2TiO4")


@pytest.fixture
def NonPolarizedBandSplitPartialCharge(raw_data):
    return make_reference_partial_charge(raw_data, "split_bands")


@pytest.fixture
def PolarizedAllSplitPartialCharge(raw_data):
    return make_reference_partial_charge(
        raw_data, "split_bands and split_kpoints and spin_polarized"
    )


@pytest.fixture(params=["up", "down", "total"])
def spin(request):
    return request.param


def make_reference_partial_charge(raw_data, selection):
    raw_partial_charge = raw_data.partial_charge(selection=selection)
    parchg = calculation.partial_charge.from_data(raw_partial_charge)
    parchg.ref = types.SimpleNamespace()
    parchg.ref.structure = calculation.structure.from_data(raw_partial_charge.structure)
    parchg.ref.plane_vectors = plane(
        cell=parchg.ref.structure.lattice_vectors(),
        cut="c",
        normal="z",
    )
    parchg.ref.partial_charge = raw_partial_charge.partial_charge
    parchg.ref.bands = raw_partial_charge.bands
    parchg.ref.kpoints = raw_partial_charge.kpoints
    parchg.ref.grid = raw_partial_charge.grid
    return parchg


def test_read(PartialCharge, Assert):
    actual = PartialCharge.read()
    expected = PartialCharge.ref
    Assert.allclose(actual["bands"], expected.bands)
    Assert.allclose(actual["kpoints"], expected.kpoints)
    Assert.allclose(actual["grid"], expected.grid)
    expected_charge = np.squeeze(np.asarray(expected.partial_charge).T)
    Assert.allclose(actual["partial_charge"], expected_charge)
    Assert.same_structure(actual["structure"], expected.structure.read())


def test_topology(PartialCharge):
    actual = PartialCharge._topology()
    expected = str(PartialCharge.ref.structure._topology())
    assert actual == expected


def test_bands(PartialCharge, Assert):
    actual = PartialCharge.bands()
    expected = PartialCharge.ref.bands
    Assert.allclose(actual, expected)


def test_kpoints(PartialCharge, Assert):
    actual = PartialCharge.kpoints()
    expected = PartialCharge.ref.kpoints
    Assert.allclose(actual, expected)


def test_grid(PartialCharge, Assert):
    actual = PartialCharge.grid()
    expected = PartialCharge.ref.grid
    Assert.allclose(actual, expected)


def test_non_split_to_numpy(PolarizedNonSplitPartialCharge, Assert):
    actual = PolarizedNonSplitPartialCharge.to_numpy("total")
    expected = PolarizedNonSplitPartialCharge.ref.partial_charge
    Assert.allclose(actual, expected[0, 0, 0].T)

    actual = PolarizedNonSplitPartialCharge.to_numpy("up")
    Assert.allclose(actual, 0.5 * (expected[0, 0, 0].T + expected[0, 0, 1].T))

    actual = PolarizedNonSplitPartialCharge.to_numpy("down")
    Assert.allclose(actual, 0.5 * (expected[0, 0, 0].T - expected[0, 0, 1].T))


def test_split_to_numpy(PolarizedAllSplitPartialCharge, Assert):
    bands = PolarizedAllSplitPartialCharge.ref.bands
    kpoints = PolarizedAllSplitPartialCharge.ref.kpoints
    for band_index, band in enumerate(bands):
        for kpoint_index, kpoint in enumerate(kpoints):
            actual = PolarizedAllSplitPartialCharge.to_numpy(
                band=band, kpoint=kpoint, selection="total"
            )
            expected = PolarizedAllSplitPartialCharge.ref.partial_charge
            Assert.allclose(actual, np.asarray(expected)[kpoint_index, band_index, 0].T)

    msg = f"Band {max(bands) + 1} not found in the bands array."
    with pytest.raises(NoData) as excinfo:
        PolarizedAllSplitPartialCharge.to_numpy(
            band=max(bands) + 1, kpoint=max(kpoints), selection="up"
        )
    assert msg in str(excinfo.value)

    msg = f"K-point {min(kpoints) - 1} not found in the kpoints array."
    with pytest.raises(NoData) as excinfo:
        PolarizedAllSplitPartialCharge.to_numpy(
            band=min(bands), kpoint=min(kpoints) - 1, selection="down"
        )
    assert msg in str(excinfo.value)


def test_non_polarized_to_numpy(NonSplitPartialCharge, spin, Assert):
    actual = NonSplitPartialCharge.to_numpy(selection=spin)
    expected = NonSplitPartialCharge.ref.partial_charge
    Assert.allclose(actual, np.asarray(expected).T[:, :, :, 0, 0, 0])


def test_split_bands_to_numpy(NonPolarizedBandSplitPartialCharge, spin, Assert):
    bands = NonPolarizedBandSplitPartialCharge.ref.bands
    for band_index, band in enumerate(bands):
        actual = NonPolarizedBandSplitPartialCharge.to_numpy(spin, band=band)
    expected = NonPolarizedBandSplitPartialCharge.ref.partial_charge
    Assert.allclose(actual, np.asarray(expected).T[:, :, :, 0, band_index, 0])


def test_to_stm_split(PolarizedAllSplitPartialCharge):
    msg = "set LSEPK and LSEPB to .FALSE. in the INCAR file."
    with pytest.raises(NotImplemented) as excinfo:
        PolarizedAllSplitPartialCharge.to_stm(selection="constant_current")
    assert msg in str(excinfo.value)


def test_to_stm_nonsplit_tip_to_high(NonSplitPartialCharge):
    actual = NonSplitPartialCharge
    tip_height = 8.4
    error = f"""The tip position at {tip_height:.2f} is above half of the
             estimated vacuum thickness {actual._estimate_vacuum():.2f} Angstrom.
            You would be sampling the bottom of your slab, which is not supported."""
    with pytest.raises(IncorrectUsage, match=error):
        actual.to_stm(tip_height=tip_height)


def test_to_stm_nonsplit_not_orthogonal_no_vacuum(
    PolarizedNonSplitPartialChargeSr2TiO4,
):
    msg = "The vacuum region in your cell is too small for STM simulations."
    with pytest.raises(IncorrectUsage) as excinfo:
        PolarizedNonSplitPartialChargeSr2TiO4.to_stm()
    assert msg in str(excinfo.value)


def test_to_stm_wrong_spin_nonsplit(PolarizedNonSplitPartialCharge):
    msg = "'up', 'down', or 'total'"
    with pytest.raises(IncorrectUsage) as excinfo:
        PolarizedNonSplitPartialCharge.to_stm(selection="all")
    assert msg in str(excinfo.value)


def test_to_stm_wrong_mode(PolarizedNonSplitPartialCharge):
    with pytest.raises(IncorrectUsage) as excinfo:
        PolarizedNonSplitPartialCharge.to_stm(selection="stm")
    assert "STM mode" in str(excinfo.value)


def test_wrong_vacuum_direction(NonSplitPartialChargeNi_100):
    msg = """The vacuum region in your cell is not located along
        the third lattice vector."""
    with pytest.raises(NotImplemented) as excinfo:
        NonSplitPartialChargeNi_100.to_stm()
    assert msg in str(excinfo.value)


@pytest.mark.parametrize("alias", ("constant_height", "ch", "height"))
def test_to_stm_nonsplit_constant_height(
    PolarizedNonSplitPartialCharge, alias, spin, Assert, not_core
):
    supercell = 3
    actual = PolarizedNonSplitPartialCharge.to_stm(
        selection=f"{alias}({spin})", tip_height=2.0, supercell=supercell
    )
    expected = PolarizedNonSplitPartialCharge.ref
    assert type(actual.series.data) == np.ndarray
    assert actual.series.data.shape == (expected.grid[0], expected.grid[1])
    Assert.allclose(actual.series.lattice.vectors, expected.plane_vectors.vectors)
    Assert.allclose(actual.series.supercell, np.asarray([supercell, supercell]))
    # check different elements of the label
    assert type(actual.series.label) is str
    expected = "both spin channels" if spin == "total" else f"spin {spin}"
    assert expected in actual.series.label
    assert "constant height" in actual.series.label
    assert "2.0" in actual.series.label
    assert "constant height" in actual.title
    assert "2.0" in actual.title


@pytest.mark.parametrize("alias", ("constant_current", "cc", "current"))
def test_to_stm_nonsplit_constant_current(
    PolarizedNonSplitPartialCharge, alias, spin, Assert, not_core
):
    current = 5
    supercell = np.asarray([2, 4])
    actual = PolarizedNonSplitPartialCharge.to_stm(
        selection=f"{spin}({alias})",
        current=current,
        supercell=supercell,
    )
    expected = PolarizedNonSplitPartialCharge.ref
    assert type(actual.series.data) == np.ndarray
    assert actual.series.data.shape == (expected.grid[0], expected.grid[1])
    Assert.allclose(actual.series.lattice.vectors, expected.plane_vectors.vectors)
    Assert.allclose(actual.series.supercell, supercell)
    # check different elements of the label
    assert type(actual.series.label) is str
    expected = "both spin channels" if spin == "total" else f"spin {spin}"
    assert expected in actual.series.label
    assert "constant current" in actual.series.label
    assert f"{current:.2f}" in actual.series.label
    assert "constant current" in actual.title
    assert f"{current:.2f}" in actual.title


@pytest.mark.parametrize("alias", ("constant_current", "cc", "current"))
def test_to_stm_nonsplit_constant_current_non_ortho(
    NonSplitPartialChargeCaAs3_110, alias, spin, Assert, not_core
):
    current = 5
    supercell = np.asarray([2, 4])
    actual = NonSplitPartialChargeCaAs3_110.to_stm(
        selection=f"{spin}({alias})",
        current=current,
        supercell=supercell,
    )
    expected = NonSplitPartialChargeCaAs3_110.ref
    assert type(actual.series.data) == np.ndarray
    assert actual.series.data.shape == (expected.grid[0], expected.grid[1])
    Assert.allclose(actual.series.lattice.vectors, expected.plane_vectors.vectors)
    Assert.allclose(actual.series.supercell, supercell)
    # check different elements of the label
    assert type(actual.series.label) is str
    expected = "both spin channels" if spin == "total" else f"spin {spin}"
    assert expected in actual.series.label
    assert "constant current" in actual.series.label
    assert f"{current:.2f}" in actual.series.label
    assert "constant current" in actual.title
    assert f"{current:.2f}" in actual.title


def test_stm_default_settings(PolarizedNonSplitPartialCharge):
    PolarizedNonSplitPartialCharge.to_stm()
    actual = dataclasses.asdict(PolarizedNonSplitPartialCharge.stm_settings)
    defaults = {
        "sigma_xy": 4.0,
        "sigma_z": 4.0,
        "truncate": 3.0,
        "enhancement_factor": 1000,
        "interpolation_factor": 10,
    }
    assert actual == defaults


def test_stm_updated_settings(PolarizedNonSplitPartialCharge):
    defaults = {
        "sigma_xy": 4.0,
        "sigma_z": 4.0,
        "truncate": 3.0,
        "enhancement_factor": 1000,
        "interpolation_factor": 10,
    }
    new_settings = {
        "sigma_xy": 3.0,
        "sigma_z": 5.0,
        "truncate": 2.0,
        "enhancement_factor": 2000,
        "interpolation_factor": 20,
    }
    settings = PolarizedNonSplitPartialCharge.stm_settings
    print(type(settings))
    default_settings = dataclasses.asdict(settings)
    assert default_settings == defaults
    settings = STM_settings(**new_settings)
    print(settings)
    PolarizedNonSplitPartialCharge.to_stm(stm_settings=settings)
    updated_settings = dataclasses.asdict(PolarizedNonSplitPartialCharge.stm_settings)
    assert updated_settings == new_settings


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.partial_charge("spin_polarized")
    check_factory_methods(calculation.partial_charge, data)

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import types

import numpy as np
import pytest

from py4vasp import calculation
from py4vasp.exception import IncorrectUsage, NoData, NotImplemented


@pytest.fixture(
    params=[
        "no splitting no spin",
        "no splitting no spin Ca3AsBr3",
        "no splitting no spin Sr2TiO4",
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


def make_reference_partial_charge(raw_data, selection):
    raw_partial_charge = raw_data.partial_charge(selection=selection)
    parchg = calculation.partial_charge.from_data(raw_partial_charge)
    parchg.ref = types.SimpleNamespace()
    parchg.ref.structure = calculation.structure.from_data(raw_partial_charge.structure)
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
    Assert.allclose(
        actual["partial_charge"], np.squeeze(np.asarray(expected.partial_charge).T)
    )
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


def test_non_split_to_array(PolarizedNonSplitPartialCharge, Assert):
    actual = PolarizedNonSplitPartialCharge.to_array(spin="both")
    expected = PolarizedNonSplitPartialCharge.ref.partial_charge
    Assert.allclose(actual, np.asarray(expected).T[:, :, :, 0, 0, 0])

    actual = PolarizedNonSplitPartialCharge.to_array(spin="up")
    Assert.allclose(
        actual,
        0.5
        * (
            np.asarray(expected).T[:, :, :, 0, 0, 0]
            + np.asarray(expected).T[:, :, :, 1, 0, 0]
        ),
    )

    actual = PolarizedNonSplitPartialCharge.to_array(spin="down")
    Assert.allclose(
        actual,
        0.5
        * (
            np.asarray(expected).T[:, :, :, 0, 0, 0]
            - np.asarray(expected).T[:, :, :, 1, 0, 0]
        ),
    )


def test_split_to_array(PolarizedAllSplitPartialCharge, Assert):
    bands = PolarizedAllSplitPartialCharge.ref.bands
    kpoints = PolarizedAllSplitPartialCharge.ref.kpoints
    for band_index, band in enumerate(bands):
        for kpoint_index, kpoint in enumerate(kpoints):
            actual = PolarizedAllSplitPartialCharge.to_array(
                band=band, kpoint=kpoint, spin="both"
            )
            expected = PolarizedAllSplitPartialCharge.ref.partial_charge
            Assert.allclose(
                actual, np.asarray(expected).T[:, :, :, 0, band_index, kpoint_index]
            )
    msg = f"Band {max(bands) + 1} not found in the bands array."
    with pytest.raises(NoData) as excinfo:
        PolarizedAllSplitPartialCharge.to_array(
            band=max(bands) + 1, kpoint=max(kpoints), spin="up"
        )
    assert msg in str(excinfo.value)
    msg = f"K-point {min(kpoints) - 1} not found in the kpoints array."
    with pytest.raises(NoData) as excinfo:
        PolarizedAllSplitPartialCharge.to_array(
            band=min(bands), kpoint=min(kpoints) - 1, spin="down"
        )
    assert msg in str(excinfo.value)


def test_non_polarized_to_array(NonSplitPartialCharge, Assert):
    for spin in ["both", "up", "down"]:
        actual = NonSplitPartialCharge.to_array(spin=spin)
        expected = NonSplitPartialCharge.ref.partial_charge
        Assert.allclose(actual, np.asarray(expected).T[:, :, :, 0, 0, 0])


def test_split_bands_to_array(NonPolarizedBandSplitPartialCharge, Assert):
    bands = NonPolarizedBandSplitPartialCharge.ref.bands
    for spin in ["both", "up", "down"]:
        for band_index, band in enumerate(bands):
            actual = NonPolarizedBandSplitPartialCharge.to_array(band=band, spin=spin)
        expected = NonPolarizedBandSplitPartialCharge.ref.partial_charge
        Assert.allclose(actual, np.asarray(expected).T[:, :, :, 0, band_index, 0])


def test_to_stm_split(PolarizedAllSplitPartialCharge, Assert):
    msg = "set LSEPK and LSEPB to .FALSE. in the INCAR file."
    with pytest.raises(NotImplemented) as excinfo:
        PolarizedAllSplitPartialCharge.to_stm(mode="constant_current")
    assert msg in str(excinfo.value)


def test_to_stm_nonsplit_no_vacuum(PolarizedNonSplitPartialChargeCa3AsBr3):
    actual = PolarizedNonSplitPartialChargeCa3AsBr3
    tip_height = 2.4
    error = f"""The tip position at {tip_height:.2f} is above half of the
             estimated vacuum thickness {actual._estimate_vacuum():.2f} Angstrom.
            You would be sampling the bottom of your slab, which is not supported."""
    with pytest.raises(IncorrectUsage, match=error):
        actual.to_stm(tip_height=tip_height)


def test_to_stm_nonsplit_not_orthogonal(PolarizedNonSplitPartialChargeSr2TiO4, Assert):
    msg = "STM simulations for such cells are not implemented."
    with pytest.raises(NotImplemented) as excinfo:
        PolarizedNonSplitPartialChargeSr2TiO4.to_stm()
    assert msg in str(excinfo.value)


def test_to_stm_wrong_spin_nonsplit(PolarizedNonSplitPartialCharge, Assert):
    msg = "Use 'up', 'down' or 'both'."
    with pytest.raises(IncorrectUsage) as excinfo:
        PolarizedNonSplitPartialCharge.to_stm(spin="all")
    assert msg in str(excinfo.value)


def test_to_stm_wrong_mode(PolarizedNonSplitPartialCharge, Assert):
    with pytest.raises(IncorrectUsage) as excinfo:
        PolarizedNonSplitPartialCharge.to_stm(mode="stm")
    assert "STM mode" in str(excinfo.value)


def test_to_stm_nonsplit_constant_height(PolarizedNonSplitPartialCharge, Assert):
    supercell = 3
    for spin in ["up", "down", "both"]:
        actual = PolarizedNonSplitPartialCharge.to_stm(
            spin=spin, mode="constant_height", tip_height=2.0, supercell=supercell
        )
        expected = PolarizedNonSplitPartialCharge.ref
        # assert data type and shape
        assert type(actual.series.data) == np.ndarray
        assert actual.series.data.shape == (expected.grid[0], expected.grid[1])
        # assert lattice:
        Assert.allclose(
            actual.series.lattice, expected.structure._lattice_vectors()[:2, :2]
        )
        # assert supercell
        Assert.allclose(actual.series.supercell, np.asarray([supercell, supercell]))
        # assert label and title
        assert type(actual.series.label) is str
        assert spin in actual.series.label
        assert "constant height" in actual.series.label
        assert "2.0" in actual.series.label
        assert "constant height" in actual.title
        assert "2.0" in actual.title


def test_to_stm_nonsplit_constant_current(PolarizedNonSplitPartialCharge, Assert):
    current = 5
    supercell = np.asarray([2, 4])
    for spin in ["up", "down", "both"]:
        actual = PolarizedNonSplitPartialCharge.to_stm(
            spin=spin, mode="constant_current", current=current, supercell=supercell
        )
        expected = PolarizedNonSplitPartialCharge.ref
        # assert data type and shape
        assert type(actual.series.data) == np.ndarray
        assert actual.series.data.shape == (expected.grid[0], expected.grid[1])
        # assert lattice:
        Assert.allclose(
            actual.series.lattice, expected.structure._lattice_vectors()[:2, :2]
        )
        # assert supercell
        Assert.allclose(actual.series.supercell, supercell)
        # assert label
        assert type(actual.series.label) is str
        assert spin in actual.series.label
        assert "constant current" in actual.series.label
        assert f"{current:.1e}" in actual.series.label
        assert "constant current" in actual.title
        assert f"{current:.1e}" in actual.title


def test_stm_default_settings(PolarizedNonSplitPartialCharge, Assert):
    actual = PolarizedNonSplitPartialCharge.STM_settings
    defaults = {
        "sigma_xy": 4.0,
        "sigma_z": 4.0,
        "truncate": 3.0,
        "enhancement_factor": 1000,
        "interpolation_factor": 10,
    }
    for key, value in defaults.items():
        assert getattr(actual, key) == value

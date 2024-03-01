# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import types

import numpy as np
import pytest

from py4vasp import calculation


@pytest.fixture(
    params=[
        "no splitting no spin",
        "split_bands",
        "split_bands and spin_polarized",
        "split_kpoints",
        "split_kpoints and spin_polarized",
        "spin_polarized",
        "split_bands and split_kpoints",
        "split_bands and split_kpoints and spin_polarized",
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
    error = f"Band {max(bands) + 1} not found in the bands array."
    with pytest.raises(ValueError, match=error):
        PolarizedAllSplitPartialCharge.to_array(
            band=max(bands) + 1, kpoint=max(kpoints), spin="up"
        )
    error = f"K-point {min(kpoints) - 1} not found in the kpoints array."
    with pytest.raises(ValueError):
        PolarizedAllSplitPartialCharge.to_array(
            band=min(bands), kpoint=min(kpoints) - 1, spin="down"
        )


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
    error = """Simulated STM images are only supported for non-separated bands and k-points.
            Please set LSEPK and LSEPB to .FALSE. in the INCAR file."""
    with pytest.raises(ValueError, match=error):
        PolarizedAllSplitPartialCharge.to_stm(mode="constant_current")


def test_to_stm_nonsplit_no_vacuum(PolarizedNonSplitPartialChargeCa3AsBr3, Assert):
    error = f"""The tip position at 4.0 is above half of the
             estimated vacuum thickness {PolarizedNonSplitPartialChargeCa3AsBr3._estimate_vacuum():.2f} Angstrom.
            You are probably sampling the bottom of your slab, which is not supported."""
    with pytest.raises(ValueError, match=error):
        PolarizedNonSplitPartialChargeCa3AsBr3.to_stm()


def test_to_stm_nonsplit_not_orthogonal(PolarizedNonSplitPartialChargeSr2TiO4, Assert):
    error = """The third lattice vector is not in cartesian z-direction.
            or the first two lattice vectors are not in the xy-plane.
            The STM calculation is not supported."""
    with pytest.raises(ValueError, match=error):
        PolarizedNonSplitPartialChargeSr2TiO4.to_stm()


def test_to_stm_wrong_spin_nonsplit(PolarizedNonSplitPartialCharge, Assert):
    error = "Spin 'all' not understood. Use 'up', 'down' or 'both'."
    with pytest.raises(ValueError, match=error):
        PolarizedNonSplitPartialCharge.to_stm(spin="all")


def test_to_stm_wrong_mode(PolarizedNonSplitPartialCharge, Assert):
    error = (
        "STM mode 'stm' not understood. Use 'constant_height' or 'constant_current'."
    )
    with pytest.raises(ValueError, match=error):
        PolarizedNonSplitPartialCharge.to_stm(mode="stm")


def test_to_stm_nonsplit_constant_height(PolarizedNonSplitPartialCharge, Assert):
    for spin in ["up", "down", "both"]:
        actual = PolarizedNonSplitPartialCharge.to_stm(
            spin=spin, mode="constant_height", tip_height=2.0
        )
        expected = PolarizedNonSplitPartialCharge.ref
        # assert data type and shape
        assert type(actual.data) == np.ndarray
        assert actual.data.shape == (expected.grid[0], expected.grid[1])
        # assert lattice:
        Assert.allclose(actual.lattice, expected.structure._lattice_vectors()[:2])
        # assert label
        assert type(actual.label) is str
        assert spin in actual.label
        assert "constant height" in actual.label
        assert "2.0" in actual.label


def test_to_stm_nonsplit_constant_current(PolarizedNonSplitPartialCharge, Assert):
    current = 0.5e-08
    for spin in ["up", "down", "both"]:
        actual = PolarizedNonSplitPartialCharge.to_stm(
            spin=spin, mode="constant_current", current=current
        )
        expected = PolarizedNonSplitPartialCharge.ref
        # assert data type and shape
        assert type(actual.data) == np.ndarray
        assert actual.data.shape == (expected.grid[0], expected.grid[1])
        # assert lattice:
        Assert.allclose(actual.lattice, expected.structure._lattice_vectors()[:2])
        # assert label
        assert type(actual.label) is str
        assert spin in actual.label
        assert "constant current" in actual.label
        assert f"{current:.1e}" in actual.label

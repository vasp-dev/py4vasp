# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools
import random
import re
import types

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._calculation.electron_phonon_self_energy import (
    ElectronPhononSelfEnergy,
    ElectronPhononSelfEnergyInstance,
    SparseTensor,
)
from py4vasp._util import convert


@pytest.fixture
def raw_self_energy(raw_data):
    return raw_data.electron_phonon_self_energy("default")


@pytest.fixture
def self_energy(raw_self_energy):
    self_energy = ElectronPhononSelfEnergy.from_data(raw_self_energy)
    self_energy.ref = types.SimpleNamespace()
    self_energy.ref.eigenvalues = raw_self_energy.eigenvalues
    self_energy.ref.debye_waller = raw_self_energy.debye_waller
    self_energy.ref.fan = [convert.to_complex(fan[:]) for fan in raw_self_energy.fan]
    self_energy.ref.energies = raw_self_energy.energies
    self_energy.ref.temperatures = raw_self_energy.temperatures
    self_energy.ref.nbands_sum = raw_self_energy.nbands_sum
    self_energy.ref.selfen_delta = raw_self_energy.delta
    self_energy.ref.selfen_carrier_den = _make_reference_carrier_den(raw_self_energy)
    self_energy.ref.scattering_approx = raw_self_energy.scattering_approximation
    self_energy.ref.band_kpoint_spin_index = raw_self_energy.band_kpoint_spin_index
    self_energy.ref.mapping_pattern = _make_reference_pattern()
    self_energy.ref.instance_pattern = _make_reference_pattern(raw_self_energy)
    return self_energy


@pytest.fixture(params=["carrier_den", "carrier_per_cell", "mu"])
def chemical_potential(raw_data, request):
    raw_potential = raw_data.electron_phonon_chemical_potential(request.param)
    raw_potential.ref = types.SimpleNamespace()
    raw_potential.ref.param = request.param
    raw_potential.ref.expected_data = getattr(raw_potential, request.param)
    return raw_potential


def _make_reference_carrier_den(raw_self_energy):
    chemical_potential = raw_self_energy.chemical_potential
    indices = raw_self_energy.id_index[:, 2] - 1
    return np.array([chemical_potential.carrier_den[index_] for index_ in indices])


def _make_reference_pattern(raw_self_energy=None):
    if raw_self_energy is None:
        return r"""Electron-phonon self energy with 5 instance\(s\):
    selfen_carrier_den: \[.*\]
    nbands_sum: \[.*\]
    selfen_delta: \[.*\]
    scattering_approx: \[.*\]"""
    else:
        return r"""Electron-phonon self-energy instance 1:
    selfen_carrier_den: .*
    nbands_sum: .*
    selfen_delta: .*
    scattering_approx: .*"""


def test_len(self_energy):
    # Should match the number of valid indices in the raw data
    assert len(self_energy) == len(self_energy._raw_data.valid_indices)


def test_indexing_and_iteration(self_energy):
    # Indexing and iteration should yield instances
    for i, instance in enumerate(self_energy):
        assert isinstance(instance, ElectronPhononSelfEnergyInstance)
        assert instance.index == i
        assert instance.parent is self_energy
    assert isinstance(self_energy[0], ElectronPhononSelfEnergyInstance)


def test_read_mapping(self_energy):
    # Check that to_dict returns expected keys
    assert self_energy.read() == {"naccumulators": len(self_energy)}


def test_read_instance(self_energy, Assert):
    # Each instance's to_dict should match the raw data for that index
    for i, instance in enumerate(self_energy):
        d = instance.read()
        expected_keys = {
            "eigenvalues",
            "debye_waller",
            "fan",
            "energies",
            "temperatures",
            "metadata",
        }
        assert d.keys() == expected_keys
        Assert.allclose(d["eigenvalues"], self_energy.ref.eigenvalues)
        Assert.allclose(d["debye_waller"], self_energy.ref.debye_waller[i])
        Assert.allclose(d["fan"], self_energy.ref.fan[i])
        Assert.allclose(d["energies"], self_energy.ref.energies[i])
        Assert.allclose(d["temperatures"], self_energy.ref.temperatures[i])
        assert d["metadata"] == instance.read_metadata()


def test_read_instance_metadata(self_energy):
    for i, instance in enumerate(self_energy):
        assert instance.read_metadata() == {
            "nbands_sum": self_energy.ref.nbands_sum[i],
            "selfen_delta": self_energy.ref.selfen_delta[i],
            "selfen_carrier_den": self_energy.ref.selfen_carrier_den[i],
            "scattering_approx": self_energy.ref.scattering_approx[i],
        }


def test_selections(raw_self_energy, chemical_potential, Assert):
    # Should return a dictionary with expected selection keys
    raw_self_energy.chemical_potential = chemical_potential
    self_energy = ElectronPhononSelfEnergy.from_data(raw_self_energy)
    selections = self_energy.selections()
    selections.pop("electron_phonon_self_energy")
    expected = selections.pop(f"selfen_{chemical_potential.ref.param}")
    Assert.allclose(expected, np.unique(chemical_potential.ref.expected_data))
    expected_keys = {"nbands_sum", "scattering_approx", "selfen_delta"}
    assert selections.keys() == expected_keys
    Assert.allclose(selections["nbands_sum"], np.unique(raw_self_energy.nbands_sum))
    Assert.allclose(selections["selfen_delta"], np.unique(raw_self_energy.delta))
    scattering_approximation = np.unique(raw_self_energy.scattering_approximation)
    Assert.allclose(selections["scattering_approx"], scattering_approximation)


@pytest.mark.parametrize(
    "attribute",
    ["nbands_sum", "selfen_delta", "selfen_carrier_den", "scattering_approx"],
)
def test_select_returns_instances(self_energy, attribute):
    choices = getattr(self_energy.ref, attribute)
    choice = random.choice(list(choices))
    indices, *_ = np.where(choices == choice)
    selected = self_energy.select(f"{attribute}={choice.item()}")
    assert len(selected) == len(indices)
    for index_, instance in zip(indices, selected):
        assert isinstance(instance, ElectronPhononSelfEnergyInstance)
        assert instance.index == index_


def test_select_multiple(self_energy):
    index_nbands_sum = 1
    index_selfen_delta = 3
    indices = [index_nbands_sum, index_selfen_delta]
    choice_nbands_sum = self_energy.ref.nbands_sum[index_nbands_sum]
    choice_selfen_delta = self_energy.ref.selfen_delta[index_selfen_delta]
    selection = f"nbands_sum={choice_nbands_sum.item()}, selfen_delta={choice_selfen_delta.item()}"
    selected = self_energy.select(selection)
    assert len(selected) == len(indices)
    for index_, instance in zip(indices, selected):
        assert isinstance(instance, ElectronPhononSelfEnergyInstance)
        assert instance.index == index_


def test_select_nested(self_energy):
    index_ = 0
    choice_nbands_sum = self_energy.ref.nbands_sum[index_]
    choice_selfen_carrier_den = self_energy.ref.selfen_carrier_den[index_]
    count_ = sum(self_energy.ref.selfen_carrier_den == choice_selfen_carrier_den)
    assert count_ > 1
    selection = f"nbands_sum={choice_nbands_sum.item()}(selfen_carrier_den={choice_selfen_carrier_den.item()})"
    selected = self_energy.select(selection)
    assert len(selected) == 1
    instance = selected[0]
    assert isinstance(instance, ElectronPhononSelfEnergyInstance)
    assert instance.index == index_


@pytest.mark.parametrize(
    "selection",
    ["invalid_selection=0.01", "nbands_sum:0.01", "selfen_delta"],
)
def test_incorrect_selection(self_energy, selection):
    with pytest.raises(exception.IncorrectUsage):
        self_energy.select(selection)


@pytest.fixture
def mock_sparse_tensor():
    # 4 bands, 3 kpoints, 2 spins, only some indices valid
    band_kpoint_spin_index = np.full((4, 3, 2), -2, dtype=int)
    band_kpoint_spin_index[0, 0, 0] = 0
    band_kpoint_spin_index[1, 1, 1] = 1
    band_kpoint_spin_index[3, 2, 1] = 2
    band_start = 2
    tensor = np.array([10.0, 20.0, 30.0])
    return SparseTensor(band_kpoint_spin_index, band_start, tensor)


@pytest.mark.parametrize(
    "valid_indices, expected_data",
    [[(2, 0, 0), 10.0], [(3, 1, 1), 20.0], [(5, 2, 1), 30.0]],
)
def test_sparse_tensor_valid_and_invalid_access(
    mock_sparse_tensor, valid_indices, expected_data
):
    # note the different order between Python and Fortan indices
    band, kpoint, spin = valid_indices
    assert mock_sparse_tensor[spin, kpoint, band] == expected_data


@pytest.mark.parametrize(
    "negative_indices, expected_data",
    [[(-1, 2, 1), 30.0], [(2, -3, 0), 10.0], [(-3, 1, -1), 20.0]],
)
def test_sparse_tensor_negative_indices(
    mock_sparse_tensor, negative_indices, expected_data
):
    band, kpoint, spin = negative_indices
    assert mock_sparse_tensor[spin, kpoint, band] == expected_data


def test_sparse_tensor_valid_bands(mock_sparse_tensor):
    assert mock_sparse_tensor.valid_bands == range(2, 6)


@pytest.mark.parametrize("invalid_indices", [[2, 1, 0], [3, 0, 1], [4, 2, 1]])
def test_sparse_tensor_invalid_access(mock_sparse_tensor, invalid_indices):
    band, kpoint, spin = invalid_indices
    with pytest.raises(exception.DataMismatch):
        mock_sparse_tensor[spin, kpoint, band]


@pytest.mark.parametrize("out_of_bounds_indices", [[1, 2, 1], [3, 3, 2]])
def test_sparse_tensor_out_of_bounds_access(mock_sparse_tensor, out_of_bounds_indices):
    band, kpoint, spin = out_of_bounds_indices
    with pytest.raises(exception.IncorrectUsage):
        mock_sparse_tensor[spin, kpoint, band]


def test_sparse_tensor_incorrect_arguments(mock_sparse_tensor):
    # Test incorrect number of arguments
    with pytest.raises(exception.IncorrectUsage):
        mock_sparse_tensor[1, 2]  # Missing spin index
    with pytest.raises(exception.IncorrectUsage):
        mock_sparse_tensor[1, 2, 3, 4]  # Too many arguments


def test_sparse_tensor_should_only_succeed_if_index_is_valid(mock_sparse_tensor):
    bks = mock_sparse_tensor._band_kpoint_spin_index
    valid_bands = mock_sparse_tensor.valid_bands
    for band, kpoint, spin in itertools.product(valid_bands, range(3), range(2)):
        try:
            val = mock_sparse_tensor[spin, kpoint, band]
            assert bks[band - valid_bands.start, kpoint, spin] >= 0
            assert val in (10.0, 20.0, 30.0)
        except exception.DataMismatch:
            assert bks[band - valid_bands.start, kpoint, spin] < 0


@pytest.mark.parametrize(
    "contribution", ["fan", "debye_waller", "self_energy", "energies"]
)
def test_sparse_tensor_self_energy(self_energy, contribution, Assert):
    first_band = 1
    for instance, indices in zip(self_energy, self_energy.ref.band_kpoint_spin_index):
        data = instance.read()
        if contribution == "self_energy":
            expected_result = data["fan"] + data["debye_waller"]
        else:
            expected_result = data[contribution]
        sparse_tensor = getattr(instance, contribution)()
        assert isinstance(sparse_tensor, SparseTensor)
        shape = indices.shape
        for spin, kpoint, band in itertools.product(
            range(shape[0]), range(shape[1]), range(shape[2])
        ):
            index_ = indices[spin, kpoint, band]
            if index_ >= 0:
                value = sparse_tensor[spin, kpoint, band + first_band]
                Assert.allclose(value, expected_result[index_ - 1])
            else:
                with pytest.raises(exception.DataMismatch):
                    sparse_tensor[spin, kpoint, band + first_band]


def test_eigenvalues(self_energy, Assert):
    Assert.allclose(self_energy.eigenvalues(), self_energy.ref.eigenvalues)


def test_print_mapping(self_energy, format_):
    actual, _ = format_(self_energy)
    assert re.search(self_energy.ref.mapping_pattern, str(self_energy), re.MULTILINE)
    assert actual == {"text/plain": str(self_energy)}


def test_print_instance(self_energy, format_):
    instance = self_energy[0]
    actual, _ = format_(instance)
    # Check if the actual output matches the expected pattern
    assert re.search(self_energy.ref.instance_pattern, str(instance), re.MULTILINE)
    assert actual == {"text/plain": str(instance)}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.electron_phonon_self_energy("default")
    parameters = {"select": {"selection": "selfen_approx=SERTA"}}
    skip_methods = ["count", "access", "index"]  # inherited from Sequence
    check_factory_methods(ElectronPhononSelfEnergy, data, parameters, skip_methods)

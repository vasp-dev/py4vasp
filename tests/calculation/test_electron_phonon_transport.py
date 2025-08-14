# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import random
import types

import numpy as np
import pytest

from py4vasp import calculation, exception
from py4vasp._calculation.electron_phonon_transport import (
    ElectronPhononTransport,
    ElectronPhononTransportInstance,
)


@pytest.fixture
def raw_transport(raw_data):
    return raw_data.electron_phonon_transport("default")


@pytest.fixture
def transport(raw_transport):
    transport = ElectronPhononTransport.from_data(raw_transport)
    transport.ref = types.SimpleNamespace()
    transport.ref.temperatures = raw_transport.temperatures
    transport.ref.transport_function = raw_transport.transport_function
    transport.ref.electronic_conductivity = raw_transport.electronic_conductivity
    transport.ref.mobility = raw_transport.mobility
    transport.ref.seebeck = raw_transport.seebeck
    transport.ref.peltier = raw_transport.peltier
    transport.ref.electronic_thermal_conductivity = (
        raw_transport.electronic_thermal_conductivity
    )
    transport.ref.nbands_sum = raw_transport.nbands_sum
    transport.ref.selfen_delta = raw_transport.delta
    transport.ref.selfen_carrier_den = _make_reference_carrier_den(raw_transport)
    transport.ref.scattering_approx = raw_transport.scattering_approximation
    return transport


@pytest.fixture(params=["carrier_den", "carrier_per_cell", "mu"])
def chemical_potential(raw_data, request):
    raw_potential = raw_data.electron_phonon_chemical_potential(request.param)
    raw_potential.ref = types.SimpleNamespace()
    raw_potential.ref.param = request.param
    raw_potential.ref.expected_data = getattr(raw_potential, request.param)
    return raw_potential


def _make_reference_carrier_den(raw_transport):
    chemical_potential = raw_transport.chemical_potential
    indices = raw_transport.id_index[:, 2] - 1
    return np.array([chemical_potential.carrier_den[index_] for index_ in indices])


def test_len(transport):
    # Should match the number of valid indices in the raw data
    assert len(transport) == len(transport._raw_data.valid_indices)


def test_indexing_and_iteration(transport):
    # Indexing and iteration should yield instances
    for i, instance in enumerate(transport):
        assert isinstance(instance, ElectronPhononTransportInstance)
        assert instance.index == i
        assert instance.parent is transport
    assert isinstance(transport[0], ElectronPhononTransportInstance)


def test_read_mapping(transport):
    # Check that to_dict returns expected keys
    assert transport.read() == {"naccumulators": len(transport)}


def test_read_instance(transport, Assert):
    # Each instance's to_dict should match the raw data for that index
    for i, instance in enumerate(transport):
        d = instance.to_dict()
        expected_keys = {
            "temperatures",
            "transport_function",
            "electronic_conductivity",
            "mobility",
            "seebeck",
            "peltier",
            "electronic_thermal_conductivity",
            "metadata",
        }
        assert d.keys() == expected_keys
        Assert.allclose(d["temperatures"], transport.ref.temperatures[i])
        Assert.allclose(d["transport_function"], transport.ref.transport_function[i])
        Assert.allclose(
            d["electronic_conductivity"], transport.ref.electronic_conductivity[i]
        )
        Assert.allclose(d["mobility"], transport.ref.mobility[i])
        Assert.allclose(d["seebeck"], transport.ref.seebeck[i])
        Assert.allclose(d["peltier"], transport.ref.peltier[i])
        Assert.allclose(
            d["electronic_thermal_conductivity"],
            transport.ref.electronic_thermal_conductivity[i],
        )
        assert d["metadata"] == {
            "nbands_sum": transport.ref.nbands_sum[i],
            "selfen_delta": transport.ref.selfen_delta[i],
            "selfen_carrier_den": transport.ref.selfen_carrier_den[i],
            "scattering_approx": transport.ref.scattering_approx[i],
        }


def test_selections(raw_transport, chemical_potential, Assert):
    # Should return a dictionary with expected selection keys
    raw_transport.chemical_potential = chemical_potential
    transport = ElectronPhononTransport.from_data(raw_transport)
    selections = transport.selections()
    selections.pop("electron_phonon_transport")
    expected = selections.pop(f"selfen_{chemical_potential.ref.param}")
    Assert.allclose(expected, np.unique(chemical_potential.ref.expected_data))
    expected_keys = {"nbands_sum", "scattering_approx", "selfen_delta"}
    assert selections.keys() == expected_keys
    Assert.allclose(selections["nbands_sum"], np.unique(raw_transport.nbands_sum))
    Assert.allclose(selections["selfen_delta"], np.unique(raw_transport.delta))
    scattering_approximation = np.unique(raw_transport.scattering_approximation)
    Assert.allclose(selections["scattering_approx"], scattering_approximation)


@pytest.mark.parametrize(
    "attribute",
    ["nbands_sum", "selfen_delta", "selfen_carrier_den", "scattering_approx"],
)
def test_select_returns_instances(transport, attribute):
    choices = getattr(transport.ref, attribute)
    choice = random.choice(list(choices))
    indices, *_ = np.where(choices == choice)
    selected = transport.select(f"{attribute}={choice.item()}")
    assert len(selected) == len(indices)
    for index_, instance in zip(indices, selected):
        assert isinstance(instance, ElectronPhononTransportInstance)
        assert instance.index == index_


def test_select_multiple(transport):
    index_nbands_sum = 1
    index_selfen_delta = 3
    indices = [index_nbands_sum, index_selfen_delta]
    choice_nbands_sum = transport.ref.nbands_sum[index_nbands_sum]
    choice_selfen_delta = transport.ref.selfen_delta[index_selfen_delta]
    selection = f"nbands_sum={choice_nbands_sum.item()}, selfen_delta={choice_selfen_delta.item()}"
    selected = transport.select(selection)
    assert len(selected) == len(indices)
    for index_, instance in zip(indices, selected):
        assert isinstance(instance, ElectronPhononTransportInstance)
        assert instance.index == index_


def test_select_nested(transport):
    index_ = 0
    choice_nbands_sum = transport.ref.nbands_sum[index_]
    choice_selfen_carrier_den = transport.ref.selfen_carrier_den[index_]
    count_ = sum(transport.ref.selfen_carrier_den == choice_selfen_carrier_den)
    assert count_ > 1
    selection = f"nbands_sum={choice_nbands_sum.item()}(selfen_carrier_den={choice_selfen_carrier_den.item()})"
    selected = transport.select(selection)
    assert len(selected) == 1
    instance = selected[0]
    assert isinstance(instance, ElectronPhononTransportInstance)
    assert instance.index == index_


@pytest.mark.parametrize(
    "selection",
    ["invalid_selection=0.01", "nbands_sum:0.01", "selfen_delta"],
)
def test_incorrect_selection(transport, selection):
    with pytest.raises(exception.IncorrectUsage):
        transport.select(selection)


@pytest.mark.skip
def test_print(transport, format_):
    actual, _ = format_(transport)
    assert actual["text/plain"] == "electron phonon transport"


@pytest.mark.skip
def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.electron_phonon_transport("default")
    # parameters = {"get_fan": {"arg": (0, 0, 0)}, "select": {"selection": "1 1"}}
    parameters = {
        "read_data": {"name": "mobility", "index": 0},
        "select": {"selection": "selfen_approx(MRTA) selfen_carrier_den(0.01,0.001)"},
        "to_graph_carrier": {"selection": "seebeck(xx)", "temperature": 300},
    }
    check_factory_methods(calculation.electron_phonon.transport, data, parameters)
    check_factory_methods(calculation.electron_phonon.transport, data, parameters)

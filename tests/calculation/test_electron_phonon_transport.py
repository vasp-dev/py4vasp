# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import numpy as np
import pytest

from py4vasp import calculation
from py4vasp._calculation.electron_phonon_transport import (
    ElectronPhononTransport,
    ElectronPhononTransportInstance,
)


@pytest.fixture
def transport(raw_data):
    raw_transport = raw_data.electron_phonon_transport("default")
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


@pytest.mark.skip
def test_selections(transport):
    # Should return a dictionary with expected selection keys
    selections = transport.selections()
    assert isinstance(selections, dict)
    assert "nbands_sum" in selections
    assert "selfen_approx" in selections
    assert "selfen_delta" in selections
    # At least one chemical potential tag should be present
    assert any(
        tag in selections
        for tag in ["selfen_carrier_den", "selfen_carrier_cell", "selfen_mu"]
    )


@pytest.mark.skip
def test_select_returns_instances(transport):
    from py4vasp._calculation.electron_phonon_transport import (
        ElectronPhononTransportInstance,
    )

    # Should return a list of TransportInstance (or the correct instance class)
    selections = transport.selections()
    # Try to import the correct instance class
    TransportInstance = type(transport[0])
    for nbands_sum in selections["nbands_sum"]:
        for selfen_approx in selections["selfen_approx"]:
            selected = transport.select(
                f"nbands_sum({nbands_sum}) selfen_approx({selfen_approx})"
            )
            assert len(selected) == 3
            assert all(isinstance(x, ElectronPhononTransportInstance) for x in selected)


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

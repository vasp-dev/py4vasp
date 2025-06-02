# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp import calculation


@pytest.fixture
def transport(raw_data):
    raw_transport = raw_data.electron_phonon_transport("default")
    transport = calculation.electron_phonon.transport.from_data(raw_transport)
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
    transport.ref.scattering_approximation = raw_transport.scattering_approximation
    return transport


def test_len(transport):
    # Should match the number of valid indices in the raw data
    assert len(transport) == len(transport._raw_data.valid_indices)

def test_to_dict_keys(transport):
    # Check that to_dict returns expected keys
    d = transport.to_dict()
    assert "naccumulators" in d
    assert d["naccumulators"] == len(transport)

def test_selections(transport):
    # Should return a dictionary with expected selection keys
    selections = transport.selections()
    assert isinstance(selections, dict)
    assert "nbands_sum" in selections
    assert "selfen_approx" in selections
    assert "selfen_delta" in selections
    # At least one chemical potential tag should be present
    assert any(tag in selections for tag in ["selfen_carrier_den", "selfen_carrier_cell", "selfen_mu"])

def test_select_returns_instances(transport):
    from py4vasp._calculation.electron_phonon_transport import ElectronPhononTransportInstance
    # Should return a list of TransportInstance (or the correct instance class)
    selections = transport.selections()
    # Try to import the correct instance class
    TransportInstance = type(transport[0])
    for nbands_sum in selections["nbands_sum"]:
        for selfen_approx in selections["selfen_approx"]:
            selected = transport.select(f"nbands_sum({nbands_sum}) selfen_approx({selfen_approx})")
            assert len(selected) == 3
            assert all(isinstance(x, ElectronPhononTransportInstance) for x in selected)

def test_indexing_and_iteration(transport):
    from py4vasp._calculation.electron_phonon_transport import ElectronPhononTransportInstance
    # Indexing and iteration should yield instances
    for i, instance in enumerate(transport):
        assert isinstance(instance, ElectronPhononTransportInstance)
        assert instance.index == i
        assert instance.parent is transport
    assert isinstance(transport[0], ElectronPhononTransportInstance)

def test_to_dict_instance_matches_raw(transport):
    # Each instance's to_dict should match the raw data for that index
    for i in range(len(transport)):
        d = transport[i].to_dict()
        assert "mobility" in d
        assert "transport_function" in d
        assert "electronic_conductivity" in d
        assert "temperatures" in d
        # Check shape matches
        assert d["mobility"].shape == transport.ref.mobility[i].shape
        assert d["transport_function"].shape == transport.ref.transport_function[i].shape
        assert d["electronic_conductivity"].shape == transport.ref.electronic_conductivity[i].shape

def test_read(transport, Assert):
    slice_ = 0
    actual = transport[slice_].to_dict()
    Assert.allclose(actual["mobility"], transport.ref.mobility[slice_])
    Assert.allclose(
        actual["transport_function"], transport.ref.transport_function[slice_]
    )
    Assert.allclose(
        actual["electronic_conductivity"], transport.ref.electronic_conductivity[slice_]
    )


def test_print(transport, format_):
    actual, _ = format_(transport)
    assert actual["text/plain"] == "electron phonon transport"


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.electron_phonon_transport("default")
    # parameters = {"get_fan": {"arg": (0, 0, 0)}, "select": {"selection": "1 1"}}
    parameters = {
        "read_data": {"name": "mobility", "index": 0},
        "select": {"selection": "selfen_approx(MRTA) selfen_carrier_den(0.01,0.001)"},
    }
    check_factory_methods(calculation.electron_phonon.transport, data, parameters)

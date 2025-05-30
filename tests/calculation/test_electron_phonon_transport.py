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

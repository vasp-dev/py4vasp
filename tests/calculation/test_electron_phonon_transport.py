# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import random
import re
import types

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._calculation.electron_phonon_transport import (
    DIRECTIONS,
    ElectronPhononTransport,
    TransportInstance,
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
    transport.ref.mapping_pattern = _make_reference_pattern()
    transport.ref.instance_pattern = _make_reference_pattern(raw_transport)
    return transport


@pytest.fixture
def transport_CRTA(raw_data):
    # CRTA data does not have nbands_sum and delta data
    raw_transport = raw_data.electron_phonon_transport("CRTA")
    transport = ElectronPhononTransport.from_data(raw_transport)
    transport.ref = types.SimpleNamespace()
    transport.ref.selfen_carrier_den = _make_reference_carrier_den(raw_transport)
    return transport


@pytest.fixture(params=["carrier_den", "carrier_per_cell", "mu"])
def chemical_potential(raw_data, request):
    raw_potential = raw_data.electron_phonon_chemical_potential(request.param)
    raw_potential.ref = types.SimpleNamespace()
    raw_potential.ref.param = request.param
    raw_potential.ref.expected_data = getattr(raw_potential, request.param)
    if request.param == "carrier_den":
        raw_potential.ref.xlabel = "Carrier density (cm^-3)"
    elif request.param == "carrier_per_cell":
        raw_potential.ref.xlabel = "Carrier per cell"
    elif request.param == "mu":
        raw_potential.ref.xlabel = "Chemical potential (eV)"
    return raw_potential


def _make_reference_carrier_den(raw_transport):
    chemical_potential = raw_transport.chemical_potential
    indices = raw_transport.id_index[:, 2] - 1
    return np.array([chemical_potential.carrier_den[index_] for index_ in indices])


def _make_reference_pattern(raw_transport=None):
    if raw_transport is None:
        return r"""Electron-phonon transport with 5 instance\(s\):
    selfen_carrier_den: \[.*\]
    nbands_sum: \[.*\]
    selfen_delta: \[.*\]
    scattering_approx: \[.*\]"""
    else:
        return r"""Electron-phonon transport instance 1:
    selfen_carrier_den: .*
    nbands_sum: .*
    selfen_delta: .*
    scattering_approx: .*"""


def test_len(transport):
    # Should match the number of valid indices in the raw data
    assert len(transport) == len(transport._raw_data.valid_indices)


def test_indexing_and_iteration(transport):
    # Indexing and iteration should yield instances
    for i, instance in enumerate(transport):
        assert isinstance(instance, TransportInstance)
        assert instance.index == i
        assert instance.parent is transport
    assert isinstance(transport[0], TransportInstance)


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
        assert d["metadata"] == instance.read_metadata()


def test_read_instance_metadata(transport):
    for i, instance in enumerate(transport):
        assert instance.read_metadata() == {
            "nbands_sum": transport.ref.nbands_sum[i],
            "selfen_delta": transport.ref.selfen_delta[i],
            "selfen_carrier_den": transport.ref.selfen_carrier_den[i],
            "scattering_approx": transport.ref.scattering_approx[i],
        }


def test_read_instance_metadata_CRTA(transport_CRTA):
    for i, instance in enumerate(transport_CRTA):
        assert instance.read_metadata() == {
            "scattering_approx": "CRTA",
            "selfen_carrier_den": transport_CRTA.ref.selfen_carrier_den[i],
        }


def test_selections(raw_transport, chemical_potential, Assert):
    # Should return a dictionary with expected selection keys
    raw_transport.chemical_potential = chemical_potential
    transport = ElectronPhononTransport.from_data(raw_transport)
    selections = transport.selections()
    selections.pop("electron_phonon_transport")
    transport_quantities = selections.pop("transport")
    assert set(transport_quantities) == transport.units.keys()
    expected = selections.pop(f"selfen_{chemical_potential.ref.param}")
    Assert.allclose(expected, np.unique(chemical_potential.ref.expected_data))
    expected_keys = {"nbands_sum", "scattering_approx", "selfen_delta"}
    assert selections.keys() == expected_keys
    Assert.allclose(selections["nbands_sum"], np.unique(raw_transport.nbands_sum))
    Assert.allclose(selections["selfen_delta"], np.unique(raw_transport.delta))
    scattering_approximation = np.unique(raw_transport.scattering_approximation)
    Assert.allclose(selections["scattering_approx"], scattering_approximation)


def test_selections_CRTA(transport_CRTA):
    selections = transport_CRTA.selections()
    assert "selfen_carrier_den" in selections
    assert "nbands_sum" not in selections
    assert "selfen_delta" not in selections
    assert "scattering_approx" in selections


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
        assert isinstance(instance, TransportInstance)
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
        assert isinstance(instance, TransportInstance)
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
    assert isinstance(instance, TransportInstance)
    assert instance.index == index_


@pytest.mark.filterwarnings("ignore:nbands_sum")
def test_select_missing(transport, transport_CRTA):
    assert transport.select("nbands_sum=246161") == []
    assert transport_CRTA.select("nbands_sum=20") == []


@pytest.mark.parametrize(
    "selection",
    ["invalid_selection=0.01", "nbands_sum:0.01", "selfen_delta"],
)
def test_incorrect_selection(transport, selection):
    with pytest.raises(exception.IncorrectUsage):
        transport.select(selection)


@pytest.mark.parametrize(
    "selection",
    (
        "electronic_conductivity",
        "mobility",
        "seebeck",
        "peltier",
        "electronic_thermal_conductivity",
    ),
)
def test_plot_mapping(transport, selection, Assert):
    graph = transport.plot(selection)
    assert graph.xlabel == "Carrier density (cm^-3)"
    quantity = selection.replace("_", " ").capitalize()
    assert graph.ylabel == f"{quantity} ({transport.units[selection]})"
    temperatures = transport.ref.temperatures[0]
    assert len(graph) == len(temperatures)
    data = np.trace(getattr(transport.ref, selection), axis1=2, axis2=3) / 3
    for temperature, series, expected_y in zip(temperatures, graph, data.T):
        Assert.allclose(series.x, transport.ref.selfen_carrier_den)
        Assert.allclose(series.y, expected_y)
        assert series.label == f"T={temperature}K"
        Assert.allclose(series.annotations["nbands_sum"], transport.ref.nbands_sum)
        Assert.allclose(series.annotations["selfen_delta"], transport.ref.selfen_delta)
        Assert.allclose(
            series.annotations["scattering_approx"], transport.ref.scattering_approx
        )
        assert series.marker == "*"


@pytest.mark.parametrize("direction, expected", DIRECTIONS.items())
def test_plot_mapping_with_direction(transport, direction, expected, Assert):
    raw_data = transport.ref.mobility
    new_shape = *raw_data.shape[:2], 9
    flattened_data = np.reshape(raw_data, new_shape)
    expected_data = np.average(flattened_data[:, :, np.atleast_1d(expected)], axis=-1).T
    selection = "mobility" if direction is None else f"mobility({direction})"
    graph = transport.plot(selection)
    temperatures = transport.ref.temperatures[0]
    assert len(graph) == len(temperatures)
    for temperature, series, expected_y in zip(temperatures, graph, expected_data):
        Assert.allclose(series.x, transport.ref.selfen_carrier_den)
        Assert.allclose(series.y, expected_y)
        if direction in (None, "isotropic"):
            assert series.label == f"T={temperature}K"
        else:
            assert series.label == f"{direction}, T={temperature}K"


def test_plot_mapping_multiple_directions(transport, Assert):
    graph = transport.plot("mobility(xx, yy)")
    temperatures = transport.ref.temperatures[0]
    assert len(graph) == 2 * len(temperatures)
    for ii, direction in enumerate(("xx", "yy")):
        expected_data = transport.ref.mobility[:, :, ii, ii]
        for jj, temperature in enumerate(temperatures):
            series = graph[ii * len(temperatures) + jj]
            Assert.allclose(series.x, transport.ref.selfen_carrier_den)
            Assert.allclose(series.y, expected_data[:, jj])
            assert series.label == f"{direction}, T={temperature}K"


def test_plot_mapping_select_scattering_approx(transport, Assert):
    graph = transport.plot("peltier(scattering_approx=ERTA_LAMDBA)")
    temperatures = transport.ref.temperatures[0]
    assert len(graph) == len(temperatures)
    mask = transport.ref.scattering_approx == "ERTA_LAMDBA"
    data = np.trace(np.squeeze(transport.ref.peltier[mask]), axis1=1, axis2=2) / 3
    for temperature, series, expected_y in zip(temperatures, graph, data):
        Assert.allclose(series.x, transport.ref.selfen_carrier_den[mask])
        Assert.allclose(series.y, expected_y)
        assert series.label == f"T={temperature}K"
        Assert.allclose(series.annotations["scattering_approx"], ["ERTA_LAMDBA"])
        assert series.marker == None


@pytest.mark.parametrize(
    "selection", ("T=300, T=400", "temperature=300, temperature=400")
)
def test_plot_mapping_select_temperature(transport, selection, Assert):
    graph = transport.plot(f"seebeck({selection})")
    assert len(graph) == 2
    temperatures = transport.ref.temperatures[0]
    mask = np.isclose(temperatures, 300.0) | np.isclose(temperatures, 400.0)
    data = np.trace(np.squeeze(transport.ref.seebeck[:, mask]), axis1=2, axis2=3) / 3
    for series, temperature, expected_y in zip(graph, (300.0, 400.0), data.T):
        Assert.allclose(series.x, transport.ref.selfen_carrier_den)
        Assert.allclose(series.y, expected_y)
        assert series.label == f"T={temperature}K"


@pytest.mark.parametrize(
    "selection",
    (
        # order should not matter
        "mobility(zz(nbands_sum=32(T=200)))",
        "T=200(mobility(zz(nbands_sum=32)))",
        "zz(T=200(mobility(nbands_sum=32)))",
        "nbands_sum=32(zz(T=200(mobility)))",
    ),
)
def test_plot_mapping_complex_selection(transport, selection, Assert):
    graph = transport.plot(selection)
    assert len(graph) == 1
    series = graph[0]
    index_nbands_sum = np.searchsorted(transport.ref.nbands_sum, 32)
    index_temperature = np.searchsorted(transport.ref.temperatures[0], 200.0)
    x = transport.ref.selfen_carrier_den[index_nbands_sum]
    y = transport.ref.mobility[index_nbands_sum, index_temperature, 2, 2]
    Assert.allclose(series.x, x)
    Assert.allclose(series.y, y)
    assert series.label == "zz, T=200.0K"
    assert series.annotations["nbands_sum"] == 32


def test_plot_mapping_chemical_potential(raw_transport, chemical_potential, Assert):
    # Should return a dictionary with expected selection keys
    raw_transport.chemical_potential = chemical_potential
    transport = ElectronPhononTransport.from_data(raw_transport)
    graph = transport.plot("mobility")
    assert graph.xlabel == chemical_potential.ref.xlabel


def test_plot_mapping_CRTA(transport_CRTA):
    graph = transport_CRTA.plot("mobility")
    for series in graph:
        assert "nbands_sum" not in series.annotations
        assert "selfen_delta" not in series.annotations
        assert all(series.annotations["scattering_approx"] == "CRTA")


@pytest.mark.parametrize(
    "incorrect_selection",
    ("unknown_selection", "mobility(seebeck)", "seebeck, peltier", "mobility(xx(yy))"),
)
def test_plot_mapping_incorrect_selection(transport, incorrect_selection, Assert):
    with pytest.raises(exception.IncorrectUsage):
        transport.plot(incorrect_selection)


@pytest.mark.parametrize(
    "selection",
    (
        "electronic_conductivity",
        "mobility",
        "seebeck",
        "peltier",
        "electronic_thermal_conductivity",
    ),
)
def test_plot_instance(transport, selection, Assert):
    for index_, instance in enumerate(transport):
        graph = instance.plot(selection)
        assert graph.xlabel == "Temperature (K)"
        quantity = selection.replace("_", " ").capitalize()
        assert graph.ylabel == f"{quantity} ({transport.units[selection]})"
        assert len(graph) == 1
        series = graph[0]
        Assert.allclose(series.x, transport.ref.temperatures[index_])
        expected = (
            np.trace(getattr(transport.ref, selection)[index_], axis1=1, axis2=2) / 3
        )
        Assert.allclose(series.y, expected)
        assert series.label == "isotropic"


@pytest.mark.parametrize("direction, expected", DIRECTIONS.items())
def test_plot_instance_direction(transport, direction, expected, Assert):
    index_ = 3
    instance = transport[index_]
    graph = instance.plot("mobility" if direction is None else f"mobility({direction})")
    assert len(graph) == 1
    series = graph[0]
    flattened_data = transport.ref.mobility[index_].reshape(-1, 9)
    expected_data = np.average(flattened_data[:, np.atleast_1d(expected)], axis=-1).T
    Assert.allclose(series.y, expected_data)
    if direction in (None, "isotropic"):
        assert series.label == "isotropic"
    else:
        assert series.label == direction


def test_plot_instance_multiple_directions(transport, Assert):
    index_ = 2
    instance = transport[index_]
    graph = instance.plot("seebeck(xx, yy, zz)")
    assert len(graph) == 3
    for ii, (series, direction) in enumerate(zip(graph, ("xx", "yy", "zz"))):
        expected_data = transport.ref.seebeck[index_, :, ii, ii]
        Assert.allclose(series.y, expected_data)
        assert series.label == direction


@pytest.mark.parametrize(
    "incorrect_selection",
    ("unknown_selection", "seebeck(peltier)", "seebeck, peltier", "seebeck(xx(yy))"),
)
def test_plot_instance_incorrect_selection(transport, incorrect_selection, Assert):
    with pytest.raises(exception.IncorrectUsage):
        transport[0].plot(incorrect_selection)


def test_figure_of_merit(transport, Assert):
    for instance in transport:
        seebeck = instance.seebeck()
        sigma = instance.electronic_conductivity()
        kappa = instance.electronic_thermal_conductivity()
        temperature = instance.temperatures()
        expected = seebeck**2 * sigma * temperature / kappa
        Assert.allclose(instance.figure_of_merit(), expected)


@pytest.mark.parametrize("kappa_lattice", [1.0, np.linspace(1.0, 3.0, 6)])
def test_figure_of_merit_with_argument(transport, kappa_lattice, Assert):
    instance = transport[3]
    seebeck = instance.seebeck()
    sigma = instance.electronic_conductivity()
    kappa_electronic = instance.electronic_thermal_conductivity()
    temperature = instance.temperatures()
    expected = seebeck**2 * sigma * temperature / (kappa_electronic + kappa_lattice)
    Assert.allclose(instance.figure_of_merit(kappa_lattice), expected)


def test_figure_of_merit_with_wrong_size_argument(transport):
    instance = transport[0]
    kappa_lattice = np.linspace(1.0, 3.0, 5)  # Incorrect size
    with pytest.raises(exception.IncorrectUsage):
        instance.figure_of_merit(kappa_lattice)


def test_print_mapping(transport, format_):
    actual, _ = format_(transport)
    assert re.search(transport.ref.mapping_pattern, str(transport), re.MULTILINE)
    assert actual == {"text/plain": str(transport)}


def test_print_instance(transport, format_):
    instance = transport[0]
    actual, _ = format_(instance)
    # Check if the actual output matches the expected pattern
    assert re.search(transport.ref.instance_pattern, str(instance), re.MULTILINE)
    assert actual == {"text/plain": str(instance)}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.electron_phonon_transport("default")
    parameters = {
        "select": {"selection": "scattering_approx=MRTA_TAU"},
        "plot": {"selection": "mobility"},
        "to_graph": {"selection": "mobility"},
        "to_frame": {"selection": "mobility"},
        "to_plotly": {"selection": "mobility"},
    }
    skip_methods = ["count", "access", "index"]  # inherited from Sequence
    check_factory_methods(ElectronPhononTransport, data, parameters, skip_methods)

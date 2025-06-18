# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import random
import re
import types

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._calculation.electron_phonon_bandgap import (
    ElectronPhononBandgap,
    ElectronPhononBandgapInstance,
)


@pytest.fixture(params=["nonpolarized", "collinear"])
def raw_band_gap(request, raw_data):
    return raw_data.electron_phonon_band_gap(request.param)


@pytest.fixture
def band_gap(raw_band_gap):
    band_gap = ElectronPhononBandgap.from_data(raw_band_gap)
    band_gap.ref = types.SimpleNamespace()
    mask = np.array(raw_band_gap.scattering_approximation) == "SERTA"
    band_gap.ref.naccumulators = 4
    band_gap.ref.indices = np.arange(5)[mask]
    band_gap.ref.fundamental = raw_band_gap.fundamental
    band_gap.ref.fundamental_renorm = raw_band_gap.fundamental_renorm
    band_gap.ref.direct = raw_band_gap.direct
    band_gap.ref.direct_renorm = raw_band_gap.direct_renorm
    band_gap.ref.temperatures = raw_band_gap.temperatures
    band_gap.ref.nbands_sum = raw_band_gap.nbands_sum
    band_gap.ref.selfen_delta = raw_band_gap.delta
    band_gap.ref.selfen_carrier_den = _make_reference_carrier_den(raw_band_gap)
    band_gap.ref.mapping_pattern = _make_reference_pattern()
    band_gap.ref.instance_pattern = _make_reference_pattern(raw_band_gap)
    return band_gap


@pytest.fixture(params=["carrier_den", "carrier_per_cell", "mu"])
def chemical_potential(raw_data, request):
    raw_potential = raw_data.electron_phonon_chemical_potential(request.param)
    raw_potential.ref = types.SimpleNamespace()
    raw_potential.ref.param = request.param
    raw_potential.ref.expected_data = getattr(raw_potential, request.param)
    return raw_potential


def _make_reference_carrier_den(raw_band_gap):
    chemical_potential = raw_band_gap.chemical_potential
    indices = raw_band_gap.id_index[:, 2] - 1
    return np.array([chemical_potential.carrier_den[index_] for index_ in indices])


def _make_reference_pattern(raw_band_gap=None):
    if raw_band_gap is None:
        return r"""Electron-phonon bandgap with 4 instance\(s\):
    selfen_carrier_den: \[.*\]
    nbands_sum: \[.*\]
    selfen_delta: \[.*\]"""
    if raw_band_gap.fundamental.shape[-1] == 1:
        return r"""Direct gap:
   Temperature \(K\)         KS gap \(eV\)         QP gap \(eV\)     KS-QP gap \(meV\)
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}

Fundamental gap:
   Temperature \(K\)         KS gap \(eV\)         QP gap \(eV\)     KS-QP gap \(meV\)
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}"""
    elif raw_band_gap.fundamental.shape[-1] == 3:
        return r"""spin independent

Direct gap:
   Temperature \(K\)         KS gap \(eV\)         QP gap \(eV\)     KS-QP gap \(meV\)
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}

Fundamental gap:
   Temperature \(K\)         KS gap \(eV\)         QP gap \(eV\)     KS-QP gap \(meV\)
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}

spin component 1

Direct gap:
   Temperature \(K\)         KS gap \(eV\)         QP gap \(eV\)     KS-QP gap \(meV\)
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}

Fundamental gap:
   Temperature \(K\)         KS gap \(eV\)         QP gap \(eV\)     KS-QP gap \(meV\)
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}

spin component 2

Direct gap:
   Temperature \(K\)         KS gap \(eV\)         QP gap \(eV\)     KS-QP gap \(meV\)
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}

Fundamental gap:
   Temperature \(K\)         KS gap \(eV\)         QP gap \(eV\)     KS-QP gap \(meV\)
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}
^\s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6} \s*[-+]?\d*\.\d{6}"""
    else:
        raise NotImplementedError


def test_len(band_gap):
    # Should match the number of valid indices in the raw data
    assert len(band_gap) == band_gap.ref.naccumulators


def test_indexing_and_iteration(band_gap):
    # Indexing and iteration should yield instances
    for i, instance in enumerate(band_gap):
        assert isinstance(instance, ElectronPhononBandgapInstance)
        assert instance.index == band_gap.ref.indices[i]
        assert instance.parent is band_gap
    assert isinstance(band_gap[0], ElectronPhononBandgapInstance)


def test_read_mapping(band_gap):
    # Check that to_dict returns expected keys
    assert band_gap.to_dict() == {"naccumulators": band_gap.ref.naccumulators}


def test_read_instance(band_gap, Assert):
    # Each instance's to_dict should match the raw data for that index
    for i, instance in enumerate(band_gap):
        d = instance.read()
        index = band_gap.ref.indices[i]
        Assert.allclose(d["fundamental"], band_gap.ref.fundamental[index])
        Assert.allclose(d["fundamental_renorm"], band_gap.ref.fundamental_renorm[index])
        Assert.allclose(d["direct"], band_gap.ref.direct[index])
        Assert.allclose(d["direct_renorm"], band_gap.ref.direct_renorm[index])
        Assert.allclose(d["temperatures"], band_gap.ref.temperatures[index])
        assert d["metadata"] == {
            "nbands_sum": band_gap.ref.nbands_sum[index],
            "selfen_delta": band_gap.ref.selfen_delta[index],
            "selfen_carrier_den": band_gap.ref.selfen_carrier_den[index],
            "scattering_approx": "SERTA",
        }


def test_plot_instance(band_gap, Assert):
    # Plotting should not raise an error and return a graph object
    graph = band_gap[0].plot("direct_renorm")
    assert len(graph) == 1
    Assert.allclose(graph.series[0].x, band_gap.ref.temperatures[0])
    Assert.allclose(graph.series[0].y, band_gap.ref.direct_renorm[0])
    assert graph.xlabel == "Temperature (K)"
    assert graph.ylabel == "Energy (eV)"


def test_plot_multiple_selections(band_gap, Assert):
    graph = band_gap[1].plot("fundamental, fundamental_renorm")
    assert len(graph) == 2
    Assert.allclose(graph.series[0].x, band_gap.ref.temperatures[1])
    Assert.allclose(graph.series[0].y, band_gap.ref.fundamental[1, :, np.newaxis])
    assert graph.series[0].label == "fundamental"
    Assert.allclose(graph.series[1].x, band_gap.ref.temperatures[1])
    Assert.allclose(graph.series[1].y, band_gap.ref.fundamental_renorm[1])
    assert graph.series[1].label == "fundamental_renorm"


def test_plot_direct_gap_renormalization(band_gap, Assert):
    # Plotting the direct gap should return a graph with correct data
    i = 3
    j = band_gap.ref.indices[i]
    expected = band_gap.ref.direct[j, :, np.newaxis] - band_gap.ref.direct_renorm[j]
    graph = band_gap[i].plot("direct - direct_renorm")
    assert len(graph) == 1
    Assert.allclose(graph.series[0].y, expected)


def test_selections(raw_band_gap, chemical_potential, Assert):
    # Should return a dictionary with expected selection keys
    raw_band_gap.chemical_potential = chemical_potential
    band_gap = ElectronPhononBandgap.from_data(raw_band_gap)
    selections = band_gap.selections()
    selections.pop("electron_phonon_bandgap")
    expected = selections.pop(f"selfen_{chemical_potential.ref.param}")
    Assert.allclose(expected, np.unique(chemical_potential.ref.expected_data))
    assert selections.keys() == {"nbands_sum", "selfen_delta"}
    Assert.allclose(selections["nbands_sum"], np.unique(raw_band_gap.nbands_sum))
    Assert.allclose(selections["selfen_delta"], np.unique(raw_band_gap.delta))


@pytest.mark.parametrize(
    "attribute", ["nbands_sum", "selfen_delta", "selfen_carrier_den"]
)
def test_select_returns_instances(band_gap, attribute):
    choices = getattr(band_gap.ref, attribute)
    choice = random.choice(list(choices[band_gap.ref.indices]))
    indices, *_ = np.where(choices == choice)
    selected = band_gap.select(f"{attribute}={choice.item()}")
    assert len(selected) == len(indices)
    for index_, instance in zip(indices, selected):
        assert isinstance(instance, ElectronPhononBandgapInstance)
        assert instance.index == index_


def test_select_non_SERTA_returns_empty(band_gap):
    # Selecting an element corresponding to a scattering approximation other than SERTA
    # should return an empty list
    index_nonserta = 2
    assert index_nonserta not in band_gap.ref.indices
    choice = band_gap.ref.nbands_sum[index_nonserta]
    selected = band_gap.select(f"nbands_sum={choice.item()}")
    assert len(selected) == 0


def test_select_multiple(band_gap):
    index_nbands_sum = 1
    index_selfen_delta = 3
    indices = [index_nbands_sum, index_selfen_delta]
    choice_nbands_sum = band_gap.ref.nbands_sum[index_nbands_sum]
    choice_selfen_delta = band_gap.ref.selfen_delta[index_selfen_delta]
    selection = f"nbands_sum={choice_nbands_sum.item()}, selfen_delta={choice_selfen_delta.item()}"
    selected = band_gap.select(selection)
    assert len(selected) == len(indices)
    for index_, instance in zip(indices, selected):
        assert isinstance(instance, ElectronPhononBandgapInstance)
        assert instance.index == index_


def test_select_nested(band_gap):
    index_ = 0
    choice_nbands_sum = band_gap.ref.nbands_sum[index_]
    choice_selfen_carrier_den = band_gap.ref.selfen_carrier_den[index_]
    count_ = sum(band_gap.ref.selfen_carrier_den == choice_selfen_carrier_den)
    assert count_ > 1
    selection = f"nbands_sum={choice_nbands_sum.item()}(selfen_carrier_den={choice_selfen_carrier_den.item()})"
    selected = band_gap.select(selection)
    assert len(selected) == 1
    instance = selected[0]
    assert isinstance(instance, ElectronPhononBandgapInstance)
    assert instance.index == index_


@pytest.mark.parametrize(
    "selection",
    ["invalid_selection=0.01", "nbands_sum:0.01", "selfen_delta"],
)
def test_incorrect_selection(band_gap, selection):
    with pytest.raises(exception.IncorrectUsage):
        band_gap.select(selection)


def test_print_mapping(band_gap, format_):
    actual, _ = format_(band_gap)
    assert re.search(band_gap.ref.mapping_pattern, str(band_gap), re.MULTILINE)
    assert actual == {"text/plain": str(band_gap)}


def test_print_instance(band_gap, format_):
    instance = band_gap[0]
    actual, _ = format_(instance)
    # Check if the actual output matches the expected pattern
    assert re.search(band_gap.ref.instance_pattern, str(instance), re.MULTILINE)
    assert actual == {"text/plain": str(instance)}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.electron_phonon_band_gap("default")
    parameters = {
        "select": {"selection": "selfen_carrier_den=0.01"},
    }
    skip_methods = ["count", "access", "index"]  # inherited from Sequence
    check_factory_methods(ElectronPhononBandgap, data, parameters, skip_methods)

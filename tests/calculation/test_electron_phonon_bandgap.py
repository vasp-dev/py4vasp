# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import re
import types

import numpy as np
import pytest

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
    band_gap.ref.naccumulators = len(raw_band_gap.valid_indices)
    band_gap.ref.fundamental = raw_band_gap.fundamental
    band_gap.ref.fundamental_renorm = raw_band_gap.fundamental_renorm
    band_gap.ref.direct = raw_band_gap.direct
    band_gap.ref.direct_renorm = raw_band_gap.direct_renorm
    band_gap.ref.temperatures = raw_band_gap.temperatures
    band_gap.ref.nbands_sum = raw_band_gap.nbands_sum
    band_gap.ref.delta = raw_band_gap.delta
    band_gap.ref.carrier_den = raw_band_gap.chemical_potential.carrier_den
    band_gap.ref.pattern = _make_reference_pattern(raw_band_gap)
    return band_gap


@pytest.fixture(params=["carrier_den", "carrier_per_cell", "mu"])
def chemical_potential(raw_data, request):
    raw_potential = raw_data.electron_phonon_chemical_potential(request.param)
    raw_potential.ref = types.SimpleNamespace()
    raw_potential.ref.param = request.param
    raw_potential.ref.expected_data = getattr(raw_potential, request.param)
    return raw_potential


def _make_reference_pattern(raw_band_gap):
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
        assert instance.index == i
        assert instance.parent is band_gap
    assert isinstance(band_gap[0], ElectronPhononBandgapInstance)


def test_read_mapping(band_gap):
    # Check that to_dict returns expected keys
    assert band_gap.to_dict() == {"naccumulators": band_gap.ref.naccumulators}


def test_read_instance(band_gap, Assert):
    # Each instance's to_dict should match the raw data for that index
    for i, instance in enumerate(band_gap):
        d = instance.read()
        Assert.allclose(d["fundamental"], band_gap.ref.fundamental[i])
        Assert.allclose(d["fundamental_renorm"], band_gap.ref.fundamental_renorm[i])
        Assert.allclose(d["direct"], band_gap.ref.direct[i])
        Assert.allclose(d["direct_renorm"], band_gap.ref.direct_renorm[i])
        Assert.allclose(d["temperatures"], band_gap.ref.temperatures[i])
        assert d["metadata"] == {
            "nbands_sum": band_gap.ref.nbands_sum[i],
            "selfen_delta": band_gap.ref.delta[i],
            "selfen_carrier_den": band_gap.ref.carrier_den[i],
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
    fundamental_gap = band_gap.ref.fundamental[1, :, np.newaxis]
    expected = fundamental_gap + band_gap.ref.fundamental_renorm[1]
    assert len(graph) == 2
    Assert.allclose(graph.series[0].x, band_gap.ref.temperatures[1])
    Assert.allclose(graph.series[0].y, expected)
    assert graph.series[0].label == "fundamental"
    Assert.allclose(graph.series[1].x, band_gap.ref.temperatures[1])
    Assert.allclose(graph.series[1].y, band_gap.ref.fundamental_renorm[1])
    assert graph.series[1].label == "fundamental_renorm"


def test_plot_direct_gap(band_gap, Assert):
    # Plotting the direct gap should return a graph with correct data
    graph = band_gap[2].plot("direct - direct_renorm")
    assert len(graph) == 1
    Assert.allclose(graph.series[0].y, band_gap.ref.direct[2, :, np.newaxis])


def test_selections(raw_band_gap, chemical_potential, Assert):
    # Should return a dictionary with expected selection keys
    raw_band_gap.chemical_potential = chemical_potential
    band_gap = ElectronPhononBandgap.from_data(raw_band_gap)
    selections = band_gap.selections()
    selections.pop("electron_phonon_bandgap")
    expected = selections.pop(f"selfen_{chemical_potential.ref.param}")
    Assert.allclose(expected, chemical_potential.ref.expected_data)
    assert selections.keys() == {"nbands_sum", "selfen_delta"}
    Assert.allclose(selections["nbands_sum"], raw_band_gap.nbands_sum)
    Assert.allclose(selections["selfen_delta"], raw_band_gap.delta)


# @pytest.mark.xfail(reason="This test is expected to fail due to missing implementation")
def test_select_returns_instances(band_gap):
    selected = band_gap.select("nbands_sum=12 scattering_approximation=SERTA")
    assert len(selected) == 5
    assert all(isinstance(x, ElectronPhononBandgapInstance) for x in selected)


def test_print_mapping(band_gap, format_):
    actual, _ = format_(band_gap)
    assert actual["text/plain"] == "electron phonon bandgap"


def test_print_instance(band_gap, format_):
    instance = band_gap[0]
    actual, _ = format_(instance)
    # Check if the actual output matches the expected pattern
    assert re.search(band_gap.ref.pattern, str(instance), re.MULTILINE)
    assert re.search(band_gap.ref.pattern, actual["text/plain"], re.MULTILINE)


# def test_factory_methods(raw_data, check_factory_methods):
#     data = raw_data.electron_phonon_band_gap("default")
#     parameters = {
#         "select": {"selection": "scattering_approximation=SERTA carrier_den=0.01"},
#     }
#     check_factory_methods(ElectronPhononBandgap, data, parameters)

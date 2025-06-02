# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp._calculation.electron_phonon_bandgap import ElectronPhononBandgap


@pytest.fixture
def band_gap(raw_data):
    raw_band_gap = raw_data.electron_phonon_band_gap("default")
    band_gap = ElectronPhononBandgap.from_data(raw_band_gap)
    band_gap.ref = types.SimpleNamespace()
    band_gap.ref.fundamental = raw_band_gap.fundamental
    band_gap.ref.direct = raw_band_gap.direct
    return band_gap


def test_len(band_gap):
    # Should match the number of valid indices in the raw data
    assert len(band_gap) == len(band_gap._raw_data.valid_indices)


def test_to_dict_keys(band_gap):
    # Check that to_dict returns expected keys
    d = band_gap.to_dict()
    assert "naccumulators" in d
    assert d["naccumulators"] == len(band_gap)


def test_selections(band_gap):
    # Should return a dictionary with expected selection keys
    selections = band_gap.selections()
    assert isinstance(selections, dict)
    assert "nbands_sum" in selections
    assert "selfen_approx" in selections
    assert "selfen_delta" in selections
    assert "selfen_carrier_den" in selections or \
           "selfen_carrier_cell" in selections or \
           "selfen_mu" in selections


def test_select_returns_instances(band_gap):
    # Should return a list of ElectronPhononBandgapInstance
    selections = band_gap.selections()
    from py4vasp._calculation.electron_phonon_bandgap import ElectronPhononBandgapInstance
    for nbands_sum in selections["nbands_sum"]:
        for selfen_approx in selections["selfen_approx"]:
            #check if we got an ElectronPhononBandgapInstance
            selected = band_gap.select(f"nbands_sum({nbands_sum}) selfen_approx({selfen_approx})")
            assert len(selected) == 3
            assert all(isinstance(x, ElectronPhononBandgapInstance) for x in selected)


def test_indexing_and_iteration(band_gap):
    # Indexing and iteration should yield instances
    from py4vasp._calculation.electron_phonon_bandgap import ElectronPhononBandgapInstance
    for i, instance in enumerate(band_gap):
        assert isinstance(instance, ElectronPhononBandgapInstance)
        assert instance.index == i
        assert instance.parent is band_gap
    assert isinstance(band_gap[0], ElectronPhononBandgapInstance)


def test_to_dict_instance_matches_raw(band_gap):
    # Each instance's to_dict should match the raw data for that index
    for i in range(len(band_gap)):
        d = band_gap[i].to_dict()
        assert "fundamental" in d
        assert "direct" in d
        assert "temperatures" in d
        assert "nbands_sum" in d
        # Check shape matches
        assert d["fundamental"].shape == band_gap.ref.fundamental[i].shape
        assert d["direct"].shape == band_gap.ref.direct[i].shape


def test_read(band_gap, Assert):
    slice_ = 0
    actual = band_gap[slice_].to_dict()
    Assert.allclose(actual["fundamental"], band_gap.ref.fundamental[slice_])
    Assert.allclose(actual["direct"], band_gap.ref.direct[slice_])


def test_print(band_gap, format_):
    actual, _ = format_(band_gap)
    assert actual["text/plain"] == "electron phonon bandgap"


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.electron_phonon_band_gap("default")
    parameters = {
        "select": {"selection": "selfen_approx(SERTA) selfen_carrier_den(0.01,0.001)"},
    }
    check_factory_methods(ElectronPhononBandgap, data, parameters)

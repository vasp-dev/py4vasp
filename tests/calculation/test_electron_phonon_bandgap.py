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
    parameters = {}
    check_factory_methods(ElectronPhononBandgap, data, parameters)

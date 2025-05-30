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
    band_gap.ref.eigenvalues = raw_band_gap.eigenvalues
    band_gap.ref.debye_waller = raw_band_gap.debye_waller
    band_gap.ref.fan = raw_band_gap.fan
    return band_gap


def test_read(band_gap, Assert):
    slice_ = 0
    actual = band_gap[slice_].to_dict()
    Assert.allclose(actual["eigenvalues"], band_gap.ref.eigenvalues)
    Assert.allclose(actual["debye_waller"], band_gap.ref.debye_waller[slice_])
    Assert.allclose(actual["fan"], band_gap.ref.fan[slice_])


def test_print(band_gap, format_):
    actual, _ = format_(band_gap)
    assert actual["text/plain"] == "electron phonon self energy"


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.electron_phonon_band_gap("default")
    # parameters = {"get_fan": {"arg": (0, 0, 0)}, "select": {"selection": "1 1"}}
    check_factory_methods(ElectronPhononBandGap, data, parameters)

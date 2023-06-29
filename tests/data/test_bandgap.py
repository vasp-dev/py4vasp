# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp import data


@pytest.fixture
def bandgap(raw_data):
    VBM = 0
    CBM = 1
    BOTTOM = 2
    TOP = 3
    FERMI = 4
    KPOINT_VBM = slice(5, 8)
    KPOINT_CBM = slice(8, 11)
    KPOINT_OPTICAL = slice(11, 14)
    raw_bandgap = raw_data.bandgap("default")
    bandgap = data.Bandgap.from_data(raw_bandgap)
    bandgap.ref = types.SimpleNamespace()
    bandgap.ref.fundamental = raw_bandgap.values[:, CBM] - raw_bandgap.values[:, VBM]
    bandgap.ref.kpoint_vbm = raw_bandgap.values[:, KPOINT_VBM]
    bandgap.ref.kpoint_cbm = raw_bandgap.values[:, KPOINT_CBM]
    bandgap.ref.optical = raw_bandgap.values[:, TOP] - raw_bandgap.values[:, BOTTOM]
    bandgap.ref.kpoint_optical = raw_bandgap.values[:, KPOINT_OPTICAL]
    bandgap.ref.fermi_energy = raw_bandgap.values[:, FERMI]
    return bandgap


@pytest.fixture(params=[slice(None), slice(1, 3), 0, -1])
def steps(request):
    return request.param


def test_read(bandgap, steps, Assert):
    actual = bandgap.read() if steps == -1 else bandgap[steps].read()
    Assert.allclose(actual["fundamental"], bandgap.ref.fundamental[steps])
    Assert.allclose(actual["kpoint_VBM"], bandgap.ref.kpoint_vbm[steps])
    Assert.allclose(actual["kpoint_CBM"], bandgap.ref.kpoint_cbm[steps])
    Assert.allclose(actual["optical"], bandgap.ref.optical[steps])
    Assert.allclose(actual["kpoint_optical"], bandgap.ref.kpoint_optical[steps])
    Assert.allclose(actual["fermi_energy"], bandgap.ref.fermi_energy[steps])

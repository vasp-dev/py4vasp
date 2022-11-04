import numpy as np
import pytest
import types
from py4vasp.data import Fatband, Dispersion


@pytest.fixture
def fatband(raw_data):
    raw_fatband = raw_data.fatband("default")
    fatband = Fatband.from_data(raw_fatband)
    fatband.ref = types.SimpleNamespace()
    fatband.ref.dispersion = Dispersion.from_data(raw_fatband.dispersion)
    fatbands = raw_fatband.fatbands
    fatband.ref.fatbands = fatbands[:,:,0] + fatbands[:,:,1] * 1j
    fatband.ref.fermi_energy = raw_fatband.fermi_energy
    # convert to Python indices
    fatband.ref.bse_index = raw_fatband.bse_index - 1
    fatband.ref.first_valence_band = raw_fatband.first_valence_band - 1
    fatband.ref.first_conduction_band = raw_fatband.first_conduction_band - 1
    return fatband


def test_fatband_read(fatband, Assert):
    actual = fatband.read()
    dispersion = fatband.ref.dispersion.read()
    Assert.allclose(actual["kpoint_distances"], dispersion["kpoint_distances"])
    assert np.all(actual["kpoint_labels"] == dispersion["kpoint_labels"])
    bands = dispersion["eigenvalues"] - fatband.ref.fermi_energy
    Assert.allclose(actual["bands"], bands)
    assert np.all(actual["bse_index"] == fatband.ref.bse_index)
    Assert.allclose(actual["fatbands"], fatband.ref.fatbands)
    Assert.allclose(actual["fermi_energy"], fatband.ref.fermi_energy)
    assert actual["first_valence_band"] == fatband.ref.first_valence_band
    assert actual["first_conduction_band"] == fatband.ref.first_conduction_band


def test_fatband_print(fatband, format_):
    actual, _ = format_(fatband)
    reference = """\
BSE fatband data:
    48 k-points
    2 valence bands
    1 conduction bands"""
    assert actual == {"text/plain": reference}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.fatband("default")
    check_factory_methods(Fatband, data)

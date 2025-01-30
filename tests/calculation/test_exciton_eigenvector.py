# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import numpy as np
import pytest

from py4vasp._calculation._dispersion import Dispersion
from py4vasp._calculation.exciton_eigenvector import ExcitonEigenvector


@pytest.fixture
def exciton_eigenvector(raw_data):
    raw_eigenvector = raw_data.exciton_eigenvector("default")
    eigenvector = ExcitonEigenvector.from_data(raw_eigenvector)
    eigenvector.ref = types.SimpleNamespace()
    eigenvector.ref.dispersion = Dispersion.from_data(raw_eigenvector.dispersion)
    eigenvectors = raw_eigenvector.eigenvectors
    eigenvector.ref.eigenvectors = eigenvectors[:, :, 0] + eigenvectors[:, :, 1] * 1j
    eigenvector.ref.fermi_energy = raw_eigenvector.fermi_energy
    # convert to Python indices
    eigenvector.ref.bse_index = raw_eigenvector.bse_index - 1
    eigenvector.ref.first_valence_band = raw_eigenvector.first_valence_band - 1
    eigenvector.ref.first_conduction_band = raw_eigenvector.first_conduction_band - 1
    return eigenvector


def test_eigenvector_read(exciton_eigenvector, Assert):
    actual = exciton_eigenvector.read()
    dispersion = exciton_eigenvector.ref.dispersion.read()
    Assert.allclose(actual["kpoint_distances"], dispersion["kpoint_distances"])
    assert np.all(actual["kpoint_labels"] == dispersion["kpoint_labels"])
    bands = dispersion["eigenvalues"] - exciton_eigenvector.ref.fermi_energy
    Assert.allclose(actual["bands"], bands)
    assert np.all(actual["bse_index"] == exciton_eigenvector.ref.bse_index)
    Assert.allclose(actual["eigenvectors"], exciton_eigenvector.ref.eigenvectors)
    Assert.allclose(actual["fermi_energy"], exciton_eigenvector.ref.fermi_energy)
    assert actual["first_valence_band"] == exciton_eigenvector.ref.first_valence_band
    assert (
        actual["first_conduction_band"] == exciton_eigenvector.ref.first_conduction_band
    )


def test_eigenvector_print(exciton_eigenvector, format_):
    actual, _ = format_(exciton_eigenvector)
    reference = """\
BSE eigenvector data:
    48 k-points
    2 valence bands
    1 conduction bands"""
    assert actual == {"text/plain": reference}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.exciton_eigenvector("default")
    check_factory_methods(ExcitonEigenvector, data)

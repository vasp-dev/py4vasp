# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest
import types
from py4vasp.data import Dispersion, Kpoint


@pytest.fixture
def single_band(raw_data):
    raw_dispersion = raw_data.dispersion("single_band")
    dispersion = Dispersion.from_data(raw_dispersion)
    dispersion.ref = types.SimpleNamespace()
    dispersion.ref.kpoints = Kpoint.from_data(raw_dispersion.kpoints)
    dispersion.ref.eigenvalues = raw_dispersion.eigenvalues
    return dispersion


def test_read_single_band(single_band, Assert):
    dispersion = single_band.read()
    Assert.allclose(dispersion["kpoint_distances"], single_band.ref.kpoints.distances())
    assert dispersion["kpoint_labels"] == single_band.ref.kpoints.labels()
    Assert.allclose(dispersion["eigenvalues"], single_band.ref.eigenvalues)

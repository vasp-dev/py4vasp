# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest
import types
from py4vasp.data import Dispersion, Kpoint


@pytest.fixture(params=["single_band", "spin_polarized", "line", "phonon"])
def dispersion(raw_data, request):
    raw_dispersion = raw_data.dispersion(request.param)
    dispersion = Dispersion.from_data(raw_dispersion)
    dispersion.ref = types.SimpleNamespace()
    dispersion.ref.kpoints = Kpoint.from_data(raw_dispersion.kpoints)
    dispersion.ref.eigenvalues = raw_dispersion.eigenvalues
    return dispersion


def test_read_dispersion(dispersion, Assert):
    actual = dispersion.read()
    Assert.allclose(actual["kpoint_distances"], dispersion.ref.kpoints.distances())
    assert actual["kpoint_labels"] == dispersion.ref.kpoints.labels()
    Assert.allclose(actual["eigenvalues"], dispersion.ref.eigenvalues)

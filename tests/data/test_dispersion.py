# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
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
    dispersion.ref.spin_polarized = request.param == "spin_polarized"
    return dispersion


def test_read_dispersion(dispersion, Assert):
    actual = dispersion.read()
    Assert.allclose(actual["kpoint_distances"], dispersion.ref.kpoints.distances())
    assert actual["kpoint_labels"] == dispersion.ref.kpoints.labels()
    Assert.allclose(actual["eigenvalues"], dispersion.ref.eigenvalues)


def test_plot_dispersion(dispersion, Assert):
    graph = dispersion.plot()
    expected_num_series = 2 if dispersion.ref.spin_polarized else 1
    assert len(graph.series) == expected_num_series
    bands = np.atleast_3d(dispersion.ref.eigenvalues.T)
    for component, series in enumerate(graph.series):
        assert series.width is None
        Assert.allclose(series.x, dispersion.ref.kpoints.distances())
        Assert.allclose(series.y, bands[:, :, component])

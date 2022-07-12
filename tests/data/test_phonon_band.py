# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools
import pytest
import types
from py4vasp.data import PhononBand, Kpoint, Topology


@pytest.fixture
def phonon_band(raw_data):
    raw_band = raw_data.phonon_band("default")
    band = PhononBand.from_data(raw_band)
    band.ref = types.SimpleNamespace()
    band.ref.bands = raw_band.dispersion.eigenvalues
    band.ref.modes = raw_band.eigenvectors
    raw_qpoints = raw_band.dispersion.kpoints
    band.ref.kpoints = Kpoint.from_data(raw_qpoints)
    band.ref.topology = Topology.from_data(raw_band.topology)
    return band


def test_read(phonon_band, Assert):
    band = phonon_band.read()
    Assert.allclose(band["bands"], phonon_band.ref.bands)
    Assert.allclose(band["modes"], phonon_band.ref.modes)


def test_plot(phonon_band, Assert):
    graph = phonon_band.plot()
    assert graph.ylabel == "Energy (meV)"
    assert len(graph.series) == 1
    assert graph.series[0].width is None
    Assert.allclose(graph.series[0].x, phonon_band.ref.kpoints.distances())
    Assert.allclose(graph.series[0].y, phonon_band.ref.bands.T)


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.phonon_band("default")
    check_factory_methods(PhononBand, data)
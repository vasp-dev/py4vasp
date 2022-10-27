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
    spin_polarized = request.param == "spin_polarized"
    dispersion.ref.spin_polarized = spin_polarized
    dispersion.ref.names = ("up", "down") if spin_polarized else ("bands",)
    dispersion.ref.xticks = expected_xticks(request.param)
    return dispersion


def expected_xticks(selection):
    if selection == "line":
        return (
            "$[0 0 0]$",
            "$[0 0 \\frac{1}{2}]$",
            "$[\\frac{1}{2} \\frac{1}{2} \\frac{1}{2}]$|$[0 0 0]$",
            "$[\\frac{1}{2} \\frac{1}{2} 0]$",
            "$[\\frac{1}{2} \\frac{1}{2} \\frac{1}{2}]$",
        )
    elif selection == "phonon":
        return (r"$\Gamma$", "", r"M|$\Gamma$", "Y", "M")
    else:
        return ("", "")  # empty labels


def test_read_dispersion(dispersion, Assert):
    actual = dispersion.read()
    Assert.allclose(actual["kpoint_distances"], dispersion.ref.kpoints.distances())
    assert actual["kpoint_labels"] == dispersion.ref.kpoints.labels()
    Assert.allclose(actual["eigenvalues"], dispersion.ref.eigenvalues)


def test_plot_dispersion(dispersion, Assert):
    graph = dispersion.plot()
    assert len(graph.series) == len(dispersion.ref.names)
    check_xticks(graph.xticks, dispersion.ref, Assert)
    bands = np.atleast_3d(dispersion.ref.eigenvalues.T)
    for component, (series, name) in enumerate(zip(graph.series, dispersion.ref.names)):
        Assert.allclose(series.x, dispersion.ref.kpoints.distances())
        Assert.allclose(series.y, bands[:, :, component])
        assert series.name == name
        assert series.width is None


def check_xticks(actual, reference, Assert):
    dists = reference.kpoints.distances()
    xticks = (*dists[:: reference.kpoints.line_length()], dists[-1])
    Assert.allclose(list(actual.keys()), np.array(xticks))
    assert tuple(actual.values()) == reference.xticks


def test_plot_dispersion_with_projections(dispersion, Assert):
    shape = dispersion.ref.eigenvalues.shape[-2], dispersion.ref.eigenvalues.shape[-1]
    if dispersion.ref.spin_polarized:
        projections = {
            "one up": np.random.uniform(low=0.1, high=0.5, size=shape),
            "one down": np.random.uniform(low=0.1, high=0.5, size=shape),
            "two up": np.random.uniform(low=0.1, high=0.5, size=shape),
        }
    else:
        projections = {
            "one": np.random.uniform(low=0.1, high=0.5, size=shape),
            "two": np.random.uniform(low=0.1, high=0.5, size=shape),
        }
    graph = dispersion.plot(projections)
    assert len(graph.series) == len(projections)
    check_xticks(graph.xticks, dispersion.ref, Assert)
    bands = np.atleast_3d(dispersion.ref.eigenvalues.T)
    for series, (name, width) in zip(graph.series, projections.items()):
        component = 1 if "down" in name else 0
        Assert.allclose(series.x, dispersion.ref.kpoints.distances())
        Assert.allclose(series.y, bands[:, :, component])
        assert series.name == name
        Assert.allclose(series.width, width.T)


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.dispersion("single_band")
    check_factory_methods(Dispersion, data)

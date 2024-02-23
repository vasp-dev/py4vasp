# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import types

import pytest

from py4vasp import calculation


@pytest.fixture
def OSZICAR(raw_data):
    raw_oszicar = raw_data.OSZICAR()
    oszicar = calculation.OSZICAR.from_data(raw_oszicar)
    oszicar.ref = types.SimpleNamespace()
    convergence_data = raw_oszicar.convergence_data
    oszicar.ref.N = convergence_data[:, 0]
    oszicar.ref.E = convergence_data[:, 1]
    oszicar.ref.dE = convergence_data[:, 2]
    oszicar.ref.deps = convergence_data[:, 3]
    oszicar.ref.ncg = convergence_data[:, 4]
    oszicar.ref.rms = convergence_data[:, 5]
    oszicar.ref.rmsc = convergence_data[:, 6]
    return oszicar


def test_read(OSZICAR, Assert):
    actual = OSZICAR.read()
    expected = OSZICAR.ref
    print(actual)
    print(expected)
    Assert.allclose(actual["N"], expected.N)
    Assert.allclose(actual["E"], expected.E)
    Assert.allclose(actual["dE"], expected.dE)
    Assert.allclose(actual["deps"], expected.deps)
    Assert.allclose(actual["ncg"], expected.ncg)
    Assert.allclose(actual["rms"], expected.rms)
    Assert.allclose(actual["rms(c)"], expected.rmsc)


def test_plot(OSZICAR, Assert):
    graph = OSZICAR.plot()
    assert graph.xlabel == "Iteration number"
    assert graph.ylabel == "E"
    assert len(graph.series) == 1
    Assert.allclose(graph.series[0].x, OSZICAR.ref.N)
    Assert.allclose(graph.series[0].y, OSZICAR.ref.E)

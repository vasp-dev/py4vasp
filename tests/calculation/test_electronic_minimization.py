# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import types

import numpy as np
import pytest

from py4vasp import calculation, exception


@pytest.fixture
def ElectronicMinimization(raw_data):
    raw_oszicar = raw_data.electronic_minimization()
    oszicar = calculation.electronic_minimization.from_data(raw_oszicar)
    oszicar.ref = types.SimpleNamespace()
    convergence_data = raw_oszicar.convergence_data
    oszicar.ref.N = np.int64(convergence_data[:, 0])
    oszicar.ref.E = convergence_data[:, 1]
    oszicar.ref.dE = convergence_data[:, 2]
    oszicar.ref.deps = convergence_data[:, 3]
    oszicar.ref.ncg = convergence_data[:, 4]
    oszicar.ref.rms = convergence_data[:, 5]
    oszicar.ref.rmsc = convergence_data[:, 6]
    oszicar.ref.is_elmin_converged = [raw_oszicar.is_elmin_converged == [0.0]]
    string_rep = "N\t\tE\t\tdE\t\tdeps\t\tncg\trms\t\trms(c)\n"
    format_rep = "{0:g}\t{1:0.12E}\t{2:0.6E}\t{3:0.6E}\t{4:g}\t{5:0.3E}\t{6:0.3E}\n"
    for idx in range(len(convergence_data)):
        string_rep += format_rep.format(*convergence_data[idx])
    oszicar.ref.string_rep = str(string_rep)
    return oszicar


def test_read(ElectronicMinimization, Assert):
    actual = ElectronicMinimization.read()
    expected = ElectronicMinimization.ref
    Assert.allclose(actual["N"], expected.N)
    Assert.allclose(actual["E"], expected.E)
    Assert.allclose(actual["dE"], expected.dE)
    Assert.allclose(actual["deps"], expected.deps)
    Assert.allclose(actual["ncg"], expected.ncg)
    Assert.allclose(actual["rms"], expected.rms)
    Assert.allclose(actual["rms(c)"], expected.rmsc)


@pytest.mark.parametrize(
    "quantity_name", ["N", "E", "dE", "deps", "ncg", "rms", "rms(c)"]
)
def test_read_selection(quantity_name, ElectronicMinimization, Assert):
    actual = ElectronicMinimization.read(quantity_name)
    expected = getattr(
        ElectronicMinimization.ref, quantity_name.replace("(", "").replace(")", "")
    )
    Assert.allclose(actual[quantity_name], expected)


def test_read_incorrect_selection(ElectronicMinimization):
    with pytest.raises(exception.RefinementError):
        ElectronicMinimization.read("forces")


def test_slice(ElectronicMinimization, Assert):
    actual = ElectronicMinimization[0:1].read()
    expected = ElectronicMinimization.ref
    Assert.allclose(actual["N"], expected.N)
    Assert.allclose(actual["E"], expected.E)
    Assert.allclose(actual["dE"], expected.dE)
    Assert.allclose(actual["deps"], expected.deps)
    Assert.allclose(actual["ncg"], expected.ncg)
    Assert.allclose(actual["rms"], expected.rms)
    Assert.allclose(actual["rms(c)"], expected.rmsc)


def test_plot(ElectronicMinimization, Assert):
    graph = ElectronicMinimization.plot()
    assert graph.xlabel == "Iteration number"
    assert graph.ylabel == "E"
    assert len(graph.series) == 1
    Assert.allclose(graph.series[0].x, ElectronicMinimization.ref.N)
    Assert.allclose(graph.series[0].y, ElectronicMinimization.ref.E)


def test_print(ElectronicMinimization, format_):
    actual, _ = format_(ElectronicMinimization)
    assert actual["text/plain"] == ElectronicMinimization.ref.string_rep


def test_is_converged(ElectronicMinimization):
    actual = ElectronicMinimization.is_converged()
    expected = ElectronicMinimization.ref.is_elmin_converged
    assert actual == expected

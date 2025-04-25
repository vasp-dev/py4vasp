# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import types

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._calculation.electronic_minimization import ElectronicMinimization


@pytest.fixture
def electronic_minimization(raw_data):
    raw_elmin = raw_data.electronic_minimization()
    constructor = ElectronicMinimization.from_data
    electronic_minimization = ElectronicMinimization.from_data(raw_elmin)
    electronic_minimization.ref = types.SimpleNamespace()
    convergence_data = raw_elmin.convergence_data
    electronic_minimization.ref.N = np.int64(convergence_data[:, 0])
    electronic_minimization.ref.E = convergence_data[:, 1]
    electronic_minimization.ref.dE = convergence_data[:, 2]
    electronic_minimization.ref.deps = convergence_data[:, 3]
    electronic_minimization.ref.ncg = convergence_data[:, 4]
    electronic_minimization.ref.rms = convergence_data[:, 5]
    electronic_minimization.ref.rmsc = convergence_data[:, 6]
    is_elmin_converged = [raw_elmin.is_elmin_converged == [0.0]]
    electronic_minimization.ref.is_elmin_converged = is_elmin_converged
    string_rep = "N\t\tE\t\tdE\t\tdeps\t\tncg\trms\t\trms(c)\n"
    format_rep = "{0:g}\t{1:0.12E}\t{2:0.6E}\t{3:0.6E}\t{4:g}\t{5:0.3E}\t{6:0.3E}\n"
    for idx in range(len(convergence_data)):
        string_rep += format_rep.format(*convergence_data[idx])
    electronic_minimization.ref.string_rep = str(string_rep)
    return electronic_minimization


def test_read(electronic_minimization, Assert):
    actual = electronic_minimization.read()
    expected = electronic_minimization.ref
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
def test_read_selection(quantity_name, electronic_minimization, Assert):
    actual = electronic_minimization.read(quantity_name)
    name_without_parenthesis = quantity_name.replace("(", "").replace(")", "")
    expected = getattr(electronic_minimization.ref, name_without_parenthesis)
    Assert.allclose(actual[quantity_name], expected)


def test_read_incorrect_selection(electronic_minimization):
    with pytest.raises(exception.RefinementError):
        electronic_minimization.read("forces")


def test_slice(electronic_minimization, Assert):
    actual = electronic_minimization[0:1].read()
    expected = electronic_minimization.ref
    Assert.allclose(actual["N"], expected.N)
    Assert.allclose(actual["E"], expected.E)
    Assert.allclose(actual["dE"], expected.dE)
    Assert.allclose(actual["deps"], expected.deps)
    Assert.allclose(actual["ncg"], expected.ncg)
    Assert.allclose(actual["rms"], expected.rms)
    Assert.allclose(actual["rms(c)"], expected.rmsc)


def test_plot(electronic_minimization, Assert):
    graph = electronic_minimization.plot()
    assert graph.xlabel == "Iteration number"
    assert graph.ylabel == "E"
    assert len(graph.series) == 1
    Assert.allclose(graph.series[0].x, electronic_minimization.ref.N)
    Assert.allclose(graph.series[0].y, electronic_minimization.ref.E)


def test_print(electronic_minimization, format_):
    actual, _ = format_(electronic_minimization)
    assert actual["text/plain"] == electronic_minimization.ref.string_rep


def test_is_converged(electronic_minimization):
    actual = electronic_minimization.is_converged()
    expected = electronic_minimization.ref.is_elmin_converged
    assert actual == expected


# def test_factory_methods(raw_data, check_factory_methods):
#     data = raw_data.electronic_minimization()
#     check_factory_methods(ElectronicMinimization, data)

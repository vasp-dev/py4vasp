# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import types
from dataclasses import fields

import numpy as np
import pytest

from py4vasp import exception, raw
from py4vasp._calculation.electronic_minimization import (
    ElectronicMinimization,
    ElectronicMinimizationHandler,
)
from py4vasp._raw.data_db import ElectronicMinimization_DB


@pytest.fixture
def electronic_minimization(raw_data):
    raw_elmin = raw_data.electronic_minimization()
    electronic_minimization = ElectronicMinimization.from_data(raw_elmin)
    electronic_minimization.ref = types.SimpleNamespace()
    convergence_data = raw_elmin.convergence_data
    electronic_minimization.ref.N = np.int64(convergence_data[:, 0])
    electronic_minimization.ref.E = convergence_data[:, 1]
    electronic_minimization.ref.dE = convergence_data[:, 2]
    electronic_minimization.ref.deps = convergence_data[:, 3]
    electronic_minimization.ref.ncg = convergence_data[:, 4]
    electronic_minimization.ref.rms = convergence_data[:, 5]
    # read() nulls out the not-yet-computed (zero) rms(c) entries of the early steps
    rms_c = np.array(convergence_data[:, 6], dtype=float)
    rms_c[rms_c == 0.0] = np.nan
    electronic_minimization.ref.rms_c = rms_c
    is_elmin_converged = [raw_elmin.is_elmin_converged == [0.0]]
    electronic_minimization.ref.is_elmin_converged = is_elmin_converged
    string_rep = "N\t\tE\t\tdE\t\tdeps\t\tncg\trms\t\trms(c)\n"
    format_rep = "{0:g}\t{1:0.12E}\t{2:0.6E}\t{3:0.6E}\t{4:g}\t{5:0.3E}\t{6:0.3E}\n"
    for idx in range(len(convergence_data)):
        string_rep += format_rep.format(*convergence_data[idx])
    electronic_minimization.ref.string_rep = str(string_rep)
    electronic_minimization.ref.overview_data = {
        "num_electronic_steps": len(electronic_minimization.ref.N),
        "elmin_is_converged_all": all(is_elmin_converged),
        "elmin_is_converged_final": is_elmin_converged[-1],
        "num_max_electronic_steps_per_ionic_step": len(electronic_minimization.ref.N),
        "num_min_electronic_steps_per_ionic_step": len(electronic_minimization.ref.N),
    }
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
    Assert.allclose(actual["rms_c"], expected.rms_c)


@pytest.mark.parametrize(
    "quantity_name", ["N", "E", "dE", "deps", "ncg", "rms", "rms_c"]
)
def test_read_selection(quantity_name, electronic_minimization, Assert):
    actual = electronic_minimization.read(quantity_name)
    expected = getattr(electronic_minimization.ref, quantity_name)
    Assert.allclose(actual[quantity_name], expected)


def test_read_list(electronic_minimization, Assert):
    actual = electronic_minimization.read("E, dE")
    assert set(actual) == {"E", "dE"}
    Assert.allclose(actual["E"], electronic_minimization.ref.E)
    Assert.allclose(actual["dE"], electronic_minimization.ref.dE)


def test_read_addition(electronic_minimization, Assert):
    actual = electronic_minimization.read("E + dE")
    assert list(actual) == ["E + dE"]
    expected = electronic_minimization.ref.E + electronic_minimization.ref.dE
    Assert.allclose(actual["E + dE"], expected)


def test_read_nulls_early_rms_c(electronic_minimization, Assert):
    # the first (zero) rms(c) entries should be reported as NaN, real ones kept
    actual = electronic_minimization.read("rms_c")["rms_c"]
    assert np.all(np.isnan(actual[:5]))
    assert not np.any(np.isnan(actual[5:]))
    Assert.allclose(actual, electronic_minimization.ref.rms_c)


def test_sanity_check_applies_to_any_column(Assert):
    # values below the threshold are nulled regardless of which column they are in
    convergence_data = np.array(
        [
            [1, -8.0, 1e-30, 5e-1, 5, 3e-1, 0.0],
            [2, -8.1, 1e-3, 1e-25, 6, 1e-1, 0.0],
        ]
    )
    raw_elmin = raw.ElectronicMinimization(
        convergence_data=raw.VaspData(convergence_data),
        label=raw.VaspData([b"N", b"E", b"dE", b"deps", b"ncg", b"rms", b"rms(c)"]),
        is_elmin_converged=[0],
    )
    data = ElectronicMinimizationHandler.from_data(raw_elmin).to_dict()
    assert np.isnan(data["dE"][0]) and not np.isnan(data["dE"][1])
    assert not np.isnan(data["deps"][0]) and np.isnan(data["deps"][1])
    assert np.all(np.isnan(data["rms_c"]))
    assert not np.any(np.isnan(data["E"]))


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
    Assert.allclose(actual["rms_c"], expected.rms_c)


def test_plot(electronic_minimization, Assert):
    graph = electronic_minimization.plot()
    ref = electronic_minimization.ref
    assert graph.xlabel == "Iteration number"
    assert graph.ylabel == "Energy change (eV)"
    assert graph.y2label == "Residual"
    assert graph.yscale == "log"
    assert graph.y2scale == "log"
    assert len(graph.series) == 5
    by_label = {series.label: series for series in graph.series}
    for series in graph.series:
        Assert.allclose(series.x, ref.N)
    # energy changes on the (log) left axis
    energy_change = np.abs(ref.E[-1] - ref.E)
    energy_change[-1] = np.nan  # final point is zero and dropped on the log axis
    Assert.allclose(by_label["|E_final - E|"].y, energy_change)
    Assert.allclose(by_label["|dE|"].y, np.abs(ref.dE))
    Assert.allclose(by_label["|d eps|"].y, np.abs(ref.deps))
    assert not by_label["|E_final - E|"].y2
    # residuals on the (log) secondary axis
    Assert.allclose(by_label["rms"].y, ref.rms)
    Assert.allclose(by_label["rms_c"].y, ref.rms_c)
    assert by_label["rms"].y2
    assert by_label["rms_c"].y2


def test_plot_selection(electronic_minimization, Assert):
    graph = electronic_minimization.plot("E, rms")
    ref = electronic_minimization.ref
    assert graph.yscale == "log"
    assert {series.label for series in graph.series} == {"E", "rms"}
    by_label = {series.label: series for series in graph.series}
    Assert.allclose(by_label["E"].x, ref.N)
    Assert.allclose(by_label["E"].y, ref.E)
    Assert.allclose(by_label["rms"].y, ref.rms)


def test_print(electronic_minimization, format_):
    actual, _ = format_(electronic_minimization)
    assert actual["text/plain"] == electronic_minimization.ref.string_rep


def test_is_converged(electronic_minimization):
    actual = electronic_minimization.is_converged()
    expected = electronic_minimization.ref.is_elmin_converged
    assert actual == expected


def test_to_database(electronic_minimization, raw_data):
    raw_elmin = raw_data.electronic_minimization()
    handler = ElectronicMinimizationHandler.from_data(raw_elmin)
    database_data: ElectronicMinimization_DB = handler.to_database()
    overview_data = electronic_minimization.ref.overview_data

    assert isinstance(database_data, ElectronicMinimization_DB)

    for fld in fields(database_data):
        k = fld.name
        v = getattr(database_data, k)
        if k.startswith("__"):
            continue

        assert (
            v == overview_data[k]
        ), f"{k} has unexpected value {v}, expected {overview_data[k]}"
        if k.startswith("num"):
            assert isinstance(
                v, (int, type(None))
            ), f"{k} has unexpected type {type(v)}: {v}"
        elif k.startswith("elmin_is_converged"):
            assert isinstance(
                v, (bool, type(None))
            ), f"{k} has unexpected type {type(v)}: {v}"


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.electronic_minimization()
    check_factory_methods(ElectronicMinimization, data)

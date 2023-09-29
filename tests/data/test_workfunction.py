# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp import data


@pytest.fixture(params=[1, 2, 3])
def workfunction(raw_data, request):
    raw_workfunction = raw_data.workfunction(str(request.param))
    return setup_reference(raw_workfunction)


def setup_reference(raw_workfunction):
    workfunction = data.Workfunction.from_data(raw_workfunction)
    workfunction.ref = raw_workfunction
    raw_gap = raw_workfunction.reference_potential
    workfunction.ref.lattice_vector = f"lattice vector {raw_workfunction.idipol}"
    workfunction.ref.vbm = raw_gap.values[-1, 0, 0]
    workfunction.ref.cbm = raw_gap.values[-1, 0, 1]
    return workfunction


def test_read(workfunction, Assert):
    actual = workfunction.read()
    actual["direction"] == workfunction.ref.lattice_vector
    Assert.allclose(actual["distance"], workfunction.ref.distance)
    Assert.allclose(actual["average_potential"], workfunction.ref.average_potential)
    Assert.allclose(actual["vacuum_potential"], workfunction.ref.vacuum_potential)
    Assert.allclose(actual["valence_band_maximum"], workfunction.ref.vbm)
    Assert.allclose(actual["conduction_band_minimum"], workfunction.ref.cbm)
    Assert.allclose(actual["fermi_energy"], workfunction.ref.fermi_energy)


def test_plot(workfunction, Assert):
    graph = workfunction.plot()
    assert graph.xlabel == "distance (Å)"
    assert graph.ylabel == "average potential (eV)"
    Assert.allclose(graph.series.x, workfunction.ref.distance)
    Assert.allclose(graph.series.y, workfunction.ref.average_potential)
    assert graph.series.name == workfunction.ref.lattice_vector
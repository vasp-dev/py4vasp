# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from unittest.mock import patch

import numpy as np
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
    # Uncomment out these lines when vbm and cbm are added to VASP 6.5
    # Assert.allclose(actual["valence_band_maximum"], workfunction.ref.vbm)
    # Assert.allclose(actual["conduction_band_minimum"], workfunction.ref.cbm)
    Assert.allclose(actual["fermi_energy"], workfunction.ref.fermi_energy)


def test_plot(workfunction, Assert):
    graph = workfunction.plot()
    assert graph.xlabel == f"distance along {workfunction.ref.lattice_vector} (Å)"
    assert graph.ylabel == "average potential (eV)"
    Assert.allclose(graph.series.x, workfunction.ref.distance)
    Assert.allclose(graph.series.y, workfunction.ref.average_potential)
    assert graph.series.name == "potential"


@patch("py4vasp._data.workfunction.Workfunction.to_graph")
def test_to_plotly(mock_plot, workfunction):
    fig = workfunction.to_plotly()
    mock_plot.assert_called_once_with()
    graph = mock_plot.return_value
    graph.to_plotly.assert_called_once()
    assert fig == graph.to_plotly.return_value


def test_to_image(workfunction):
    check_to_image(workfunction, None, "workfunction.png")
    custom_filename = "custom.jpg"
    check_to_image(workfunction, custom_filename, custom_filename)


def check_to_image(workfunction, filename_argument, expected_filename):
    with patch("py4vasp._data.workfunction.Workfunction.to_plotly") as plot:
        workfunction.to_image("args", filename=filename_argument, key="word")
        plot.assert_called_once_with("args", key="word")
        fig = plot.return_value
        fig.write_image.assert_called_once_with(workfunction._path / expected_filename)


def test_print(workfunction, format_):
    actual, _ = format_(workfunction)
    reference = """\
workfunction along {lattice_vector}:
    vacuum potential: {vacuum1:.3f} {vacuum2:.3f}
    Fermi energy: {fermi_energy:.3f}"""
    reference = reference.format(
        lattice_vector=workfunction.ref.lattice_vector,
        vacuum1=workfunction.ref.vacuum_potential[0],
        vacuum2=workfunction.ref.vacuum_potential[1],
        fermi_energy=workfunction.ref.fermi_energy,
    )
    assert actual == {"text/plain": reference}
    # valence band maximum: {vbm:.3f}
    # conduction band minimum: {cbm:.3f}
    # vbm=workfunction.ref.vbm,
    # cbm=workfunction.ref.cbm,


def test_factory_methods(raw_data, check_factory_methods):
    raw_workfunction = raw_data.workfunction("1")
    check_factory_methods(data.Workfunction, raw_workfunction)

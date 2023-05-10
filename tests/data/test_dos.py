# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._util import select
from py4vasp.data import Dos


@pytest.fixture
def Sr2TiO4(raw_data):
    raw_dos = raw_data.dos("Sr2TiO4")
    dos = Dos.from_data(raw_dos)
    dos.ref = types.SimpleNamespace()
    dos.ref.energies = raw_dos.energies - raw_dos.fermi_energy
    dos.ref.dos = raw_dos.dos[0]
    dos.ref.fermi_energy = raw_dos.fermi_energy
    return dos


@pytest.fixture
def Fe3O4(raw_data):
    raw_dos = raw_data.dos("Fe3O4")
    dos = Dos.from_data(raw_dos)
    dos.ref = types.SimpleNamespace()
    dos.ref.energies = raw_dos.energies - raw_dos.fermi_energy
    dos.ref.dos_up = raw_dos.dos[0]
    dos.ref.dos_down = raw_dos.dos[1]
    dos.ref.fermi_energy = raw_dos.fermi_energy
    return dos


@pytest.fixture
def Sr2TiO4_projectors(raw_data):
    raw_dos = raw_data.dos("Sr2TiO4 with_projectors")
    dos = Dos.from_data(raw_dos)
    dos.ref = types.SimpleNamespace()
    dos.ref.s = np.sum(raw_dos.projections[0, :, 0, :], axis=0)
    dos.ref.Sr_p = np.sum(raw_dos.projections[0, 0:2, 1:4, :], axis=(0, 1))
    dos.ref.Sr_d = np.sum(raw_dos.projections[0, 0:2, 4:9, :], axis=(0, 1))
    dos.ref.Sr_2_p = np.sum(raw_dos.projections[0, 1, 1:4, :], axis=0)
    dos.ref.Ti = np.sum(raw_dos.projections[0, 2, :, :], axis=0)
    dos.ref.Ti_dz2 = raw_dos.projections[0, 2, 6, :]
    dos.ref.O_px = np.sum(raw_dos.projections[0, 3:7, 3, :], axis=0)
    dos.ref.O_dxy = np.sum(raw_dos.projections[0, 3:7, 4, :], axis=0)
    dos.ref.O_1 = np.sum(raw_dos.projections[0, 3, :, :], axis=0)
    return dos


@pytest.fixture
def Fe3O4_projectors(raw_data):
    raw_dos = raw_data.dos("Fe3O4 with_projectors")
    dos = Dos.from_data(raw_dos)
    dos.ref = types.SimpleNamespace()
    dos.ref.Fe_up = np.sum(raw_dos.projections[0, 0:3, :, :], axis=(0, 1))
    dos.ref.Fe_down = np.sum(raw_dos.projections[1, 0:3, :, :], axis=(0, 1))
    dos.ref.p_up = np.sum(raw_dos.projections[0, :, 1, :], axis=0)
    dos.ref.p_down = np.sum(raw_dos.projections[1, :, 1, :], axis=0)
    dos.ref.O_d_up = np.sum(raw_dos.projections[0, 3:7, 2, :], axis=0)
    dos.ref.O_d_down = np.sum(raw_dos.projections[1, 3:7, 2, :], axis=0)
    return dos


def test_Sr2TiO4_read(Sr2TiO4, Assert):
    actual = Sr2TiO4.read()
    Assert.allclose(actual["energies"], Sr2TiO4.ref.energies)
    Assert.allclose(actual["total"], Sr2TiO4.ref.dos)
    assert actual["fermi_energy"] == Sr2TiO4.ref.fermi_energy


def test_Fe3O4_read(Fe3O4, Assert):
    actual = Fe3O4.read()
    Assert.allclose(actual["energies"], Fe3O4.ref.energies)
    Assert.allclose(actual["up"], Fe3O4.ref.dos_up)
    Assert.allclose(actual["down"], Fe3O4.ref.dos_down)
    assert actual["fermi_energy"] == Fe3O4.ref.fermi_energy


def test_Fe3O4_projectors_read(Fe3O4_projectors, Assert):
    actual = Fe3O4_projectors.read("Fe p O(d)")
    Assert.allclose(actual["Fe_up"], Fe3O4_projectors.ref.Fe_up)
    Assert.allclose(actual["Fe_down"], Fe3O4_projectors.ref.Fe_down)
    Assert.allclose(actual["p_up"], Fe3O4_projectors.ref.p_up)
    Assert.allclose(actual["p_down"], Fe3O4_projectors.ref.p_down)
    Assert.allclose(actual["O_d_up"], Fe3O4_projectors.ref.O_d_up)
    Assert.allclose(actual["O_d_down"], Fe3O4_projectors.ref.O_d_down)


def test_read_missing_projectors(Sr2TiO4):
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4.read("s")


def test_read_excess_orbital_types(raw_data, Assert):
    """Vasp 6.1 may store more orbital types then projections available. This
    test checks that this does not lead to any issues when an available element
    is used."""
    dos = Dos.from_data(raw_data.dos("Fe3O4 excess_orbitals"))
    actual = dos.read("s p g")
    zero = np.zeros_like(actual["energies"])
    Assert.allclose(actual["g_up"], zero)
    Assert.allclose(actual["g_down"], zero)


def test_Sr2TiO4_to_frame(Sr2TiO4, Assert):
    actual = Sr2TiO4.to_frame()
    Assert.allclose(actual.energies, Sr2TiO4.ref.energies)
    Assert.allclose(actual.total, Sr2TiO4.ref.dos)
    assert actual.fermi_energy == Sr2TiO4.ref.fermi_energy


def test_Fe3O4_to_frame(Fe3O4, Assert):
    actual = Fe3O4.to_frame()
    Assert.allclose(actual.energies, Fe3O4.ref.energies)
    Assert.allclose(actual.up, Fe3O4.ref.dos_up)
    Assert.allclose(actual.down, Fe3O4.ref.dos_down)
    assert actual.fermi_energy == Fe3O4.ref.fermi_energy


def test_Sr2TiO4_projectors_to_frame(Sr2TiO4_projectors, Assert):
    equivalent_selections = [
        "s Sr(d) Ti O(px,dxy) 2(p) 4 3(dz2) 1:2(p)",
        "2( p), dz2(3) Sr(d) p(1:2), s, 4 Ti px(O) O(dxy)",
    ]
    for selection in equivalent_selections:
        actual = Sr2TiO4_projectors.to_frame(selection)
        Assert.allclose(actual.s, Sr2TiO4_projectors.ref.s)
        Assert.allclose(actual["1:2_p"], Sr2TiO4_projectors.ref.Sr_p)
        Assert.allclose(actual.Sr_d, Sr2TiO4_projectors.ref.Sr_d)
        Assert.allclose(actual.Sr_2_p, Sr2TiO4_projectors.ref.Sr_2_p)
        Assert.allclose(actual.Ti, Sr2TiO4_projectors.ref.Ti)
        Assert.allclose(actual.Ti_1_dz2, Sr2TiO4_projectors.ref.Ti_dz2)
        Assert.allclose(actual.O_px, Sr2TiO4_projectors.ref.O_px)
        Assert.allclose(actual.O_dxy, Sr2TiO4_projectors.ref.O_dxy)
        Assert.allclose(actual.O_1, Sr2TiO4_projectors.ref.O_1)


def test_Sr2TiO4_plot(Sr2TiO4, Assert):
    fig = Sr2TiO4.plot()
    assert fig.xlabel == "Energy (eV)"
    assert fig.ylabel == "DOS (1/eV)"
    assert len(fig.series) == 1
    Assert.allclose(fig.series[0].x, Sr2TiO4.ref.energies)
    Assert.allclose(fig.series[0].y, Sr2TiO4.ref.dos)


def test_Fe3O4_plot(Fe3O4, Assert):
    fig = Fe3O4.plot()
    assert len(fig.series) == 2
    Assert.allclose(fig.series[0].x, fig.series[1].x)
    Assert.allclose(fig.series[0].y, Fe3O4.ref.dos_up)
    Assert.allclose(fig.series[1].y, -Fe3O4.ref.dos_down)


def test_Sr2TiO4_projectors_plot(Sr2TiO4_projectors, Assert):
    fig = Sr2TiO4_projectors.plot("s O(px) dz2(3)")
    assert len(fig.series) == 4  # total Dos + 3 selections
    Assert.allclose(fig.series[1].y, Sr2TiO4_projectors.ref.s)
    Assert.allclose(fig.series[2].y, Sr2TiO4_projectors.ref.O_px)
    Assert.allclose(fig.series[3].y, Sr2TiO4_projectors.ref.Ti_dz2)


def test_Fe3O4_projectors_plot(Fe3O4_projectors, Assert):
    fig = Fe3O4_projectors.plot("Fe p O(d)")
    data = fig.series
    assert len(data) == 8  # spin resolved total + 3 selections
    names = [d.name for d in data]
    Fe_up = names.index("Fe_up")
    Assert.allclose(data[Fe_up].y, Fe3O4_projectors.ref.Fe_up)
    Fe_down = names.index("Fe_down")
    Assert.allclose(data[Fe_down].y, -Fe3O4_projectors.ref.Fe_down)
    p_up = names.index("p_up")
    Assert.allclose(data[p_up].y, Fe3O4_projectors.ref.p_up)
    p_down = names.index("p_down")
    Assert.allclose(data[p_down].y, -Fe3O4_projectors.ref.p_down)
    O_d_up = names.index("O_d_up")
    Assert.allclose(data[O_d_up].y, Fe3O4_projectors.ref.O_d_up)
    O_d_down = names.index("O_d_down")
    Assert.allclose(data[O_d_down].y, -Fe3O4_projectors.ref.O_d_down)


@patch("py4vasp._data.dos.Dos.to_graph")
def test_Sr2TiO4_to_plotly(mock_plot, Sr2TiO4):
    fig = Sr2TiO4.to_plotly("selection")
    mock_plot.assert_called_once_with("selection")
    graph = mock_plot.return_value
    graph.to_plotly.assert_called_once()
    assert fig == graph.to_plotly.return_value


def test_Sr2TiO4_to_image(Sr2TiO4):
    check_to_image(Sr2TiO4, None, "dos.png")
    custom_filename = "custom.jpg"
    check_to_image(Sr2TiO4, custom_filename, custom_filename)


def check_to_image(Sr2TiO4, filename_argument, expected_filename):
    with patch("py4vasp._data.dos.Dos.to_plotly") as plot:
        Sr2TiO4.to_image("args", filename=filename_argument, key="word")
        plot.assert_called_once_with("args", key="word")
        fig = plot.return_value
        fig.write_image.assert_called_once_with(Sr2TiO4._path / expected_filename)


def test_Sr2TiO4_print(Sr2TiO4, format_):
    actual, _ = format_(Sr2TiO4)
    reference = f"""
Dos:
    energies: [-1.00, 3.00] 50 points
no projectors
    """.strip()
    assert actual == {"text/plain": reference}


def test_Fe3O4_print(Fe3O4, format_):
    actual, _ = format_(Fe3O4)
    reference = f"""
spin polarized Dos:
    energies: [-2.00, 2.00] 50 points
no projectors
    """.strip()
    assert actual == {"text/plain": reference}


def test_Sr2TiO4_projectors_print(Sr2TiO4_projectors, format_):
    actual, _ = format_(Sr2TiO4_projectors)
    reference = f"""
Dos:
    energies: [-1.00, 3.00] 50 points
projectors:
    atoms: Sr, Ti, O
    orbitals: s, py, pz, px, dxy, dyz, dz2, dxz, dx2y2, fy3x2, fxyz, fyz2, fz3, fxz2, fzx2, fx3
    """.strip()
    assert actual == {"text/plain": reference}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.dos("Sr2TiO4")
    check_factory_methods(Dos, data)

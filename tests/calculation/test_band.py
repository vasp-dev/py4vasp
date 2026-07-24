# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types
from dataclasses import fields
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._calculation._dispersion import DispersionHandler
from py4vasp._calculation.band import _OCCUPATION_CUTOFF, Band, BandHandler
from py4vasp._calculation.kpoint import Kpoint
from py4vasp._calculation.projector import Projector
from py4vasp._raw.models import BandModel
from py4vasp._util import slicing


@pytest.fixture
def single_band(raw_data):
    raw_band = raw_data.band("single")
    band = Band.from_data(raw_band)
    band.ref = types.SimpleNamespace()
    band.ref.fermi_energy_argument = 1.23
    band.ref.fermi_energy = 0.0
    band.ref.bands = raw_band.dispersion.eigenvalues[0] - band.ref.fermi_energy_argument
    band.ref.occupations = raw_band.occupations[0]
    band.ref.num_occupied_bands = int(
        np.max(np.sum(band.ref.occupations > _OCCUPATION_CUTOFF, axis=-1))
    )
    band.ref.kpoints = Kpoint.from_data(raw_band.dispersion.kpoints)
    band.ref.raw_data = raw_band
    return band


@pytest.fixture
def multiple_bands(raw_data):
    raw_band = raw_data.band("multiple")
    band = Band.from_data(raw_band)
    band.ref = types.SimpleNamespace()
    band.ref.fermi_energy = raw_band.fermi_energy
    band.ref.bands = raw_band.dispersion.eigenvalues[0] - raw_band.fermi_energy
    band.ref.occupations = raw_band.occupations[0]
    band.ref.num_occupied_bands = int(
        np.max(np.sum(band.ref.occupations > _OCCUPATION_CUTOFF, axis=-1))
    )
    band.ref.kpoints = Kpoint.from_data(raw_band.dispersion.kpoints)
    band.ref.raw_data = raw_band
    return band


@pytest.fixture
def with_projectors(raw_data):
    raw_band = raw_data.band("multiple with_projectors")
    band = Band.from_data(raw_band)
    band.ref = types.SimpleNamespace()
    band.ref.bands = raw_band.dispersion.eigenvalues[0] - raw_band.fermi_energy
    band.ref.Sr = np.sum(raw_band.projections[0, 0:2, :, :, :], axis=(0, 1))
    band.ref.p = np.sum(raw_band.projections[0, :, 1:4, :, :], axis=(0, 1))
    band.ref.selections = Projector.from_data(raw_band.projectors).selections()
    return band


@pytest.fixture
def line_no_labels(raw_data):
    raw_band = raw_data.band("line no_labels")
    band = Band.from_data(raw_band)
    band.ref = types.SimpleNamespace()
    band.ref.kpoints = Kpoint.from_data(raw_band.dispersion.kpoints)
    return band


@pytest.fixture
def line_with_labels(raw_data):
    raw_band = raw_data.band("line with_labels")
    band = Band.from_data(raw_band)
    band.ref = types.SimpleNamespace()
    band.ref.kpoints = Kpoint.from_data(raw_band.dispersion.kpoints)
    return band


@pytest.fixture
def spin_polarized(raw_data):
    raw_band = raw_data.band("spin_polarized")
    band = Band.from_data(raw_band)
    band.ref = types.SimpleNamespace()
    assert raw_band.fermi_energy == 0
    band.ref.fermi_energy = raw_band.fermi_energy
    band.ref.bands_up = raw_band.dispersion.eigenvalues[0]
    band.ref.bands_down = raw_band.dispersion.eigenvalues[1]
    band.ref.occupations_up = raw_band.occupations[0]
    band.ref.num_occupied_bands_up = int(
        np.max(np.sum(band.ref.occupations_up > _OCCUPATION_CUTOFF, axis=-1))
    )
    band.ref.occupations_down = raw_band.occupations[1]
    band.ref.num_occupied_bands_down = int(
        np.max(np.sum(band.ref.occupations_down > _OCCUPATION_CUTOFF, axis=-1))
    )
    band.ref.raw_data = raw_band
    return band


@pytest.fixture
def spin_projectors(raw_data):
    raw_band = raw_data.band("spin_polarized with_projectors")
    band = Band.from_data(raw_band)
    band.ref = types.SimpleNamespace()
    band.ref.bands_up = raw_band.dispersion.eigenvalues[0]
    band.ref.bands_down = raw_band.dispersion.eigenvalues[1]
    band.ref.s_up = np.sum(raw_band.projections[0, :, 0, :, :], axis=0)
    band.ref.s_down = np.sum(raw_band.projections[1, :, 0, :, :], axis=0)
    band.ref.Fe_d_up = np.sum(raw_band.projections[0, 0:3, 2, :, :], axis=0)
    band.ref.Fe_d_down = np.sum(raw_band.projections[1, 0:3, 2, :, :], axis=0)
    band.ref.O_up = np.sum(raw_band.projections[0, 3:7, :, :, :], axis=(0, 1))
    band.ref.O_down = np.sum(raw_band.projections[1, 3:7, :, :, :], axis=(0, 1))
    projector = Projector.from_data(raw_band.projectors)
    band.ref.projectors_string = str(projector)
    return band


@pytest.fixture
def noncollinear_projectors(raw_data):
    raw_band = raw_data.band("noncollinear with_projectors")
    band = Band.from_data(raw_band)
    band.ref = types.SimpleNamespace()
    band.ref.total = np.sum(raw_band.projections[0], axis=(0, 1))
    band.ref.sigma_x = np.sum(raw_band.projections[1], axis=(0, 1))
    band.ref.Ba_sigma_y = np.sum(raw_band.projections[2, 0:2], axis=(0, 1))
    band.ref.d_sigma_z = np.sum(raw_band.projections[3, :, 2], axis=0)
    band.ref.occupations = raw_band.occupations[0]
    band.ref.num_occupied_bands = int(
        np.max(np.sum(band.ref.occupations > _OCCUPATION_CUTOFF, axis=-1))
    )
    band.ref.fermi_energy = raw_band.fermi_energy
    band.ref.raw_data = raw_band
    return band


@pytest.fixture(params=["x~y", "x~z", "y~z"])
def spin_texture(raw_data, request):
    raw_band = raw_data.band(f"spin_texture {request.param}")
    band = Band.from_data(raw_band)
    band.ref = types.SimpleNamespace()
    project_all_xy = np.sum(raw_band.projections[1:3, ..., 1], axis=(1, 2))
    project_Pb_xy = np.sum(raw_band.projections[1:3, 2, :, :, 0], axis=1)
    project_d_yz = np.sum(raw_band.projections[2:4, :, 2, :, 2], axis=1)
    project_5_p_zx = raw_band.projections[[1, 3], 4, 1, :, 0]
    cut = {"x~y": "c", "x~z": "b", "y~z": "a"}[request.param]
    plot_plane = _spin_texture_plane(raw_band, cut)
    band.ref.expected_data = {
        "sigma_x~sigma_y_band=2": _expected_quiver_data(
            project_all_xy, (0, 1), plot_plane
        ),
        "Pb_sigma_1~sigma_2_band=1": _expected_quiver_data(
            project_Pb_xy, (0, 1), plot_plane
        ),
        "d_y~z_band=3": _expected_quiver_data(project_d_yz, (1, 2), plot_plane),
        "O_2_p_x~z_band=1": _expected_quiver_data(project_5_p_zx, (0, 2), plot_plane),
    }
    band.ref.expected_lattice = expected_lattice(request.param)
    return band


def _spin_texture_plane(raw_band, cut):
    reciprocal_cell = Kpoint.from_data(
        raw_band.dispersion.kpoints
    )._reciprocal_lattice_vectors()
    return slicing.plane(reciprocal_cell, cut, normal=None)


def _expected_quiver_data(two_component_data, axes, plot_plane):
    """Embed 2-component spin data into 3D, reshape grid, and project onto plane."""
    nkp1, nkp2 = 4, 3
    embedded = np.zeros(
        (3, two_component_data.shape[-1]), dtype=two_component_data.dtype
    )
    embedded[list(axes)] = two_component_data
    embedded = embedded.reshape(3, nkp2, nkp1).transpose(0, 2, 1)
    return slicing._project_vectors_to_plane(plot_plane, embedded)


@pytest.fixture
def spin_texture_xy(raw_data):
    raw_band = raw_data.band("spin_texture x~y")
    band = Band.from_data(raw_band)
    band.ref = types.SimpleNamespace()
    band.ref.expected_lattice = expected_lattice("x~y")
    return band


@pytest.fixture
def asymmetric_spin_texture():
    """15x5x1 mesh where sigma_x increases along kx and sigma_y along ky."""
    from py4vasp import _demo, raw

    nkpx, nkpy, nkpz = 15, 5, 1
    num_kpoints = nkpx * nkpy
    coordinates = np.array(
        [
            (kx, ky, 0.0)
            for ky in np.linspace(0, 1, nkpy, endpoint=False)
            for kx in np.linspace(0, 1, nkpx, endpoint=False)
        ]
    )
    cell = raw.Cell(np.diag([5.0, 5.0, 10.0]).astype(float), scale=raw.VaspData(1.0))
    kpoints = raw.Kpoint(
        mode="explicit",
        number=num_kpoints,
        number_x=nkpx,
        number_y=nkpy,
        number_z=nkpz,
        coordinates=coordinates,
        weights=np.ones(num_kpoints),
        cell=cell,
    )
    dispersion = raw.Dispersion(kpoints, np.zeros((4, num_kpoints, 1)))
    projectors = _demo.projector.Ba2PbO4(use_orbitals=True)
    num_orbitals = len(projectors.orbital_types)
    projections = np.zeros((4, _demo.NUMBER_ATOMS, num_orbitals, num_kpoints, 1))
    kx_index = np.tile(np.arange(nkpx), nkpy)
    ky_index = np.repeat(np.arange(nkpy), nkpx)
    projections[1, :, :, :, 0] = kx_index[None, None, :]
    projections[2, :, :, :, 0] = ky_index[None, None, :]
    raw_band = raw.Band(
        dispersion=dispersion,
        fermi_energy=0.0,
        occupations=np.ones((4, num_kpoints, 1)),
        projectors=projectors,
        projections=projections,
    )
    band = Band.from_data(raw_band)
    band.ref = types.SimpleNamespace()
    scale = _demo.NUMBER_ATOMS * num_orbitals
    embedded = np.zeros((3, num_kpoints))
    embedded[0] = kx_index.astype(float) * scale
    embedded[1] = ky_index.astype(float) * scale
    embedded = embedded.reshape(3, nkpy, nkpx).transpose(0, 2, 1)
    reciprocal_cell = Kpoint.from_data(kpoints)._reciprocal_lattice_vectors()
    plot_plane = slicing.plane(reciprocal_cell, "c", normal=None)
    band.ref.expected_data = slicing._project_vectors_to_plane(plot_plane, embedded)
    return band


@pytest.fixture
def rotated_spin_texture():
    """4x3x1 mesh with a 45-degree rotated cell and uniform sigma_x = 1."""
    from py4vasp import _demo, raw

    nkpx, nkpy, nkpz = 4, 3, 1
    num_kpoints = nkpx * nkpy
    coordinates = np.array(
        [
            (kx, ky, 0.0)
            for ky in np.linspace(0, 1, nkpy, endpoint=False)
            for kx in np.linspace(0, 1, nkpx, endpoint=False)
        ]
    )
    s = 5.0 / np.sqrt(2)
    lattice = np.array([[s, s, 0.0], [-s, s, 0.0], [0.0, 0.0, 10.0]])
    cell = raw.Cell(lattice, scale=raw.VaspData(1.0))
    kpoints = raw.Kpoint(
        mode="explicit",
        number=num_kpoints,
        number_x=nkpx,
        number_y=nkpy,
        number_z=nkpz,
        coordinates=coordinates,
        weights=np.ones(num_kpoints),
        cell=cell,
    )
    dispersion = raw.Dispersion(kpoints, np.zeros((4, num_kpoints, 1)))
    projectors = _demo.projector.Ba2PbO4(use_orbitals=True)
    num_orbitals = len(projectors.orbital_types)
    projections = np.zeros((4, _demo.NUMBER_ATOMS, num_orbitals, num_kpoints, 1))
    projections[1, :, :, :, 0] = 1.0  # uniform sigma_x
    raw_band = raw.Band(
        dispersion=dispersion,
        fermi_energy=0.0,
        occupations=np.ones((4, num_kpoints, 1)),
        projectors=projectors,
        projections=projections,
    )
    band = Band.from_data(raw_band)
    band.ref = types.SimpleNamespace()
    scale = _demo.NUMBER_ATOMS * num_orbitals
    embedded = np.zeros((3, num_kpoints))
    embedded[0] = scale
    embedded = embedded.reshape(3, nkpy, nkpx).transpose(0, 2, 1)
    reciprocal_cell = Kpoint.from_data(kpoints)._reciprocal_lattice_vectors()
    plot_plane = slicing.plane(reciprocal_cell, "c", normal=None)
    band.ref.expected_data = slicing._project_vectors_to_plane(plot_plane, embedded)
    return band


def expected_lattice(selection):
    if selection == "x~y":
        return np.array([[1.52216787, 0.0], [0.14521927, 1.51522486]])
    else:
        return np.array([[1.52216787, 0.0], [0.29043854, 0.89433656]])


def test_single_band_read(single_band, Assert):
    band = single_band.read(fermi_energy=single_band.ref.fermi_energy_argument)
    assert band["fermi_energy"] == single_band.ref.fermi_energy
    Assert.allclose(band["bands"], single_band.ref.bands)
    Assert.allclose(band["occupations"], single_band.ref.occupations)
    Assert.allclose(band["kpoint_distances"], single_band.ref.kpoints.distances())
    assert "kpoint_labels" not in band
    assert "projections" not in band


def test_multiple_bands_read(multiple_bands, Assert):
    band = multiple_bands.read()
    assert band["fermi_energy"] == multiple_bands.ref.fermi_energy
    Assert.allclose(band["bands"], multiple_bands.ref.bands)
    Assert.allclose(band["occupations"], multiple_bands.ref.occupations)


def test_with_projectors_read(with_projectors, Assert):
    band = with_projectors.read("Sr p")
    Assert.allclose(band["Sr"], with_projectors.ref.Sr)
    Assert.allclose(band["p"], with_projectors.ref.p)


def test_line_with_labels_read(line_with_labels, Assert):
    band = line_with_labels.read()
    Assert.allclose(band["kpoint_distances"], line_with_labels.ref.kpoints.distances())
    assert band["kpoint_labels"] == line_with_labels.ref.kpoints.labels()


def test_spin_polarized_read(spin_polarized, Assert):
    band = spin_polarized.read()
    Assert.allclose(band["bands_up"], spin_polarized.ref.bands_up)
    Assert.allclose(band["bands_down"], spin_polarized.ref.bands_down)
    Assert.allclose(band["occupations_up"], spin_polarized.ref.occupations_up)
    Assert.allclose(band["occupations_down"], spin_polarized.ref.occupations_down)


def test_spin_projectors_read(spin_projectors, Assert):
    band = spin_projectors.read(selection="s Fe(d)")
    Assert.allclose(band["s_up"], spin_projectors.ref.s_up)
    Assert.allclose(band["s_down"], spin_projectors.ref.s_down)
    Assert.allclose(band["Fe_d_up"], spin_projectors.ref.Fe_d_up)
    Assert.allclose(band["Fe_d_down"], spin_projectors.ref.Fe_d_down)


def test_noncollinear_projectors_read(noncollinear_projectors, Assert):
    band = noncollinear_projectors.read(selection="total sigma_x Ba(y) d(sigma_3)")
    Assert.allclose(band["total"], noncollinear_projectors.ref.total)
    Assert.allclose(band["sigma_x"], noncollinear_projectors.ref.sigma_x)
    Assert.allclose(band["Ba_y"], noncollinear_projectors.ref.Ba_sigma_y)
    Assert.allclose(band["d_sigma_3"], noncollinear_projectors.ref.d_sigma_z)


def test_combining_projections(with_projectors, Assert):
    band = with_projectors.read("Sr + p, Sr - p")
    addition = with_projectors.ref.Sr + with_projectors.ref.p
    subtraction = with_projectors.ref.Sr - with_projectors.ref.p
    Assert.allclose(band["Sr + p"], addition)
    Assert.allclose(band["Sr - p"], subtraction)


def test_more_projections_style(raw_data, Assert):
    """Vasp 6.1 may store more orbital types then projections available. This
    test checks that this does not lead to any issues when an available element
    is used."""
    raw_band = raw_data.band("spin_polarized excess_orbitals")
    band = Band.from_data(raw_band).read("Fe g")
    zero = np.zeros_like(band["Fe_up"])
    Assert.allclose(band["g_up"], zero)
    Assert.allclose(band["g_down"], zero)


def test_single_polarized_to_frame(single_band, Assert):
    pytest.importorskip("pandas")
    actual = single_band.to_frame(fermi_energy=single_band.ref.fermi_energy_argument)
    Assert.allclose(actual.bands, single_band.ref.bands[:, 0])
    Assert.allclose(actual.occupations, single_band.ref.occupations[:, 0])
    Assert.allclose(actual.kpoint_distances, single_band.ref.kpoints.distances())


def test_multiple_bands_to_frame(multiple_bands, Assert):
    pytest.importorskip("pandas")
    actual = multiple_bands.to_frame()
    Assert.allclose(actual.bands, multiple_bands.ref.bands.T.flatten())
    Assert.allclose(actual.occupations, multiple_bands.ref.occupations.T.flatten())
    kpoint_distances = np.repeat(multiple_bands.ref.kpoints.distances(), repeats=3)
    Assert.allclose(actual.kpoint_distances, kpoint_distances)


def test_line_with_labels_to_frame(line_with_labels, Assert):
    pytest.importorskip("pandas")
    actual = line_with_labels.to_frame()
    kpoint_distances = np.repeat(line_with_labels.ref.kpoints.distances(), repeats=3)
    kpoint_labels = np.repeat(line_with_labels.ref.kpoints.labels(), repeats=3)
    actual_kpoint_labels = np.array(actual.kpoint_labels).astype(np.str_)
    Assert.allclose(actual.kpoint_distances, kpoint_distances)
    Assert.allclose(actual_kpoint_labels, kpoint_labels)


def test_with_projectors_to_frame(with_projectors, Assert):
    pytest.importorskip("pandas")
    actual = with_projectors.to_frame("Sr p")
    Assert.allclose(actual.Sr, with_projectors.ref.Sr.T.flatten())
    Assert.allclose(actual.p, with_projectors.ref.p.T.flatten())


def test_spin_polarized_to_frame(spin_polarized, Assert):
    pytest.importorskip("pandas")
    actual = spin_polarized.to_frame()
    ref = spin_polarized.ref
    Assert.allclose(actual.bands_up, ref.bands_up.T.flatten())
    Assert.allclose(actual.bands_down, ref.bands_down.T.flatten())
    Assert.allclose(actual.occupations_up, ref.occupations_up.T.flatten())
    Assert.allclose(actual.occupations_down, ref.occupations_down.T.flatten())


def test_spin_projectors_to_frame(spin_projectors, Assert):
    pytest.importorskip("pandas")
    actual = spin_projectors.to_frame(selection="O Fe(d)")
    Assert.allclose(actual.O_up, spin_projectors.ref.O_up.T.flatten())
    Assert.allclose(actual.O_down, spin_projectors.ref.O_down.T.flatten())
    Assert.allclose(actual.Fe_d_up, spin_projectors.ref.Fe_d_up.T.flatten())
    Assert.allclose(actual.Fe_d_down, spin_projectors.ref.Fe_d_down.T.flatten())


def test_single_band_plot(single_band, Assert):
    fig = single_band.plot(fermi_energy=single_band.ref.fermi_energy_argument)
    assert fig.ylabel == "Energy (eV)"
    assert len(fig.series) == 1
    assert fig.series[0].weight is None
    Assert.allclose(fig.series[0].x, single_band.ref.kpoints.distances())
    Assert.allclose(fig.series[0].y, single_band.ref.bands.T)


def test_multiple_bands_plot(multiple_bands, Assert):
    fig = multiple_bands.plot()
    assert len(fig.series) == 1  # all bands in one plot
    assert len(fig.series[0].x) == fig.series[0].y.shape[-1]
    Assert.allclose(fig.series[0].y, multiple_bands.ref.bands.T)


def test_with_projectors_plot_default_width(with_projectors, Assert):
    default_width = 0.5
    fig = with_projectors.plot(selection="Sr, p")
    check_figure(fig, default_width, with_projectors.ref, Assert)


def test_with_projectors_plot_custom_width(with_projectors, Assert):
    width = 0.1
    fig = with_projectors.plot(selection="Sr, p", width=width)
    check_figure(fig, width, with_projectors.ref, Assert)


def test_spin_projectors_plot(spin_projectors, Assert):
    reference = spin_projectors.ref
    width = 0.05
    fig = spin_projectors.plot("O", width=width)
    assert len(fig.series) == 2
    assert fig.series[0].label == "O_up"
    check_data(fig.series[0], width, reference.bands_up, reference.O_up, Assert)
    assert fig.series[1].label == "O_down"
    check_data(fig.series[1], width, reference.bands_down, reference.O_down, Assert)


def check_figure(fig, weight, reference, Assert):
    assert len(fig.series) == 2
    assert fig.series[0].label == "Sr"
    assert fig.series[1].label == "p"
    check_data(fig.series[0], weight, reference.bands, reference.Sr, Assert)
    check_data(fig.series[1], weight, reference.bands, reference.p, Assert)


def check_data(series, weight, band, projection, Assert):
    assert len(series.x) == series.y.shape[-1]
    assert series.y.shape == series.weight.shape
    Assert.allclose(series.y, band.T)
    Assert.allclose(series.weight, weight * projection.T)


def test_spin_polarized_plot(spin_polarized, Assert):
    fig = spin_polarized.plot()
    assert len(fig.series) == 2
    assert fig.series[0].label == "up"
    Assert.allclose(fig.series[0].y, spin_polarized.ref.bands_up.T)
    assert fig.series[1].label == "down"
    Assert.allclose(fig.series[1].y, spin_polarized.ref.bands_down.T)


def test_line_no_labels_plot(line_no_labels, Assert):
    fig = line_no_labels.plot()
    check_ticks(fig, line_no_labels.ref.kpoints, Assert)
    reference_labels = (
        "$[0 0 0]$",
        "$[0 0 \\frac{1}{2}]$",
        "$[\\frac{1}{2} \\frac{1}{2} \\frac{1}{2}]$|$[0 0 0]$",
        "$[\\frac{1}{2} \\frac{1}{2} 0]$",
        "$[\\frac{1}{2} \\frac{1}{2} \\frac{1}{2}]$",
    )
    assert tuple(fig.xticks.values()) == reference_labels


def test_line_with_labels_plot(line_with_labels, Assert):
    fig = line_with_labels.plot()
    check_ticks(fig, line_with_labels.ref.kpoints, Assert)
    assert tuple(fig.xticks.values()) == (r"$\Gamma$", "", r"M|$\Gamma$", "Y", "M")


def test_noncollinear_plot(noncollinear_projectors, Assert):
    default_width = 0.5
    fig = noncollinear_projectors.plot("total sigma_x")
    assert len(fig.series) == 2
    assert fig.series[0].label == "total"
    reference = noncollinear_projectors.ref
    Assert.allclose(fig.series[0].weight, default_width * reference.total.T)
    assert fig.series[0].marker is None
    assert fig.series[1].label == "sigma_x"
    Assert.allclose(fig.series[1].weight, reference.sigma_x.T)
    assert fig.series[1].marker == "o"
    assert fig.series[1].weight_mode == "color"


def check_ticks(fig, kpoints, Assert):
    dists = kpoints.distances()
    xticks = (*dists[:: kpoints.line_length()], dists[-1])
    Assert.allclose(list(fig.xticks.keys()), np.array(xticks))


def test_plot_incorrect_width(with_projectors):
    with pytest.raises(exception.IncorrectUsage):
        with_projectors.plot("Sr", width="not a number")


@patch.object(Band, "to_graph")
def test_to_plotly(mock_plot, single_band):
    fig = single_band.to_plotly("selection", width=0.2)
    mock_plot.assert_called_once_with("selection", width=0.2)
    graph = mock_plot.return_value
    graph.to_plotly.assert_called_once()
    assert fig == graph.to_plotly.return_value


def test_to_image(single_band):
    check_to_image(single_band, None, "band.png")
    custom_filename = "custom.jpg"
    check_to_image(single_band, custom_filename, custom_filename)


def check_to_image(single_band, filename_argument, expected_filename):
    with patch.object(Band, "to_plotly") as plot:
        single_band.to_image("args", filename=filename_argument, key="word")
        plot.assert_called_once_with("args", key="word")
        fig = plot.return_value
        fig.write_image.assert_called_once_with(single_band._path / expected_filename)


def test_band_selections(with_projectors):
    actual = with_projectors.selections()
    actual.pop("band")  # remove band selections
    assert actual == with_projectors.ref.selections


def test_to_quiver_with_incorrect_selection_raises_error(spin_texture_xy):
    with pytest.raises(exception.IncorrectUsage):
        spin_texture_xy.to_quiver("x")
    with pytest.raises(exception.IncorrectUsage):
        spin_texture_xy.to_quiver("x~y")
    with pytest.raises(exception.IncorrectUsage):
        spin_texture_xy.to_quiver("band=2")


@pytest.mark.parametrize(
    "selection",
    [
        "band=2(sigma_x~sigma_y)",
        "Pb(sigma_1~sigma_2(band=1))",
        "band=3(d(y~z))",
        "p(5(band=1(z~x)))",
    ],
)
def test_band_to_quiver(spin_texture, selection, Assert):
    graph = spin_texture.to_quiver(selection)
    assert graph.title == "Spin Texture"
    assert len(graph) == 1
    series = graph.series[0]
    Assert.allclose(
        series.lattice.vectors, spin_texture.ref.expected_lattice, tolerance=1e6
    )
    assert series.label in spin_texture.ref.expected_data
    Assert.allclose(series.data, spin_texture.ref.expected_data[series.label])


@pytest.mark.parametrize("normal", [None, "x", "y", "z"])
def test_band_to_quiver_normal(spin_texture_xy, normal, Assert):
    graph = spin_texture_xy.to_quiver("band=1(x~z)", normal=normal)
    assert len(graph) == 1
    quiver_plot = graph.series[0]
    if normal == "x":
        expected_lattice = [[-0.32989453, 1.48598944], [1.44773855, 0.47015754]]
    elif normal == "y":
        expected_lattice = [[1.44773855, 0.47015754], [-0.32989453, 1.48598944]]
    elif normal == "z":
        expected_lattice = [[1.52043113, 0.07269258], [0.07269258, 1.52043113]]
    else:
        expected_lattice = spin_texture_xy.ref.expected_lattice
    Assert.allclose(quiver_plot.lattice.vectors, expected_lattice, tolerance=1e6)


@pytest.mark.parametrize(
    "supercell, expected_supercell", [(None, (1, 1)), (2, (2, 2)), ([2, 4], (2, 4))]
)
def test_band_to_quiver_supercell(
    spin_texture_xy, supercell, expected_supercell, Assert
):
    graph = spin_texture_xy.to_quiver("band=3(sigma_2~sigma_1)", supercell=supercell)
    assert len(graph) == 1
    quiver_plot = graph.series[0]
    Assert.allclose(quiver_plot.supercell, expected_supercell)


def test_band_to_quiver_asymmetric_mesh(asymmetric_spin_texture, Assert):
    """With a heavily asymmetric k-mesh (15x5x1), verify correct axis ordering."""
    graph = asymmetric_spin_texture.to_quiver("x~y(band=1)")
    quiver_data = graph.series[0].data
    assert quiver_data.shape[1] == 15
    assert quiver_data.shape[2] == 5
    Assert.allclose(quiver_data, asymmetric_spin_texture.ref.expected_data)


def test_band_to_quiver_rotation_projects_spin_vectors(rotated_spin_texture, Assert):
    """Verify that spin vectors are rotated into the plot frame."""
    graph = rotated_spin_texture.to_quiver("x~y(band=1)")
    quiver_data = graph.series[0].data
    assert quiver_data.shape == (2, 4, 3)
    # A pure Cartesian x-vector should project onto both plot axes for a rotated cell
    assert not np.allclose(quiver_data[1], 0), (
        "Rotation is not applied: sigma_x should project onto both plot axes "
        "for a 45-degree rotated cell"
    )
    Assert.allclose(quiver_data, rotated_spin_texture.ref.expected_data)


def test_multiple_bands_print(multiple_bands, format_):
    actual, _ = format_(multiple_bands)
    reference = f"""
band data:
    48 k-points
    3 bands
no projectors
    """.strip()
    assert actual == {"text/plain": reference}


def test_line_no_labels_print(line_no_labels, format_):
    actual, _ = format_(line_no_labels)
    reference = f"""
band data:
    20 k-points
    3 bands
no projectors
    """.strip()
    assert actual == {"text/plain": reference}


def test_line_with_labels_print(line_with_labels, format_):
    actual, _ = format_(line_with_labels)
    reference = f"""
band data:
    20 k-points
    3 bands
no projectors
    """.strip()
    assert actual == {"text/plain": reference}


def test_spin_projectors_print(spin_projectors, format_):
    actual, _ = format_(spin_projectors)
    reference = f"""
spin polarized band data:
    48 k-points
    3 bands
{spin_projectors.ref.projectors_string}
    """.strip()
    assert actual == {"text/plain": reference}


def _check_to_database(_band):
    handler = BandHandler.from_data(_band.ref.raw_data)
    database_data: BandModel = handler.to_database(
        fermi_energy=getattr(_band.ref, "fermi_energy_argument", None)
    )

    assert isinstance(database_data, BandModel)

    assert database_data.fermi_energy_raw == _band.ref.fermi_energy
    assert database_data.fermi_energy == getattr(
        _band.ref, "fermi_energy_argument", _band.ref.fermi_energy
    )

    # dispersion is folded into the band model
    dispersion = DispersionHandler.from_data(
        _band.ref.raw_data.dispersion
    ).to_database()
    assert database_data.eigenvalue_min == dispersion.eigenvalue_min
    assert database_data.eigenvalue_max == dispersion.eigenvalue_max
    assert database_data.eigenvalue_min_up == dispersion.eigenvalue_min_up
    assert database_data.eigenvalue_max_up == dispersion.eigenvalue_max_up
    assert database_data.eigenvalue_min_down == dispersion.eigenvalue_min_down
    assert database_data.eigenvalue_max_down == dispersion.eigenvalue_max_down

    if getattr(_band.ref, "num_occupied_bands", None) is not None:
        assert database_data.num_occupied_bands == _band.ref.num_occupied_bands
    elif (
        getattr(_band.ref, "occupations_up", None) is not None
        and getattr(_band.ref, "occupations_down", None) is not None
    ):
        assert database_data.num_occupied_bands_up == _band.ref.num_occupied_bands_up
        assert (
            database_data.num_occupied_bands_down == _band.ref.num_occupied_bands_down
        )

    for fld in fields(BandModel):
        if fld.name.startswith("num"):
            assert getattr(database_data, fld.name) is None or isinstance(
                getattr(database_data, fld.name), int
            ), f"{fld.name} has unexpected type {type(getattr(database_data, fld.name))}: {getattr(database_data, fld.name)}"
        else:
            assert (
                getattr(database_data, fld.name) is None
                or isinstance(getattr(database_data, fld.name), float)
                or (
                    fld.name.startswith("__")
                    and isinstance(getattr(database_data, fld.name), str)
                )
            ), f"{fld.name} has unexpected type {type(getattr(database_data, fld.name))}: {getattr(database_data, fld.name)}"


def test_to_database_single_band(single_band):
    _check_to_database(single_band)


def test_to_database_multiple_bands(multiple_bands):
    _check_to_database(multiple_bands)


def test_to_database_spin_polarized(spin_polarized):
    _check_to_database(spin_polarized)


def test_to_database_noncollinear_projectors(noncollinear_projectors):
    _check_to_database(noncollinear_projectors)


def test_dispatcher_to_database_default(single_band):
    """Dispatcher._to_database() returns {quantity: {selection: handler_result}}."""
    result = single_band._to_database()
    assert isinstance(result, dict)
    assert "band" in result
    assert isinstance(result["band"], dict)
    assert isinstance(result["band"]["default"], BandModel)


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.band("multiple")
    parameters = {"to_quiver": {"selection": "x~y(band=1)"}}
    check_factory_methods(Band, data, parameters)


def test_is_available_to_quiver(raw_data):
    noncollinear = Band.from_data(raw_data.band("noncollinear with_projectors"))
    collinear = Band.from_data(raw_data.band("spin_polarized with_projectors"))
    without_projections = Band.from_data(raw_data.band("noncollinear"))
    # to_quiver needs the optional projections and a noncollinear calculation
    assert noncollinear.is_available(method="to_quiver") is True
    assert collinear.is_available(method="to_quiver") is False
    assert without_projections.is_available(method="to_quiver") is False
    # read stays available in every case
    assert noncollinear.is_available() is True

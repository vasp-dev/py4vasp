from py4vasp.data import Band, Kpoints, Projectors
from IPython.lib.pretty import pretty
from unittest.mock import patch
import py4vasp.exceptions as exception
import pytest
import numpy as np
import types


@pytest.fixture
def single_band(raw_data):
    raw_band = raw_data.band("single")
    band = Band(raw_band)
    band.ref = types.SimpleNamespace()
    band.ref.fermi_energy = 0.0
    band.ref.bands = raw_band.eigenvalues[0]
    band.ref.occupations = raw_band.occupations[0]
    raw_kpoints = raw_band.kpoints
    band.ref.kpoints = Kpoints(raw_kpoints)
    formatter = {"float": lambda x: f"{x:.2f}"}
    kpoint_to_string = lambda vec: np.array2string(vec, formatter=formatter) + " 1"
    band.ref.index = [kpoint_to_string(kpoint) for kpoint in raw_kpoints.coordinates]
    return band


@pytest.fixture
def multiple_bands(raw_data):
    raw_band = raw_data.band("multiple")
    band = Band(raw_band)
    band.ref = types.SimpleNamespace()
    band.ref.fermi_energy = raw_band.fermi_energy
    band.ref.bands = raw_band.eigenvalues[0] - raw_band.fermi_energy
    band.ref.occupations = raw_band.occupations[0]
    return band


@pytest.fixture
def with_projectors(raw_data):
    raw_band = raw_data.band("multiple with_projectors")
    band = Band(raw_band)
    band.ref = types.SimpleNamespace()
    band.ref.bands = raw_band.eigenvalues[0] - raw_band.fermi_energy
    band.ref.Sr = np.sum(raw_band.projections[0, 0:2, :, :, :], axis=(0, 1))
    band.ref.p = np.sum(raw_band.projections[0, :, 1:4, :, :], axis=(0, 1))
    return band


@pytest.fixture
def line_no_labels(raw_data):
    raw_band = raw_data.band("line no_labels")
    band = Band(raw_band)
    band.ref = types.SimpleNamespace()
    band.ref.kpoints = Kpoints(raw_band.kpoints)
    return band


@pytest.fixture
def line_with_labels(raw_data):
    raw_band = raw_data.band("line with_labels")
    band = Band(raw_band)
    band.ref = types.SimpleNamespace()
    band.ref.kpoints = Kpoints(raw_band.kpoints)
    return band


@pytest.fixture
def spin_polarized(raw_data):
    raw_band = raw_data.band("spin_polarized")
    band = Band(raw_band)
    band.ref = types.SimpleNamespace()
    assert raw_band.fermi_energy == 0
    band.ref.bands_up = raw_band.eigenvalues[0]
    band.ref.bands_down = raw_band.eigenvalues[1]
    band.ref.occupations_up = raw_band.occupations[0]
    band.ref.occupations_down = raw_band.occupations[1]
    return band


@pytest.fixture
def spin_projectors(raw_data):
    raw_band = raw_data.band("spin_polarized with_projectors")
    band = Band(raw_band)
    band.ref = types.SimpleNamespace()
    band.ref.bands_up = raw_band.eigenvalues[0]
    band.ref.bands_down = raw_band.eigenvalues[1]
    band.ref.s_up = np.sum(raw_band.projections[0, :, 0, :, :], axis=0)
    band.ref.s_down = np.sum(raw_band.projections[1, :, 0, :, :], axis=0)
    band.ref.Fe_d_up = np.sum(raw_band.projections[0, 0:3, 2, :, :], axis=0)
    band.ref.Fe_d_down = np.sum(raw_band.projections[1, 0:3, 2, :, :], axis=0)
    band.ref.O_up = np.sum(raw_band.projections[0, 3:7, :, :, :], axis=(0, 1))
    band.ref.O_down = np.sum(raw_band.projections[1, 3:7, :, :, :], axis=(0, 1))
    band.ref.projectors_string = pretty(Projectors(raw_band.projectors))
    return band


def test_single_band_read(single_band, Assert):
    band = single_band.read()
    assert band["fermi_energy"] == single_band.ref.fermi_energy
    Assert.allclose(band["bands"], single_band.ref.bands)
    Assert.allclose(band["occupations"], single_band.ref.occupations)
    Assert.allclose(band["kpoint_distances"], single_band.ref.kpoints.distances())
    assert band["kpoint_labels"] == single_band.ref.kpoints.labels()
    assert len(band["projections"]) == 0


def test_multiple_bands_read(multiple_bands, Assert):
    band = multiple_bands.read()
    assert band["fermi_energy"] == multiple_bands.ref.fermi_energy
    Assert.allclose(band["bands"], multiple_bands.ref.bands)
    Assert.allclose(band["occupations"], multiple_bands.ref.occupations)


def test_with_projectors_read(with_projectors, Assert):
    band = with_projectors.read("Sr p")
    Assert.allclose(band["projections"]["Sr"], with_projectors.ref.Sr)
    Assert.allclose(band["projections"]["p"], with_projectors.ref.p)


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
    Assert.allclose(band["projections"]["s_up"], spin_projectors.ref.s_up)
    Assert.allclose(band["projections"]["s_down"], spin_projectors.ref.s_down)
    Assert.allclose(band["projections"]["Fe_d_up"], spin_projectors.ref.Fe_d_up)
    Assert.allclose(band["projections"]["Fe_d_down"], spin_projectors.ref.Fe_d_down)


def test_more_projections_style(raw_data, Assert):
    """Vasp 6.1 may store more orbital types then projections available. This
    test checks that this does not lead to any issues when an available element
    is used."""
    band = Band(raw_data.band("spin_polarized excess_orbitals")).read("Fe g")
    zero = np.zeros_like(band["projections"]["Fe_up"])
    Assert.allclose(band["projections"]["g_up"], zero)
    Assert.allclose(band["projections"]["g_down"], zero)


def test_single_polarized_to_frame(single_band, Assert):
    actual = single_band.to_frame()
    assert all(actual.index == single_band.ref.index)
    Assert.allclose(actual.bands, single_band.ref.bands[:, 0])
    Assert.allclose(actual.occupations, single_band.ref.occupations[:, 0])


def test_multiple_bands_to_frame(multiple_bands, Assert):
    actual = multiple_bands.to_frame()
    assert actual.index[0] == "[0.00 0.00 0.12] 1"
    assert actual.index[1] == "2"
    assert actual.index[2] == "3"
    assert actual.index[3] == "[0.00 0.00 0.38] 1"
    Assert.allclose(actual.bands, multiple_bands.ref.bands.T.flatten())
    Assert.allclose(actual.occupations, multiple_bands.ref.occupations.T.flatten())


def test_with_projectors_to_frame(with_projectors, Assert):
    actual = with_projectors.to_frame("Sr p")
    Assert.allclose(actual.Sr, with_projectors.ref.Sr.T.flatten())
    Assert.allclose(actual.p, with_projectors.ref.p.T.flatten())


def test_spin_polarized_to_frame(spin_polarized, Assert):
    actual = spin_polarized.to_frame()
    ref = spin_polarized.ref
    Assert.allclose(actual.bands_up, ref.bands_up.T.flatten())
    Assert.allclose(actual.bands_down, ref.bands_down.T.flatten())
    Assert.allclose(actual.occupations_up, ref.occupations_up.T.flatten())
    Assert.allclose(actual.occupations_down, ref.occupations_down.T.flatten())


def test_spin_projectors_to_frame(spin_projectors, Assert):
    actual = spin_projectors.to_frame(selection="O Fe(d)")
    Assert.allclose(actual.O_up, spin_projectors.ref.O_up.T.flatten())
    Assert.allclose(actual.O_down, spin_projectors.ref.O_down.T.flatten())
    Assert.allclose(actual.Fe_d_up, spin_projectors.ref.Fe_d_up.T.flatten())
    Assert.allclose(actual.Fe_d_down, spin_projectors.ref.Fe_d_down.T.flatten())


def test_single_band_plot(single_band, Assert):
    fig = single_band.plot()
    assert fig.layout.yaxis.title.text == "Energy (eV)"
    assert len(fig.data) == 1
    assert fig.data[0].fill is None
    assert fig.data[0].mode is None
    mask = np.isfinite(fig.data[0].x)  # Band may insert NaN to split plot
    Assert.allclose(fig.data[0].x[mask], single_band.ref.kpoints.distances())
    Assert.allclose(fig.data[0].y[mask], single_band.ref.bands.flatten())


def test_multiple_bands_plot(multiple_bands, Assert):
    fig = multiple_bands.plot()
    assert len(fig.data) == 1  # all bands in one plot
    assert len(fig.data[0].x) == len(fig.data[0].y)
    num_NaN_x = np.count_nonzero(np.isnan(fig.data[0].x))
    num_NaN_y = np.count_nonzero(np.isnan(fig.data[0].y))
    assert num_NaN_x == num_NaN_y > 0
    mask = np.isfinite(fig.data[0].x)
    Assert.allclose(fig.data[0].y[mask], multiple_bands.ref.bands.T.flatten())


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
    fig = spin_projectors.plot("O", width)
    assert len(fig.data) == 2
    assert fig.data[0].name == "O_up"
    check_data(fig.data[0], width, reference.bands_up, reference.O_up, Assert)
    assert fig.data[1].name == "O_down"
    check_data(fig.data[1], width, reference.bands_down, reference.O_down, Assert)


def check_figure(fig, width, reference, Assert):
    assert len(fig.data) == 2
    assert fig.data[0].name == "Sr"
    assert fig.data[1].name == "p"
    check_data(fig.data[0], width, reference.bands, reference.Sr, Assert)
    check_data(fig.data[1], width, reference.bands, reference.p, Assert)


def check_data(data, width, band, projection, Assert):
    check_data_internal_consistency(data)
    for band, weight in zip(np.nditer(band), np.nditer(projection)):
        upper, lower = band + width * weight, band - width * weight
        check_data_vs_reference(data, upper, lower, Assert)


def check_data_internal_consistency(data):
    assert len(data.x) == len(data.y)
    assert data.fill == "toself"
    assert data.mode == "none"
    num_NaN_x = np.count_nonzero(np.isnan(data.x))
    num_NaN_y = np.count_nonzero(np.isnan(data.y))
    assert num_NaN_x == num_NaN_y > 0


def check_data_vs_reference(data, upper, lower, Assert):
    pos_upper = data.x[np.where(np.isclose(data.y, upper, 1e-10, 1e-10))]
    pos_lower = data.x[np.where(np.isclose(data.y, lower, 1e-10, 1e-10))]
    assert len(pos_upper) == len(pos_lower) == 1
    Assert.allclose(pos_upper, pos_lower)


def test_spin_polarized_plot(spin_polarized, Assert):
    fig = spin_polarized.plot()
    assert len(fig.data) == 2
    assert fig.data[0].name == "bands_up"
    mask = np.isfinite(fig.data[0].x)
    Assert.allclose(fig.data[0].y[mask], spin_polarized.ref.bands_up.T.flatten())
    assert fig.data[1].name == "bands_down"
    mask = np.isfinite(fig.data[1].x)
    Assert.allclose(fig.data[1].y[mask], spin_polarized.ref.bands_down.T.flatten())


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
    assert fig.layout.xaxis.ticktext == reference_labels


def test_line_with_labels_plot(line_with_labels, Assert):
    fig = line_with_labels.plot()
    check_ticks(fig, line_with_labels.ref.kpoints, Assert)
    assert fig.layout.xaxis.ticktext == (r"$\Gamma$", " ", r"M|$\Gamma$", "Y", "M")


def check_ticks(fig, kpoints, Assert):
    dists = kpoints.distances()
    xticks = (*dists[:: kpoints.line_length()], dists[-1])
    assert fig.layout.xaxis.tickmode == "array"
    Assert.allclose(fig.layout.xaxis.tickvals, np.array(xticks))


def test_plot_incorrect_width(with_projectors):
    with pytest.raises(exception.IncorrectUsage):
        with_projectors.plot("Sr", width="not a number")


def test_to_png(single_band):
    filename = "image.png"
    with patch("py4vasp.data.band._to_plotly") as plot:
        single_band.to_png(filename, "args", key="word")
        plot.assert_called_once()
        assert plot.call_args.args[1] == "args"
        assert plot.call_args.kwargs == {"key": "word"}
        fig = plot.return_value
        fig.write_image.assert_called_once_with(filename)


def test_multiple_bands_print(multiple_bands, format_):
    actual, _ = format_(multiple_bands)
    reference = f"""
band data:
    48 k-points
    3 bands
    """.strip()
    assert actual == {"text/plain": reference}


def test_line_no_labels_print(line_no_labels, format_):
    actual, _ = format_(line_no_labels)
    reference = f"""
band data:
    20 k-points
    3 bands
    """.strip()
    assert actual == {"text/plain": reference}


def test_line_with_labels_print(line_with_labels, format_):
    actual, _ = format_(line_with_labels)
    reference = f"""
band data:
    20 k-points
    3 bands
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


def test_descriptor(single_band, check_descriptors):
    descriptors = {
        "_to_dict": ["to_dict", "read"],
        "_to_plotly": ["to_plotly", "plot"],
        "_to_frame": ["to_frame"],
        "_to_png": ["to_png"],
    }
    check_descriptors(single_band, descriptors)


#
# def test_default_from_file(raw_band, mock_file, check_read):
#     with mock_file("band", raw_band) as mocks:
#         check_read(Band, mocks, raw_band)

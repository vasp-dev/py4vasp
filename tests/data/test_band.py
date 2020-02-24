from py4vasp.data import Band, Kpoints
import py4vasp.raw as raw
import pytest
import numpy as np

number_spins = 1
number_kpoints = 60
number_bands = 1
number_atoms = 1
number_orbitals = 1


@pytest.fixture
def raw_band():
    return raw.Band(
        fermi_energy=0.0,
        eigenvalues=np.array([np.linspace([0], [1], number_kpoints)]),
        kpoints=raw.Kpoints(
            mode="explicit",
            number=number_kpoints,
            coordinates=np.linspace(np.zeros(3), np.ones(3), number_kpoints),
            weights=None,
            cell=raw.Cell(scale=1.0, lattice_vectors=np.eye(3)),
        ),
    )


def test_default_read(raw_band, Assert):
    band = Band(raw_band).read()
    assert band["fermi_energy"] == raw_band.fermi_energy
    Assert.allclose(band["bands"], raw_band.eigenvalues[0])
    kpoints = Kpoints(raw_band.kpoints)
    Assert.allclose(band["kpoint_distances"], kpoints.distances())
    assert band["kpoint_labels"] == kpoints.labels()
    assert len(band["projections"]) == 0


def test_default_plot(raw_band, Assert):
    fig = Band(raw_band).plot()
    assert fig.layout.yaxis.title.text == "Energy (eV)"
    assert len(fig.data) == 1
    assert fig.data[0].fill is None
    assert fig.data[0].mode is None
    ref_dists = Kpoints(raw_band.kpoints).distances()
    ref_bands = raw_band.eigenvalues.flatten()
    mask = np.isfinite(fig.data[0].x)  # Band may insert NaN to split plot
    Assert.allclose(fig.data[0].x[mask], ref_dists)
    Assert.allclose(fig.data[0].y[mask], ref_bands)


def test_default_from_file(raw_band, mock_file, check_read):
    with mock_file("band", raw_band) as mocks:
        check_read(Band, mocks, raw_band)


@pytest.fixture
def multiple_bands(raw_band):
    number_bands_ = 3
    shape = (number_spins, number_kpoints, number_bands_)
    raw_band.eigenvalues = np.arange(np.prod(shape)).reshape(shape)
    return raw_band


def test_multiple_bands_read(multiple_bands, Assert):
    band = Band(multiple_bands).read()
    ref_bands = multiple_bands.eigenvalues[0]
    Assert.allclose(band["bands"], ref_bands)


def test_multiple_bands_plot(multiple_bands, Assert):
    fig = Band(multiple_bands).plot()
    assert len(fig.data) == 1  # all bands in one plot
    assert len(fig.data[0].x) == len(fig.data[0].y)
    num_NaN_x = np.count_nonzero(np.isnan(fig.data[0].x))
    num_NaN_y = np.count_nonzero(np.isnan(fig.data[0].y))
    assert num_NaN_x == num_NaN_y > 0
    ref_bands = multiple_bands.eigenvalues.flatten("F")
    mask = np.isfinite(fig.data[0].x)
    Assert.allclose(fig.data[0].y[mask], ref_bands)


def test_nontrivial_cell(raw_band, Assert):
    raw_band.kpoints.cell = raw.Cell(
        scale=2.0, lattice_vectors=np.array([[3, 0, 0], [-1, 2, 0], [0, 0, 4]])
    )
    cartesian_kpoints = np.linspace(np.zeros(3), np.ones(3))
    cell = raw_band.kpoints.cell.lattice_vectors * raw_band.kpoints.cell.scale
    raw_band.kpoints.coordinates = cartesian_kpoints @ cell.T
    band = Band(raw_band).read()
    ref_dists = np.linalg.norm(cartesian_kpoints, axis=1)
    Assert.allclose(band["kpoint_distances"], ref_dists)


@pytest.fixture
def fermi_energy(raw_band):
    raw_band.fermi_energy = 0.5
    return raw_band


def test_fermi_energy_read(fermi_energy, Assert):
    raw_band = fermi_energy
    band = Band(raw_band).read()
    assert band["fermi_energy"] == raw_band.fermi_energy
    Assert.allclose(band["bands"], raw_band.eigenvalues[0] - raw_band.fermi_energy)


def test_fermi_energy_plot(fermi_energy, Assert):
    raw_band = fermi_energy
    fig = Band(raw_band).plot()
    ref_bands = raw_band.eigenvalues.flatten() - raw_band.fermi_energy
    mask = np.isfinite(fig.data[0].x)
    Assert.allclose(fig.data[0].y[mask], ref_bands)


@pytest.fixture
def line_without_labels(raw_band):
    R = [0.5, 0.5, 0.5]
    X = [1, 0, 0]
    Y = [0, 1, 0]
    G = [0, 0, 0]
    M = [0.5, 0.5, 0]
    path1 = np.linspace(R, X, number_kpoints)
    path2 = np.linspace(Y, G, number_kpoints)
    path3 = np.linspace(G, X, number_kpoints)
    path4 = np.linspace(X, M, number_kpoints)
    raw_band.kpoints.mode = "line"
    raw_band.kpoints.coordinates = np.concatenate((path1, path2, path3, path4))
    return raw_band


def test_line_without_labels_plot(line_without_labels, Assert):
    fig = Band(line_without_labels).plot()
    check_ticks(fig, line_without_labels, Assert)
    assert fig.layout.xaxis.ticktext == (" ", " ", " ", " ", " ")


@pytest.fixture
def line_with_labels(line_without_labels):
    labels = ["X", "Y", "G", "X"]
    line_without_labels.kpoints.labels = np.array(labels, dtype="S")
    line_without_labels.kpoints.label_indices = label_indices = [2, 3, 4, 6]
    return line_without_labels


def test_line_with_labels_read(line_with_labels, Assert):
    band = Band(line_with_labels).read()
    kpoints = Kpoints(line_with_labels.kpoints)
    Assert.allclose(band["kpoint_distances"], kpoints.distances())
    assert band["kpoint_labels"] == kpoints.labels()


def test_line_with_labels_plot(line_with_labels, Assert):
    fig = Band(line_with_labels).plot()
    check_ticks(fig, line_with_labels, Assert)
    assert fig.layout.xaxis.ticktext == (" ", "X|Y", "G", "X", " ")


def check_ticks(fig, raw_band, Assert):
    dists = Kpoints(raw_band.kpoints).distances()
    xticks = (*dists[::number_kpoints], dists[-1])
    assert fig.layout.xaxis.tickmode == "array"
    Assert.allclose(fig.layout.xaxis.tickvals, np.array(xticks))


@pytest.fixture
def spin_band(raw_band):
    number_spins_ = 2
    shape = (number_spins_, number_kpoints, number_bands)
    raw_band.eigenvalues = np.arange(np.prod(shape)).reshape(shape)
    return raw_band


def test_spin_band_read(spin_band, Assert):
    band = Band(spin_band).read()
    Assert.allclose(band["up"], spin_band.eigenvalues[0])
    Assert.allclose(band["down"], spin_band.eigenvalues[1])


def test_spin_band_fermi_energy(spin_band, Assert):
    spin_band.fermi_energy = 0.5
    band = Band(spin_band).read()
    Assert.allclose(band["up"], spin_band.eigenvalues[0] - spin_band.fermi_energy)
    Assert.allclose(band["down"], spin_band.eigenvalues[1] - spin_band.fermi_energy)


def test_spin_band_plot(spin_band, Assert):
    fig = Band(spin_band).plot()
    assert len(fig.data) == 2
    spins = ["up", "down"]
    for i, (spin, data) in enumerate(zip(spins, fig.data)):
        assert data.name == spin
        mask = np.isfinite(data.x)
        Assert.allclose(data.y[mask], spin_band.eigenvalues[i].flatten("F"))


@pytest.fixture
def spin_projections(spin_band):
    number_spins_ = 2
    shape = (number_spins_, number_atoms, number_orbitals, number_kpoints, number_bands)
    return set_projections(spin_band, shape)


def test_spin_projections_read(spin_projections, Assert):
    raw_band = spin_projections
    band = Band(raw_band).read("s")
    Assert.allclose(band["projections"]["s_up"], raw_band.projections[0, 0])
    Assert.allclose(band["projections"]["s_down"], raw_band.projections[1, 0])


def test_spin_projections_plot(spin_projections, Assert):
    raw_band = spin_projections
    width = 0.05
    fig = Band(raw_band).plot("Si", width)
    assert len(fig.data) == 2
    spins = ["up", "down"]
    for i, (spin, data) in enumerate(zip(spins, fig.data)):
        assert data.name == "Si_" + spin
        bands = np.nditer(raw_band.eigenvalues[i])
        weights = np.nditer(raw_band.projections[i, 0, 0])
        for band, weight in zip(bands, weights):
            upper = band + width * weight
            lower = band - width * weight
            pos_upper = data.x[np.where(np.isclose(data.y, upper))]
            pos_lower = data.x[np.where(np.isclose(data.y, lower))]
            assert len(pos_upper) == len(pos_lower) == 1
            Assert.allclose(pos_upper, pos_lower)


@pytest.fixture
def raw_projections(raw_band):
    shape = (number_spins, number_atoms, number_orbitals, number_kpoints, number_bands)
    return set_projections(raw_band, shape)


def test_raw_projections_read(raw_projections, Assert):
    raw_band = raw_projections
    band = Band(raw_band).read("Si(s)")
    Assert.allclose(band["projections"]["Si_s"], raw_band.projections[0, 0, 0])


def test_raw_projections_plot(raw_projections, Assert):
    raw_band = raw_projections
    default_width = 0.5
    fig = Band(raw_band).plot("s, 1")
    assert len(fig.data) == 2
    assert fig.data[0].name == "s"
    assert fig.data[1].name == "Si_1"
    for data in fig.data:
        assert len(data.x) == len(data.y)
        assert data.fill == "toself"
        assert data.mode == "none"
        num_NaN_x = np.count_nonzero(np.isnan(data.x))
        num_NaN_y = np.count_nonzero(np.isnan(data.y))
        assert num_NaN_x == num_NaN_y > 0
        bands = np.nditer(raw_band.eigenvalues[0])
        weights = np.nditer(raw_band.projections[0, 0, 0])
        for band, weight in zip(bands, weights):
            upper = band + default_width * weight
            lower = band - default_width * weight
            pos_upper = data.x[np.where(np.isclose(data.y, upper))]
            pos_lower = data.x[np.where(np.isclose(data.y, lower))]
            assert len(pos_upper) == len(pos_lower) == 1
            Assert.allclose(pos_upper, pos_lower)


def test_more_projections_style(raw_projections, Assert):
    """Vasp 6.1 may define more orbital types then are available as projections.
    Here we check that the correct orbitals are read."""
    raw_projections.projectors.orbital_types = np.array(["s", "p"], dtype="S")
    band = Band(raw_projections).read("Si")
    Assert.allclose(band["projections"]["Si"], raw_projections.projections[0, 0, 0])


def set_projections(raw_band, shape):
    raw_band.projections = np.random.uniform(low=0.2, size=shape)
    raw_band.projectors = raw.Projectors(
        number_ion_types=[1],
        ion_types=np.array(["Si"], dtype="S"),
        orbital_types=np.array(["s"], dtype="S"),
        number_spins=shape[0],
    )
    return raw_band

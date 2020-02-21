from py4vasp.data import Band, Kpoints
import py4vasp.raw as raw
import pytest
import numpy as np

number_kpoints = 60


@pytest.fixture
def raw_band():
    return raw.Band(
        fermi_energy=0.5,
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
    Assert.allclose(band["bands"], raw_band.eigenvalues[0] - raw_band.fermi_energy)
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
    ref_bands = raw_band.eigenvalues.flatten() - raw_band.fermi_energy
    mask = np.isfinite(fig.data[0].x)  # Band may insert NaN to split plot
    Assert.allclose(fig.data[0].x[mask], ref_dists)
    Assert.allclose(fig.data[0].y[mask], ref_bands)


def test_default_from_file(raw_band, mock_file, check_read):
    with mock_file("band", raw_band) as mocks:
        check_read(Band, mocks, raw_band)


@pytest.fixture
def multiple_bands(raw_band):
    number_bands = 3
    shape = (1, number_kpoints, number_bands)
    raw_band.eigenvalues = np.arange(np.prod(shape)).reshape(shape)
    return raw_band


def test_multiple_bands_read(multiple_bands, Assert):
    band = Band(multiple_bands).read()
    ref_bands = multiple_bands.eigenvalues[0] - multiple_bands.fermi_energy
    Assert.allclose(band["bands"], ref_bands)


def test_multiple_bands_plot(multiple_bands, Assert):
    fig = Band(multiple_bands).plot()
    assert len(fig.data) == 1  # all bands in one plot
    assert len(fig.data[0].x) == len(fig.data[0].y)
    num_NaN_x = np.count_nonzero(np.isnan(fig.data[0].x))
    num_NaN_y = np.count_nonzero(np.isnan(fig.data[0].y))
    assert num_NaN_x == num_NaN_y > 0
    ref_bands = multiple_bands.eigenvalues.flatten("F") - multiple_bands.fermi_energy
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
def kpoint_path(raw_band):
    kpoints = np.array([[0.5, 0.5, 0.5], [1, 0, 0], [0, 1, 0], [0, 0, 0]])
    first_path = np.linspace(kpoints[0], kpoints[1], number_kpoints)
    second_path = np.linspace(kpoints[2], kpoints[3], number_kpoints)
    raw_band.kpoints.mode = "line"
    raw_band.kpoints.coordinates = np.concatenate((first_path, second_path))
    raw_band.kpoints.labels = np.array(["X", "Y", "G"], dtype="S")
    raw_band.kpoints.label_indices = label_indices = [2, 3, 4]
    return raw_band


def test_kpoint_path_read(kpoint_path, Assert):
    band = Band(kpoint_path).read()
    kpoints = Kpoints(kpoint_path.kpoints)
    Assert.allclose(band["kpoint_distances"], kpoints.distances())
    assert band["kpoint_labels"] == kpoints.labels()


def test_kpoint_path_plot(kpoint_path, Assert):
    fig = Band(kpoint_path).plot()
    dists = Kpoints(kpoint_path.kpoints).distances()
    xticks = (dists[0], dists[number_kpoints], dists[-1])
    assert fig.layout.xaxis.tickmode == "array"
    Assert.allclose(fig.layout.xaxis.tickvals, np.array(xticks))
    assert fig.layout.xaxis.ticktext == (" ", "X|Y", "G")


@pytest.fixture
def spin_band_structure():
    num_spins, num_atoms, num_orbitals, num_kpoints, num_bands = 2, 1, 1, 50, 5
    shape_proj = (num_spins, num_atoms, num_orbitals, num_kpoints, num_bands)
    shape_eig = (num_spins, num_kpoints, num_bands)
    size_eig = np.prod(shape_eig)
    raw_band = raw.Band(
        eigenvalues=np.arange(size_eig).reshape(shape_eig),
        projections=np.random.uniform(low=0.2, size=shape_proj),
        projectors=raw.Projectors(
            number_ion_types=[1],
            ion_types=np.array(["Si"], dtype="S"),
            orbital_types=np.array(["s"], dtype="S"),
            number_spins=num_spins,
        ),
        fermi_energy=0.0,
        kpoints=raw.Kpoints(
            mode="explicit",
            coordinates=np.linspace(np.zeros(3), np.ones(3), num_kpoints),
            weights=None,
            number=num_kpoints,
            cell=raw.Cell(scale=1, lattice_vectors=np.eye(3)),
        ),
    )
    return raw_band


def test_spin_band_structure_read(spin_band_structure, Assert):
    raw_band = spin_band_structure
    band = Band(raw_band).read("s")
    Assert.allclose(band["up"], raw_band.eigenvalues[0])
    Assert.allclose(band["down"], raw_band.eigenvalues[1])
    Assert.allclose(band["projections"]["s_up"], raw_band.projections[0, 0])
    Assert.allclose(band["projections"]["s_down"], raw_band.projections[1, 0])


def test_spin_band_structure_plot(spin_band_structure, Assert):
    width = 0.05
    raw_band = spin_band_structure
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
def projected_band_structure():
    num_spins, num_atoms, num_orbitals, num_kpoints, num_bands = 1, 1, 1, 60, 2
    shape_proj = (num_spins, num_atoms, num_orbitals, num_kpoints, num_bands)
    shape_eig = (num_spins, num_kpoints, num_bands)
    raw_band = raw.Band(
        eigenvalues=np.arange(np.prod(shape_eig)).reshape(shape_eig),
        projections=np.random.uniform(low=0.2, size=shape_proj),
        projectors=raw.Projectors(
            number_ion_types=[1],
            ion_types=np.array(["Si"], dtype="S"),
            orbital_types=np.array(["s"], dtype="S"),
            number_spins=num_spins,
        ),
        fermi_energy=0.0,
        kpoints=raw.Kpoints(
            mode="explicit",
            coordinates=np.linspace(np.zeros(3), np.ones(3), num_kpoints),
            weights=None,
            number=num_kpoints,
            cell=raw.Cell(scale=1, lattice_vectors=np.eye(3)),
        ),
    )
    return raw_band


def test_projected_band_structure_read(projected_band_structure, Assert):
    raw_band = projected_band_structure
    band = Band(raw_band).read("Si(s)")
    Assert.allclose(band["projections"]["Si_s"], raw_band.projections[0, 0, 0])


def test_projected_band_structure_plot(projected_band_structure, Assert):
    default_width = 0.5
    raw_band = projected_band_structure
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

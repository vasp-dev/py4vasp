from py4vasp.data import Band
import py4vasp.raw as raw
import pytest
import numpy as np


@pytest.fixture
def two_parabolic_bands():
    cell = raw.Cell(
        scale=2.0, lattice_vectors=np.array([[3, 0, 0], [-1, 2, 0], [0, 0, 4]])
    )
    cartesian_kpoints = np.linspace(np.zeros(3), np.ones(3))
    direct_kpoints = cartesian_kpoints @ cell.lattice_vectors.T * cell.scale
    ref = {"kdists": np.linalg.norm(cartesian_kpoints, axis=1)}
    ref["valence_band"] = -ref["kdists"] ** 2
    ref["conduction_band"] = 1.0 + ref["kdists"] ** 2
    raw_band = raw.Band(
        fermi_energy=0.5,
        line_length=len(cartesian_kpoints),
        kpoints=direct_kpoints,
        eigenvalues=[np.array([ref["valence_band"], ref["conduction_band"]]).T],
        cell=cell,
    )
    return raw_band, ref


def test_parabolic_band_read(two_parabolic_bands, Assert):
    raw_band, ref = two_parabolic_bands
    band = Band(raw_band).read()
    assert band["bands"].shape == (len(ref["valence_band"]), 2)
    assert band["fermi_energy"] == raw_band.fermi_energy
    Assert.allclose(band["kpoints"], raw_band.kpoints)
    Assert.allclose(band["kpoint_distances"], ref["kdists"])
    assert band["kpoint_labels"] is None
    Assert.allclose(band["bands"][:, 0], ref["valence_band"] - raw_band.fermi_energy)
    Assert.allclose(band["bands"][:, 1], ref["conduction_band"] - raw_band.fermi_energy)


def test_parabolic_band_plot(two_parabolic_bands, Assert):
    raw_band, ref = two_parabolic_bands
    fig = Band(raw_band).plot()
    assert fig.layout.yaxis.title.text == "Energy (eV)"
    assert len(fig.data) == 1
    assert fig.data[0].fill is None
    assert fig.data[0].mode is None
    assert len(fig.data[0].x) == len(fig.data[0].y)
    num_NaN_x = np.count_nonzero(np.isnan(fig.data[0].x))
    num_NaN_y = np.count_nonzero(np.isnan(fig.data[0].y))
    assert num_NaN_x == num_NaN_y > 0
    for val, vb, cb in zip(ref["kdists"], ref["valence_band"], ref["conduction_band"]):
        bands = fig.data[0].y[np.where(np.isclose(fig.data[0].x, val))]
        ref_bands = np.array([vb, cb]) - raw_band.fermi_energy
        Assert.allclose(bands, ref_bands)


def test_parabolic_band_from_file(two_parabolic_bands, mock_file, check_read):
    raw_band, _ = two_parabolic_bands
    with mock_file("band", raw_band) as mocks:
        check_read(Band, mocks, raw_band)


@pytest.fixture
def kpoint_path():
    N = 50
    num_kpoints = 2 * N
    kpoints = np.array([[0.5, 0.5, 0.5], [1, 0, 0], [0, 1, 0], [0, 0, 0]])
    first_path = np.linspace(kpoints[0], kpoints[1], N)
    second_path = np.linspace(kpoints[2], kpoints[3], N)
    raw_band = raw.Band(
        line_length=N,
        kpoints=np.concatenate((first_path, second_path)),
        labels=np.array(["X", "Y", "G"], dtype="S"),
        label_indices=[2, 3, 4],
        fermi_energy=0.0,
        eigenvalues=np.zeros((1, num_kpoints, 1)),
        cell=raw.Cell(scale=1, lattice_vectors=np.eye(3)),
    )
    first_dists = np.linalg.norm(first_path - first_path[0], axis=1)
    second_dists = np.linalg.norm(second_path - second_path[0], axis=1)
    second_dists += first_dists[-1]
    ref = {
        "line_length": N,
        "kdists": np.concatenate((first_dists, second_dists)),
        "klabels": ([""] * (N - 1) + ["X", "Y"] + [""] * (N - 2) + ["G"]),
        "ticklabels": (" ", "X|Y", "G"),
    }
    return raw_band, ref


def test_kpoint_path_read(kpoint_path, Assert):
    raw_band, ref = kpoint_path
    band = Band(raw_band).read()
    Assert.allclose(band["kpoints"], raw_band.kpoints)
    Assert.allclose(band["kpoint_distances"], ref["kdists"])
    assert band["kpoint_labels"] == ref["klabels"]


def test_kpoint_path_plot(kpoint_path, Assert):
    raw_band, ref = kpoint_path
    fig = Band(raw_band).plot()
    xticks = (ref["kdists"][0], ref["kdists"][raw_band.line_length], ref["kdists"][-1])
    assert len(fig.data[0].x) == len(fig.data[0].y)
    assert fig.layout.xaxis.tickmode == "array"
    Assert.allclose(fig.layout.xaxis.tickvals, np.array(xticks))
    assert fig.layout.xaxis.ticktext == ref["ticklabels"]


@pytest.fixture
def spin_band_structure():
    num_spins, num_atoms, num_orbitals, num_kpoints, num_bands = 2, 1, 1, 50, 5
    shape_proj = (num_spins, num_atoms, num_orbitals, num_kpoints, num_bands)
    shape_eig = (num_spins, num_kpoints, num_bands)
    size_eig = np.prod(shape_eig)
    raw_band = raw.Band(
        line_length=num_kpoints,
        kpoints=np.linspace(np.zeros(3), np.ones(3), num_kpoints),
        eigenvalues=np.arange(size_eig).reshape(shape_eig),
        projections=np.random.uniform(low=0.2, size=shape_proj),
        projectors=raw.Projectors(
            number_ion_types=[1],
            ion_types=np.array(["Si"], dtype="S"),
            orbital_types=np.array(["s"], dtype="S"),
            number_spins=num_spins,
        ),
        fermi_energy=0.0,
        cell=raw.Cell(scale=1, lattice_vectors=np.eye(3)),
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
        line_length=num_kpoints,
        kpoints=np.linspace(np.zeros(3), np.ones(3), num_kpoints),
        eigenvalues=np.arange(np.prod(shape_eig)).reshape(shape_eig),
        projections=np.random.uniform(low=0.2, size=shape_proj),
        projectors=raw.Projectors(
            number_ion_types=[1],
            ion_types=np.array(["Si"], dtype="S"),
            orbital_types=np.array(["s"], dtype="S"),
            number_spins=num_spins,
        ),
        fermi_energy=0.0,
        cell=raw.Cell(scale=1, lattice_vectors=np.eye(3)),
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

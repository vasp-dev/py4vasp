from py4vasp.data import Band
import pytest
import h5py
import numpy as np
from numpy.testing import assert_array_almost_equal_nulp
from tempfile import TemporaryFile


def assert_allclose(actual, desired):
    assert_array_almost_equal_nulp(actual, desired, 10)


@pytest.fixture
def two_parabolic_bands():
    h5f = h5py.File(TemporaryFile(), "a")
    scale = 2.0
    cell = np.array([[3, 0, 0], [-1, 2, 0], [0, 0, 4]])
    cartesian_kpoints = np.linspace(np.zeros(3), np.ones(3))
    direct_kpoints = cartesian_kpoints @ cell.T
    ref = {"kpoints": direct_kpoints}
    ref["kdists"] = np.linalg.norm(cartesian_kpoints, axis=1)
    ref["valence_band"] = -ref["kdists"] ** 2
    ref["conduction_band"] = 1.0 + ref["kdists"] ** 2
    ref["fermi_energy"] = 0.5
    h5f["input/kpoints/number_kpoints"] = len(cartesian_kpoints)
    h5f["results/eigenvalues/kpoint_coords"] = ref["kpoints"]
    eigenvalues = [np.array([ref["valence_band"], ref["conduction_band"]]).T]
    h5f["results/eigenvalues/eigenvalues"] = eigenvalues
    h5f["results/dos/efermi"] = ref["fermi_energy"]
    h5f["results/positions/scale"] = scale
    h5f["results/positions/lattice_vectors"] = cell / scale
    return h5f, ref


def test_parabolic_band_read(two_parabolic_bands):
    h5f, ref = two_parabolic_bands
    band = Band(h5f).read()
    assert band["bands"].shape == (len(ref["valence_band"]), 2)
    assert band["fermi_energy"] == ref["fermi_energy"]
    assert_allclose(band["kpoints"], ref["kpoints"])
    assert_allclose(band["kpoint_distances"], ref["kdists"])
    assert band["kpoint_labels"] is None
    assert_allclose(band["bands"][:, 0], ref["valence_band"] - ref["fermi_energy"])
    assert_allclose(band["bands"][:, 1], ref["conduction_band"] - ref["fermi_energy"])


def test_parabolic_band_plot(two_parabolic_bands):
    h5f, ref = two_parabolic_bands
    fig = Band(h5f).plot()
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
        ref_bands = np.array([vb, cb]) - ref["fermi_energy"]
        assert_allclose(bands, ref_bands)


@pytest.fixture
def kpoint_path():
    N = 50
    h5f = h5py.File(TemporaryFile(), "a")
    kpoints = np.array([[0.5, 0.5, 0.5], [1, 0, 0], [0, 1, 0], [0, 0, 0]])
    first_path = np.linspace(kpoints[0], kpoints[1], N)
    first_dists = np.linalg.norm(first_path - first_path[0], axis=1)
    second_path = np.linspace(kpoints[2], kpoints[3], N)
    second_dists = np.linalg.norm(second_path - second_path[0], axis=1)
    second_dists += first_dists[-1]
    ref = {
        "line_length": N,
        "kpoints": np.concatenate((first_path, second_path)),
        "kdists": np.concatenate((first_dists, second_dists)),
        "klabels": ([""] * (N - 1) + ["X", "Y"] + [""] * (N - 2) + ["G"]),
        "ticklabels": (" ", "X|Y", "G"),
    }
    h5f["input/kpoints/number_kpoints"] = N
    h5f["input/kpoints/labels_kpoints"] = np.array(["X", "Y", "G"], dtype="S")
    h5f["input/kpoints/positions_labels_kpoints"] = [2, 3, 4]
    h5f["results/eigenvalues/kpoint_coords"] = ref["kpoints"]
    num_kpoints = len(ref["kpoints"])
    h5f["results/eigenvalues/eigenvalues"] = np.zeros((1, num_kpoints, 1))
    h5f["results/dos/efermi"] = 0.0
    h5f["results/positions/scale"] = 1.0
    h5f["results/positions/lattice_vectors"] = np.eye(3)
    return h5f, ref


def test_kpoint_path_read(kpoint_path):
    h5f, ref = kpoint_path
    band = Band(h5f).read()
    assert_allclose(band["kpoints"], ref["kpoints"])
    assert_allclose(band["kpoint_distances"], ref["kdists"])
    assert band["kpoint_labels"] == ref["klabels"]


def test_kpoint_path_plot(kpoint_path):
    h5f, ref = kpoint_path
    fig = Band(h5f).plot()
    xticks = (ref["kdists"][0], ref["kdists"][ref["line_length"]], ref["kdists"][-1])
    assert len(fig.data[0].x) == len(fig.data[0].y)
    assert fig.layout.xaxis.tickmode == "array"
    assert_allclose(fig.layout.xaxis.tickvals, np.array(xticks))
    assert fig.layout.xaxis.ticktext == ref["ticklabels"]


@pytest.fixture
def spin_band_structure():
    h5f = h5py.File(TemporaryFile(), "a")
    kpoints = np.linspace(np.zeros(3), np.ones(3))
    shape = (len(kpoints), 5)
    size = np.prod(shape)
    ref = {
        "up": np.arange(size).reshape(shape),
        "down": np.arange(size, 2 * size).reshape(shape),
        "proj_up": np.random.uniform(low=0.2, size=shape),
        "proj_down": np.random.uniform(low=0.2, size=shape),
        "width": 0.05,
    }
    h5f["input/kpoints/number_kpoints"] = len(kpoints)
    h5f["results/eigenvalues/kpoint_coords"] = kpoints
    h5f["results/eigenvalues/eigenvalues"] = np.array([ref["up"], ref["down"]])
    h5f["results/projectors/par"] = np.array([[[ref["proj_up"]]], [[ref["proj_down"]]]])
    h5f["results/projectors/lchar"] = np.array(["s"], dtype="S")
    h5f["results/positions/ion_types"] = np.array(["Si"], dtype="S")
    h5f["results/positions/number_ion_types"] = [1]
    h5f["results/dos/efermi"] = 0.0
    h5f["results/positions/scale"] = 1.0
    h5f["results/positions/lattice_vectors"] = np.eye(3)
    return h5f, ref


def test_spin_band_structure_read(spin_band_structure):
    h5f, ref = spin_band_structure
    band = Band(h5f).read("s")
    assert_allclose(band["up"], ref["up"])
    assert_allclose(band["down"], ref["down"])
    assert_allclose(band["projections"]["s_up"], ref["proj_up"])
    assert_allclose(band["projections"]["s_down"], ref["proj_down"])


def test_spin_band_structure_plot(spin_band_structure):
    h5f, ref = spin_band_structure
    fig = Band(h5f).plot("Si", width=ref["width"])
    assert len(fig.data) == 2
    spins = ["up", "down"]
    for spin, data in zip(spins, fig.data):
        assert data.name == "Si_" + spin
        for band, weight in zip(np.nditer(ref[spin]), np.nditer(ref["proj_" + spin])):
            upper = band + ref["width"] * weight
            lower = band - ref["width"] * weight
            pos_upper = data.x[np.where(np.isclose(data.y, upper))]
            pos_lower = data.x[np.where(np.isclose(data.y, lower))]
            assert len(pos_upper) == len(pos_lower) == 1
            assert_allclose(pos_upper, pos_lower)


@pytest.fixture
def projected_band_structure():
    h5f = h5py.File(TemporaryFile(), "a")
    kpoints = np.linspace(np.zeros(3), np.ones(3))
    shape = (len(kpoints), 2)
    ref = {
        "bands": np.arange(np.prod(shape)).reshape(shape),
        "projections": np.random.uniform(low=0.2, size=shape),
        # set lower bound to avoid accidentally triggering np.isclose
        "width": 0.5,
    }
    h5f["input/kpoints/number_kpoints"] = len(kpoints)
    h5f["results/eigenvalues/kpoint_coords"] = kpoints
    h5f["results/eigenvalues/eigenvalues"] = np.array([ref["bands"]])
    h5f["results/projectors/par"] = np.array([[[ref["projections"]]]])
    h5f["results/projectors/lchar"] = np.array(["s"], dtype="S")
    h5f["results/positions/ion_types"] = np.array(["Si"], dtype="S")
    h5f["results/positions/number_ion_types"] = [1]
    h5f["results/dos/efermi"] = 0.0
    h5f["results/positions/scale"] = 1.0
    h5f["results/positions/lattice_vectors"] = np.eye(3)
    return h5f, ref


def test_projected_band_structure_read(projected_band_structure):
    h5f, ref = projected_band_structure
    band = Band(h5f).read("Si:s")
    assert_allclose(band["projections"]["Si_s"], ref["projections"])


def test_projected_band_structure_plot(projected_band_structure):
    h5f, ref = projected_band_structure
    fig = Band(h5f).plot("s, 1")
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
        for band, weight in zip(np.nditer(ref["bands"]), np.nditer(ref["projections"])):
            upper = band + ref["width"] * weight
            lower = band - ref["width"] * weight
            pos_upper = data.x[np.where(np.isclose(data.y, upper))]
            pos_lower = data.x[np.where(np.isclose(data.y, lower))]
            assert len(pos_upper) == len(pos_lower) == 1
            assert_allclose(pos_upper, pos_lower)

from py4vasp.data import Dos
import pytest
import h5py
import numpy as np
from numpy.testing import assert_array_almost_equal_nulp
from tempfile import TemporaryFile


def assert_allclose(actual, desired):
    assert_array_almost_equal_nulp(actual, desired, 10)


@pytest.fixture
def nonmagnetic_Dos():
    """ Setup a nonmagnetic Dos file containing all important quantities."""
    return _nonmagnetic_Dos()


def _nonmagnetic_Dos():
    ref = {"energies": np.linspace(-1, 3), "fermi_energy": 1.372}
    ref["dos"] = ref["energies"] ** 2
    h5f = h5py.File(TemporaryFile(), "a")
    h5f["results/dos/energies"] = ref["energies"]
    h5f["results/dos/dos"] = np.array([ref["dos"]])
    h5f["results/dos/efermi"] = ref["fermi_energy"]
    h5f["results/dos/jobpar"] = 0
    return h5f, ref


def test_nonmagnetic_Dos_read(nonmagnetic_Dos):
    h5f, ref = nonmagnetic_Dos
    dos = Dos(h5f).read()
    assert_allclose(dos["energies"], ref["energies"] - ref["fermi_energy"])
    assert_allclose(dos["total"], ref["dos"])
    assert dos["fermi_energy"] == ref["fermi_energy"]


def test_nonmagnetic_Dos_read_error(nonmagnetic_Dos):
    h5f, _ = nonmagnetic_Dos
    with pytest.raises(ValueError):
        Dos(h5f).read("s")


def test_nonmagnetic_Dos_to_frame(nonmagnetic_Dos):
    """ Test whether reading the nonmagnetic Dos yields the expected results."""
    h5f, ref = nonmagnetic_Dos
    dos = Dos(h5f).to_frame()
    assert_allclose(dos.energies, ref["energies"] - ref["fermi_energy"])
    assert_allclose(dos.total, ref["dos"])
    assert dos.fermi_energy == ref["fermi_energy"]


def test_nonmagnetic_Dos_plot(nonmagnetic_Dos):
    """ Test whether plotting the nonmagnetic Dos yields the expected results."""
    h5f, ref = nonmagnetic_Dos
    fig = Dos(h5f).plot()
    assert fig.layout.xaxis.title.text == "Energy (eV)"
    assert fig.layout.yaxis.title.text == "DOS (1/eV)"
    assert len(fig.data) == 1
    assert_allclose(fig.data[0].x, ref["energies"] - ref["fermi_energy"])
    assert_allclose(fig.data[0].y, ref["dos"])


@pytest.fixture
def magnetic_Dos():
    return _magnetic_Dos()


def _magnetic_Dos():
    """ Setup a magnetic Dos file containing all relevant quantities."""
    ref = {"energies": np.linspace(-2, 2), "fermi_energy": -0.137}
    ref["up"] = (ref["energies"] + 0.5) ** 2
    ref["down"] = (ref["energies"] - 0.5) ** 2
    h5f = h5py.File(TemporaryFile(), "a")
    h5f["results/dos/energies"] = ref["energies"]
    h5f["results/dos/dos"] = np.array([ref["up"], ref["down"]])
    h5f["results/dos/efermi"] = -0.137
    h5f["results/dos/jobpar"] = 0
    return h5f, ref


def test_magnetic_Dos_to_frame(magnetic_Dos):
    """ Test whether reading the magnetic Dos yields the expected results."""
    h5f, ref = magnetic_Dos
    dos = Dos(h5f).to_frame()
    assert_allclose(dos.energies, ref["energies"] - ref["fermi_energy"])
    assert_allclose(dos.up, ref["up"])
    assert_allclose(dos.down, ref["down"])
    assert dos.fermi_energy == ref["fermi_energy"]


def test_magnetic_Dos_plot(magnetic_Dos):
    """ Test whether plotting the magnetic Dos yields the expected results."""
    h5f, ref = magnetic_Dos
    fig = Dos(h5f).plot()
    assert len(fig.data) == 2
    assert_allclose(fig.data[0].x, fig.data[1].x)
    assert_allclose(fig.data[0].y, ref["up"])
    assert_allclose(fig.data[1].y, -ref["down"])


@pytest.fixture
def nonmagnetic_l_Dos():
    """ Setup a l resolved Dos file containing all relevant quantities."""
    h5f, ref = _nonmagnetic_Dos()
    ref["Si_s"] = np.sqrt(np.abs(ref["energies"]))
    ref["Si_p"] = ref["energies"] + 0.5
    ref["Si_d"] = ref["energies"] ** 2
    ref["C1_s"] = np.abs(ref["energies"])
    ref["C1_p"] = (1 - ref["energies"]) ** 2
    ref["C2_s"] = np.sqrt(np.abs(ref["energies"]) + 1)
    ref["C2_p"] = 1 - ref["energies"] ** 2
    num_spins = 1
    atoms = ["Si", "C1", "C2"]
    lmax = 4
    num_energy = len(ref["energies"])
    h5f["results/dos/jobpar"][()] = 1
    h5f["results/projectors/lchar"] = np.array([" s", " p", " d", " f"], dtype="S")
    h5f["results/positions/ion_types"] = np.array(["Si", "C "], dtype="S")
    h5f["results/positions/number_ion_types"] = [1, 2]
    h5f["results/dos/dospar"] = np.zeros((num_spins, len(atoms), lmax, num_energy))
    orbitals = ["s", "p", "d"]
    for iatom, atom in enumerate(atoms):
        for l, orbital in enumerate(orbitals):
            key = atom + "_" + orbital
            if key in ref:
                h5f["results/dos/dospar"][:, iatom, l] = ref[key]
    return h5f, ref


def test_nonmagnetic_l_Dos_to_frame(nonmagnetic_l_Dos):
    """ Test whether reading the nonmagnetic l resolved Dos yields the expected results."""
    h5f, ref = nonmagnetic_l_Dos
    equivalent_selections = [
        "s Si:d Si C:s,p 1:p 2 3:s",
        "1: p, C : s Si : d, *: s, 2 Si:* C: p 3 : s",
    ]
    for selection in equivalent_selections:
        dos = Dos(h5f).to_frame(selection)
        assert_allclose(dos.s, ref["Si_s"] + ref["C1_s"] + ref["C2_s"])
        assert_allclose(dos.Si, ref["Si_s"] + ref["Si_p"] + ref["Si_d"])
        assert_allclose(dos.Si_d, ref["Si_d"])
        assert_allclose(dos.Si_1_p, ref["Si_p"])
        assert_allclose(dos.C_s, ref["C1_s"] + ref["C2_s"])
        assert_allclose(dos.C_p, ref["C1_p"] + ref["C2_p"])
        assert_allclose(dos.C_1, ref["C1_s"] + ref["C1_p"])
        assert_allclose(dos.C_2_s, ref["C2_s"])


def test_nonmagnetic_l_Dos_plot(nonmagnetic_l_Dos):
    """ Test whether plotting the nonmagnetic l resolved Dos yields the expected results."""
    h5f, ref = nonmagnetic_l_Dos
    selection = "p 3 Si:d"
    fig = Dos(h5f).plot(selection)
    assert len(fig.data) == 4  # total Dos + 3 selections
    assert_allclose(fig.data[1].y, ref["Si_p"] + ref["C1_p"] + ref["C2_p"])
    assert_allclose(fig.data[2].y, ref["C2_s"] + ref["C2_p"])
    assert_allclose(fig.data[3].y, ref["Si_d"])


@pytest.fixture
def magnetic_lm_Dos():
    """ Setup a lm resolved Dos file containing all relevant quantities."""
    h5f, ref = _magnetic_Dos()
    num_spins = 2
    lm_size = 16
    num_energy = len(ref["energies"])
    sp_orbitals = ["    s", "   py", "   pz", "   px"]
    d_orbitals = ["  dxy", "  dyz", "  dz2", "  dxz", "x2-y2"]
    f_orbitals = ["fy3x2", " fxyz", " fyz2", "  fz3", " fxz2", " fzx2", "  fx3"]
    orbitals = sp_orbitals + d_orbitals + f_orbitals
    h5f["results/dos/jobpar"][()] = 1
    h5f["results/projectors/lchar"] = np.array(orbitals, dtype="S")
    h5f["results/positions/ion_types"] = np.array(["Fe"], dtype="S")
    h5f["results/positions/number_ion_types"] = [1]
    h5f["results/dos/dospar"] = np.zeros((num_spins, 1, lm_size, num_energy))
    for ispin, spin in enumerate(["up", "down"]):
        for lm, orbital in enumerate(orbitals):
            key = orbital.strip() + "_" + spin
            ref[key] = np.random.random(num_energy)
            h5f["results/dos/dospar"][ispin, :, lm] = ref[key]
    return h5f, ref


def test_magnetic_lm_Dos_read(magnetic_lm_Dos):
    """ Test whether reading lm resolved Dos works as expected."""
    h5f, ref = magnetic_lm_Dos
    dos = Dos(h5f).read("px p d f")
    assert_allclose(dos["px_up"], ref["px_up"])
    assert_allclose(dos["px_down"], ref["px_down"])
    assert_allclose(dos["p_up"], ref["px_up"] + ref["py_up"] + ref["pz_up"])
    d_down = ref["dxy_down"] + ref["dyz_down"] + ref["dz2_down"]
    d_down += ref["dxz_down"] + ref["x2-y2_down"]
    assert_allclose(dos["d_down"], d_down)
    f_up = ref["fy3x2_up"] + ref["fxyz_up"] + ref["fyz2_up"] + ref["fz3_up"]
    f_up += ref["fxz2_up"] + ref["fzx2_up"] + ref["fx3_up"]
    assert_allclose(dos["f_up"], f_up)


def test_magnetic_lm_Dos_plot(magnetic_lm_Dos):
    """ Test whether plotting lm resolved Dos works as expected."""
    h5f, ref = magnetic_lm_Dos
    fig = Dos(h5f).plot("dxz p")
    data = fig.data
    assert len(data) == 6  # spin resolved total + 2 selections
    names = [d.name for d in data]
    dxz_up = names.index("dxz_up")
    assert_allclose(data[dxz_up].y, ref["dxz_up"])
    p_down = names.index("p_down")
    assert_allclose(data[p_down].y, -(ref["px_down"] + ref["py_down"] + ref["pz_down"]))

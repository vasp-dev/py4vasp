from py4vasp.data import Dos, _util
from py4vasp.raw import RawDos, RawVersion, RawTopology, RawProjectors
from . import current_vasp_version
import py4vasp.exceptions as exception
import pytest
import numpy as np

num_energies = 50


@pytest.fixture
def nonmagnetic_Dos():
    """ Setup a nonmagnetic Dos containing all important quantities."""
    energies = np.linspace(-1, 3, num_energies)
    raw_dos = RawDos(
        version=current_vasp_version,
        fermi_energy=1.372,
        energies=energies,
        dos=np.array([energies ** 2]),
    )
    return raw_dos


def test_nonmagnetic_Dos_read(nonmagnetic_Dos, Assert):
    raw_dos = nonmagnetic_Dos
    dos = Dos(raw_dos).read()
    Assert.allclose(dos["energies"], raw_dos.energies - raw_dos.fermi_energy)
    Assert.allclose(dos["total"], raw_dos.dos[0])
    assert dos["fermi_energy"] == raw_dos.fermi_energy


def test_nonmagnetic_Dos_read_error(nonmagnetic_Dos):
    raw_dos = nonmagnetic_Dos
    with pytest.raises(exception.IncorrectUsage):
        Dos(raw_dos).read("s")


def test_nonmagnetic_Dos_to_frame(nonmagnetic_Dos, Assert):
    """ Test whether reading the nonmagnetic Dos yields the expected results."""
    raw_dos = nonmagnetic_Dos
    dos = Dos(raw_dos).to_frame()
    Assert.allclose(dos.energies, raw_dos.energies - raw_dos.fermi_energy)
    Assert.allclose(dos.total, raw_dos.dos[0])
    assert dos.fermi_energy == raw_dos.fermi_energy


def test_nonmagnetic_Dos_plot(nonmagnetic_Dos, Assert):
    """ Test whether plotting the nonmagnetic Dos yields the expected results."""
    raw_dos = nonmagnetic_Dos
    fig = Dos(raw_dos).plot()
    assert fig.layout.xaxis.title.text == "Energy (eV)"
    assert fig.layout.yaxis.title.text == "DOS (1/eV)"
    assert len(fig.data) == 1
    Assert.allclose(fig.data[0].x, raw_dos.energies - raw_dos.fermi_energy)
    Assert.allclose(fig.data[0].y, raw_dos.dos[0])


def test_nonmagnetic_Dos_from_file(nonmagnetic_Dos, mock_file, check_read):
    with mock_file("dos", nonmagnetic_Dos) as mocks:
        check_read(Dos, mocks, nonmagnetic_Dos)


def test_print_nonmagnetic_dos(nonmagnetic_Dos):
    actual, _ = _util.format_(Dos(nonmagnetic_Dos))
    reference = f"""
Dos:
   energies: [-1.00, 3.00] {num_energies} points
    """.strip()
    assert actual == {"text/plain": reference}


@pytest.fixture
def magnetic_Dos():
    """ Setup a magnetic Dos containing all relevant quantities."""
    energies = np.linspace(-2, 2, num_energies)
    raw_dos = RawDos(
        version=current_vasp_version,
        fermi_energy=-0.137,
        energies=energies,
        dos=np.array(((energies + 0.5) ** 2, (energies - 0.5) ** 2)),
    )
    return raw_dos


def test_magnetic_Dos_to_frame(magnetic_Dos, Assert):
    """ Test whether reading the magnetic Dos yields the expected results."""
    raw_dos = magnetic_Dos
    dos = Dos(raw_dos).to_frame()
    Assert.allclose(dos.energies, raw_dos.energies - raw_dos.fermi_energy)
    Assert.allclose(dos.up, raw_dos.dos[0])
    Assert.allclose(dos.down, raw_dos.dos[1])
    assert dos.fermi_energy == raw_dos.fermi_energy


def test_magnetic_Dos_plot(magnetic_Dos, Assert):
    """ Test whether plotting the magnetic Dos yields the expected results."""
    raw_dos = magnetic_Dos
    fig = Dos(raw_dos).plot()
    assert len(fig.data) == 2
    Assert.allclose(fig.data[0].x, fig.data[1].x)
    Assert.allclose(fig.data[0].y, raw_dos.dos[0])
    Assert.allclose(fig.data[1].y, -raw_dos.dos[1])


def test_print_magnetic_dos(magnetic_Dos):
    actual, _ = _util.format_(Dos(magnetic_Dos))
    reference = f"""
spin polarized Dos:
   energies: [-2.00, 2.00] {num_energies} points
    """.strip()
    assert actual == {"text/plain": reference}


@pytest.fixture
def nonmagnetic_projections(nonmagnetic_Dos):
    """ Setup a l resolved Dos containing all relevant quantities."""
    ref = {
        "Si_s": np.random.random(num_energies),
        "Si_p": np.random.random(num_energies),
        "Si_d": np.random.random(num_energies),
        "C1_s": np.random.random(num_energies),
        "C1_p": np.random.random(num_energies),
        "C2_s": np.random.random(num_energies),
        "C2_p": np.random.random(num_energies),
    }
    num_spins = 1
    atoms = ["Si", "C1", "C2"]
    lmax = 4
    raw_proj = RawProjectors(
        version=current_vasp_version,
        topology=RawTopology(
            version=current_vasp_version,
            number_ion_types=[1, 2],
            ion_types=np.array(["Si", "C "], dtype="S"),
        ),
        orbital_types=np.array([" s", " p", " d", " f"], dtype="S"),
        number_spins=num_spins,
    )
    nonmagnetic_Dos.projectors = raw_proj
    nonmagnetic_Dos.projections = np.zeros((num_spins, len(atoms), lmax, num_energies))
    orbitals = ["s", "p", "d"]
    for iatom, atom in enumerate(atoms):
        for l, orbital in enumerate(orbitals):
            key = atom + "_" + orbital
            if key in ref:
                nonmagnetic_Dos.projections[:, iatom, l] = ref[key]
    return nonmagnetic_Dos, ref


def test_nonmagnetic_l_Dos_to_frame(nonmagnetic_projections, Assert):
    """ Test whether reading the nonmagnetic l resolved Dos yields the expected results."""
    raw_dos, ref = nonmagnetic_projections
    equivalent_selections = [
        "s Si(d) Si C(s,p) 1(p) 2 3(s)",
        "1( p), C(s) Si(d), *(s), 2 Si(*) p(C) s(3)",
    ]
    for selection in equivalent_selections:
        dos = Dos(raw_dos).to_frame(selection)
        Assert.allclose(dos.s, ref["Si_s"] + ref["C1_s"] + ref["C2_s"])
        Assert.allclose(dos.Si, ref["Si_s"] + ref["Si_p"] + ref["Si_d"])
        Assert.allclose(dos.Si_d, ref["Si_d"])
        Assert.allclose(dos.Si_1_p, ref["Si_p"])
        Assert.allclose(dos.C_s, ref["C1_s"] + ref["C2_s"])
        Assert.allclose(dos.C_p, ref["C1_p"] + ref["C2_p"])
        Assert.allclose(dos.C_1, ref["C1_s"] + ref["C1_p"])
        Assert.allclose(dos.C_2_s, ref["C2_s"])


def test_nonmagnetic_l_Dos_plot(nonmagnetic_projections, Assert):
    """ Test whether plotting the nonmagnetic l resolved Dos yields the expected results."""
    raw_dos, ref = nonmagnetic_projections
    selection = "p 3 Si(d)"
    fig = Dos(raw_dos).plot(selection)
    assert len(fig.data) == 4  # total Dos + 3 selections
    Assert.allclose(fig.data[1].y, ref["Si_p"] + ref["C1_p"] + ref["C2_p"])
    Assert.allclose(fig.data[2].y, ref["C2_s"] + ref["C2_p"])
    Assert.allclose(fig.data[3].y, ref["Si_d"])


def test_more_projections_style(nonmagnetic_projections, Assert):
    """Vasp 6.1 may store more orbital types then projections available. This
    test checks whether that leads to any issues"""
    raw_dos, ref = nonmagnetic_projections
    shape = raw_dos.projections.shape
    shape = (shape[0], shape[1], shape[2] - 1, shape[3])
    raw_dos.projections = np.random.uniform(low=0.2, size=shape)
    dos = Dos(raw_dos).read("Si")


def test_print_nonmagnetic_projections(nonmagnetic_projections):
    raw_dos, _ = nonmagnetic_projections
    actual, _ = _util.format_(Dos(raw_dos))
    reference = f"""
Dos:
   energies: [-1.00, 3.00] {num_energies} points
projectors:
   atoms: Si, C
   orbitals: s, p, d, f
    """.strip()
    assert actual == {"text/plain": reference}


@pytest.fixture
def magnetic_projections(magnetic_Dos):
    """ Setup a lm resolved Dos containing all relevant quantities."""
    num_spins = 2
    lm_size = 16
    sp_orbitals = ["    s", "   py", "   pz", "   px"]
    d_orbitals = ["  dxy", "  dyz", "  dz2", "  dxz", "x2-y2"]
    f_orbitals = ["fy3x2", " fxyz", " fyz2", "  fz3", " fxz2", " fzx2", "  fx3"]
    orbitals = sp_orbitals + d_orbitals + f_orbitals
    raw_proj = RawProjectors(
        version=current_vasp_version,
        topology=RawTopology(
            version=current_vasp_version,
            number_ion_types=[1],
            ion_types=np.array(["Fe"], dtype="S"),
        ),
        orbital_types=np.array(orbitals, dtype="S"),
        number_spins=num_spins,
    )
    magnetic_Dos.projectors = raw_proj
    magnetic_Dos.projections = np.zeros((num_spins, 1, lm_size, num_energies))
    ref = {}
    for ispin, spin in enumerate(["up", "down"]):
        for lm, orbital in enumerate(orbitals):
            key = orbital.strip() + "_" + spin
            ref[key] = np.random.random(num_energies)
            magnetic_Dos.projections[ispin, :, lm] = ref[key]
    return magnetic_Dos, ref


def test_magnetic_lm_Dos_read(magnetic_projections, Assert):
    """ Test whether reading lm resolved Dos works as expected."""
    raw_dos, ref = magnetic_projections
    dos = Dos(raw_dos).read("px p d f")
    Assert.allclose(dos["px_up"], ref["px_up"])
    Assert.allclose(dos["px_down"], ref["px_down"])
    Assert.allclose(dos["p_up"], ref["px_up"] + ref["py_up"] + ref["pz_up"])
    d_down = ref["dxy_down"] + ref["dyz_down"] + ref["dz2_down"]
    d_down += ref["dxz_down"] + ref["x2-y2_down"]
    Assert.allclose(dos["d_down"], d_down)
    f_up = ref["fy3x2_up"] + ref["fxyz_up"] + ref["fyz2_up"] + ref["fz3_up"]
    f_up += ref["fxz2_up"] + ref["fzx2_up"] + ref["fx3_up"]
    Assert.allclose(dos["f_up"], f_up)


def test_magnetic_lm_Dos_plot(magnetic_Dos, magnetic_projections, Assert):
    """ Test whether plotting lm resolved Dos works as expected."""
    raw_dos, ref = magnetic_projections
    fig = Dos(raw_dos).plot("dxz p")
    data = fig.data
    assert len(data) == 6  # spin resolved total + 2 selections
    names = [d.name for d in data]
    dxz_up = names.index("dxz_up")
    Assert.allclose(data[dxz_up].y, ref["dxz_up"])
    p_down = names.index("p_down")
    Assert.allclose(data[p_down].y, -(ref["px_down"] + ref["py_down"] + ref["pz_down"]))


def test_nonexisting_dos():
    with pytest.raises(exception.NoData):
        dos = Dos(None)


def test_version(nonmagnetic_Dos):
    nonmagnetic_Dos.version = RawVersion(_util._minimal_vasp_version.major - 1)
    with pytest.raises(exception.OutdatedVaspVersion):
        Dos(nonmagnetic_Dos)

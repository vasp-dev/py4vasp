# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import re
import types

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._calculation._stoichiometry import Stoichiometry
from py4vasp._calculation.structure import Structure

REF_POSCAR = """\
Sr2TiO4
6.9229000000000003
   1.0000000000000000    0.0000000000000000    0.0000000000000000
   0.6781122097386930    0.7349583872510080    0.0000000000000000
  -0.8390553410420490   -0.3674788590908430    0.4011800378743010
Sr Ti O
2 1 4
Direct
   0.6452900000000000    0.6452900000000000    0.0000000000000000
   0.3547100000000000    0.3547100000000000    0.0000000000000000
   0.0000000000000000    0.0000000000000000    0.0000000000000000
   0.8417800000000000    0.8417800000000000    0.0000000000000000
   0.1582300000000000    0.1582300000000000    0.0000000000000000
   0.5000000000000000    0.0000000000000000    0.5000000000000000
   0.0000000000000000    0.5000000000000000    0.5000000000000000"""

REF_HTML = """\
Sr2TiO4<br>
6.9229000000000003<br>
<table>
<tr><td>   1.0000000000000000</td><td>   0.0000000000000000</td><td>   0.0000000000000000</td></tr>
<tr><td>   0.6781122097386930</td><td>   0.7349583872510080</td><td>   0.0000000000000000</td></tr>
<tr><td>  -0.8390553410420490</td><td>  -0.3674788590908430</td><td>   0.4011800378743010</td></tr>
</table>
Sr Ti O<br>
2 1 4<br>
Direct<br>
<table>
<tr><td>   0.6452900000000000</td><td>   0.6452900000000000</td><td>   0.0000000000000000</td></tr>
<tr><td>   0.3547100000000000</td><td>   0.3547100000000000</td><td>   0.0000000000000000</td></tr>
<tr><td>   0.0000000000000000</td><td>   0.0000000000000000</td><td>   0.0000000000000000</td></tr>
<tr><td>   0.8417800000000000</td><td>   0.8417800000000000</td><td>   0.0000000000000000</td></tr>
<tr><td>   0.1582300000000000</td><td>   0.1582300000000000</td><td>   0.0000000000000000</td></tr>
<tr><td>   0.5000000000000000</td><td>   0.0000000000000000</td><td>   0.5000000000000000</td></tr>
<tr><td>   0.0000000000000000</td><td>   0.5000000000000000</td><td>   0.5000000000000000</td></tr>
</table>"""

REF_LAMMPS = r"""Configuration 1: system "Sr2TiO4"

7 atoms
3 atom types

0.0   6.9229000000000003E\+00 xlo xhi
0.0   5.0880434191000035E\+00 ylo yhi
0.0   2.7773292841999986E\+00 zlo zhi
  4.6945030167999979E\+00  -5.8086962205000017E\+00  -2.5440193935999971E\+00 xy xz yz

Atoms # atomic

1 1   7.49659399271087\d\dE\+00   3.28326353791104\d\dE\+00   0.0000000000000000E\+00
2 1   4.12080902408912\d\dE\+00   1.80477988118896\d\dE\+00   0.0000000000000000E\+00
3 2   0.00000000000000\d\dE\+00   0.00000000000000\d\dE\+00   0.0000000000000000E\+00
4 3   9.77929751148190\d\dE\+00   4.28301318933000\d\dE\+00   0.0000000000000000E\+00
5 3   1.83822167934826\d\dE\+00   8.05081110204193\d\dE\-01   0.0000000000000000E\+00
6 3   5.57101889749999\d\dE\-01  -1.27200969679999\d\dE\+00   1.3886646420999993E\+00
7 3  -5.57096601850001\d\dE\-01   1.27201201275000\d\dE\+00   1.3886646420999993E\+00"""

REF_LAMMPS_ZnS = r"""Configuration 1: system "Zn2S2"

4 atoms
2 atom types

0.0   3.8078865529319543E\+00 xlo xhi
0.0   3.2931653361218416E\+00 ylo yhi
0.0   6.2000000000000002E\+00 zlo zhi
 -1.9118216624375599E\+00  -0.0000000000000000E\+00   0.0000000000000000E\+00 xy xz yz

Atoms # atomic

1 1  -5.25225731438871\d\dE\-03   2.19544355741456\d\dE\+00   0.0000000000000000E\+00
2 1   1.90131714780878\d\dE\+00   1.09772177870728\d\dE\+00   3.1000000000000001E\+00
3 2  -5.25225731438871\d\dE\-03   2.19544355741456\d\dE\+00   2.3250000000000002E\+00
4 2   1.90131714780878\d\dE\+00   1.09772177870728\d\dE\+00   5.4249999999999998E\+00"""

REF_LAMMPS_ZnS_general = """Configuration 1: system "Zn2S2"

4 atoms
2 atom types

  1.8999999999999999E+00  -3.2999999999999998E+00   0.0000000000000000E+00 avec
  1.8999999999999999E+00   3.2999999999999998E+00   0.0000000000000000E+00 bvec
  0.0000000000000000E+00   0.0000000000000000E+00   6.2000000000000002E+00 cvec
0.0 0.0 0.0 abc origin

Atoms # atomic

1 1   1.8999999999999999E+00   1.0999999999999999E+00   0.0000000000000000E+00
2 1   1.8999999999999999E+00  -1.0999999999999999E+00   3.1000000000000001E+00
3 2   1.8999999999999999E+00   1.0999999999999999E+00   2.3250000000000002E+00
4 2   1.8999999999999999E+00  -1.0999999999999999E+00   5.4249999999999998E+00"""

REF_Ca3AsBr3 = """Ca3AsBr3
5.9299999999999997
   1.0000000000000000    0.0000000000000000    0.0000000000000000
   0.0000000000000000    1.0000000000000000    0.0000000000000000
   0.0000000000000000    0.0000000000000000    1.0000000000000000
Ca As Br Ca Br
2 1 1 1 2
Direct
   0.5000000000000000    0.0000000000000000    0.0000000000000000
   0.0000000000000000    0.5000000000000000    0.0000000000000000
   0.0000000000000000    0.0000000000000000    0.0000000000000000
   0.0000000000000000    0.5000000000000000    0.5000000000000000
   0.0000000000000000    0.0000000000000000    0.5000000000000000
   0.5000000000000000    0.0000000000000000    0.5000000000000000
   0.5000000000000000    0.5000000000000000    0.0000000000000000"""


@pytest.fixture
def Sr2TiO4(raw_data):
    return make_structure(raw_data.structure("Sr2TiO4"))


@pytest.fixture
def Fe3O4(raw_data):
    return make_structure(raw_data.structure("Fe3O4"))


@pytest.fixture
def Ca3AsBr3(raw_data):
    return make_structure(raw_data.structure("Ca3AsBr3"))


@pytest.fixture
def ZnS(raw_data):
    return make_structure(raw_data.structure("ZnS"))


@pytest.fixture(params=[None, 2, (3, 2, 1)])
def supercell(request):
    return request.param


@pytest.fixture(params=["foo", (1, 2), (2.4, 1.1, 3.5)])
def not_a_supercell(request):
    return request.param


def make_structure(raw_structure):
    structure = Structure.from_data(raw_structure)
    structure.ref = types.SimpleNamespace()
    if not raw_structure.cell.scale.is_none():
        scale = raw_structure.cell.scale[()]
    else:
        scale = 1.0
    structure.ref.lattice_vectors = scale * raw_structure.cell.lattice_vectors
    structure.ref.positions = raw_structure.positions
    stoichiometry = Stoichiometry.from_data(raw_structure.stoichiometry)
    structure.ref.elements = stoichiometry.elements()
    return structure


def test_read_Sr2TiO4(Sr2TiO4, Assert):
    check_Sr2TiO4_structure(Sr2TiO4.read(), Sr2TiO4.ref, -1, Assert)
    for steps in (slice(None), slice(1, 3), 0):
        check_Sr2TiO4_structure(Sr2TiO4[steps].read(), Sr2TiO4.ref, steps, Assert)


def check_Sr2TiO4_structure(actual, reference, steps, Assert):
    Assert.allclose(actual["lattice_vectors"], reference.lattice_vectors[steps])
    Assert.allclose(actual["positions"], reference.positions[steps])
    assert actual["elements"] == reference.elements
    assert actual["names"] == ["Sr_1", "Sr_2", "Ti_1", "O_1", "O_2", "O_3", "O_4"]


def test_read_Fe3O4(Fe3O4, Assert):
    check_Fe3O4_structure(Fe3O4.read(), Fe3O4.ref, -1, Assert)
    for steps in (slice(None), slice(1, 3), 0):
        check_Fe3O4_structure(Fe3O4[steps].read(), Fe3O4.ref, steps, Assert)


def check_Fe3O4_structure(actual, reference, steps, Assert):
    Assert.allclose(actual["lattice_vectors"], reference.lattice_vectors[steps])
    Assert.allclose(actual["positions"], reference.positions[steps])
    assert actual["elements"] == reference.elements
    assert actual["names"] == ["Fe_1", "Fe_2", "Fe_3", "O_1", "O_2", "O_3", "O_4"]


def test_read_Ca3AsBr3(Ca3AsBr3, Assert):
    # special case of single structure instead of trajectory
    actual = Ca3AsBr3.read()
    Assert.allclose(actual["lattice_vectors"], Ca3AsBr3.ref.lattice_vectors)
    Assert.allclose(actual["positions"], Ca3AsBr3.ref.positions)
    assert actual["elements"] == Ca3AsBr3.ref.elements
    assert actual["names"] == ["Ca_1", "Ca_2", "As_1", "Br_1", "Ca_3", "Br_2", "Br_3"]


def test_to_poscar(Sr2TiO4, Ca3AsBr3):
    assert Sr2TiO4.to_POSCAR() == REF_POSCAR
    assert Sr2TiO4[0].to_POSCAR() == REF_POSCAR.replace("Sr2TiO4", "Sr2TiO4 (step 1)")
    for steps in (slice(None), slice(1, 3)):
        with pytest.raises(exception.NotImplemented):
            Sr2TiO4[steps].to_POSCAR()
    assert Ca3AsBr3.to_POSCAR() == REF_Ca3AsBr3


def test_from_poscar(Sr2TiO4, Assert, not_core):
    structure = Structure.from_POSCAR(REF_POSCAR)
    check_Sr2TiO4_structure(structure.read(), Sr2TiO4.ref, -1, Assert)
    elements = ["Ba", "Zr", "S"]
    structure = Structure.from_POSCAR(REF_POSCAR, elements=elements)
    actual = structure.read()
    Assert.allclose(actual["lattice_vectors"], Sr2TiO4.ref.lattice_vectors[-1])
    Assert.allclose(actual["positions"], Sr2TiO4.ref.positions[-1])
    assert actual["elements"] == ["Ba", "Ba", "Zr", "S", "S", "S", "S"]
    assert actual["names"] == ["Ba_1", "Ba_2", "Zr_1", "S_1", "S_2", "S_3", "S_4"]


def test_from_poscar_without_elements(Sr2TiO4, Assert, not_core):
    poscar = """\
POSCAR without elements
1.0
   6.9229000000000003    0.0000000000000000    0.0000000000000000
   4.6945030167999979    5.0880434191000035    0.0000000000000000
  -5.8086962205000017   -2.5440193935999971    2.7773292841999986
2 1 4
Direct
   0.6452900000000000    0.6452900000000000    0.0000000000000000
   0.3547100000000000    0.3547100000000000    0.0000000000000000
   0.0000000000000000    0.0000000000000000    0.0000000000000000
   0.8417800000000000    0.8417800000000000    0.0000000000000000
   0.1582300000000000    0.1582300000000000    0.0000000000000000
   0.5000000000000000    0.0000000000000000    0.5000000000000000
   0.0000000000000000    0.5000000000000000    0.5000000000000000"""
    with pytest.raises(exception.IncorrectUsage):
        structure = Structure.from_POSCAR(poscar)
    structure = Structure.from_POSCAR(poscar, elements=["Sr", "Ti", "O"])
    check_Sr2TiO4_structure(structure.read(), Sr2TiO4.ref, -1, Assert)


def test_to_ase_Sr2TiO4(Sr2TiO4, Assert, not_core):
    check_Sr2TiO4_ase(Sr2TiO4.to_ase(), Sr2TiO4.ref, -1, Assert)
    check_Sr2TiO4_ase(Sr2TiO4[0].to_ase(), Sr2TiO4.ref, 0, Assert)
    for steps in (slice(None), slice(1, 3)):
        with pytest.raises(exception.NotImplemented):
            Sr2TiO4[steps].to_ase()


def check_Sr2TiO4_ase(structure, reference, steps, Assert):
    Assert.allclose(structure.cell.array, reference.lattice_vectors[steps])
    Assert.allclose(structure.get_scaled_positions(), reference.positions[steps])
    assert all(structure.symbols == "Sr2TiO4")
    assert all(structure.pbc)


def test_to_ase_Fe3O4(Fe3O4, Assert, not_core):
    check_Fe3O4_ase(Fe3O4.to_ase(), Fe3O4.ref, -1, Assert)
    check_Fe3O4_ase(Fe3O4[0].to_ase(), Fe3O4.ref, 0, Assert)


def check_Fe3O4_ase(structure, reference, steps, Assert):
    Assert.allclose(structure.cell.array, reference.lattice_vectors[steps])
    ref_positions = np.mod(reference.positions[steps], 1)
    Assert.allclose(structure.get_scaled_positions(), ref_positions)
    assert all(structure.symbols == "Fe3O4")
    assert all(structure.pbc)


def test_to_ase_Ca3AsBr3(Ca3AsBr3, Assert, not_core):
    structure = Ca3AsBr3.to_ase()
    Assert.allclose(structure.cell.array, Ca3AsBr3.ref.lattice_vectors)
    Assert.allclose(structure.get_scaled_positions(), Ca3AsBr3.ref.positions)
    assert all(structure.symbols == "Ca2AsBrCaBr2")
    assert all(structure.pbc)


def test_from_ase(Sr2TiO4, Assert, not_core):
    structure = Structure.from_ase(Sr2TiO4.to_ase())
    check_Sr2TiO4_structure(structure.read(), Sr2TiO4.ref, -1, Assert)


def test_to_mdtraj(Sr2TiO4, Assert, not_core):
    for steps in (slice(None), slice(1, 3)):
        trajectory = Sr2TiO4[steps].to_mdtraj()
        check_Sr2TiO4_mdtraj(trajectory, Sr2TiO4.ref, steps, Assert)
    with pytest.raises(exception.NotImplemented):
        Sr2TiO4[0].to_mdtraj()
    with pytest.raises(exception.NotImplemented):
        Sr2TiO4.to_mdtraj()


def check_Sr2TiO4_mdtraj(trajectory, reference, steps, Assert):
    assert trajectory.n_frames == len(reference.positions[steps])
    assert trajectory.n_atoms == 7
    unitcell_vectors = Structure.A_to_nm * reference.lattice_vectors[steps]
    cartesian_positions = [
        pos @ cell for pos, cell in zip(reference.positions[steps], unitcell_vectors)
    ]
    Assert.allclose(trajectory.xyz, np.array(cartesian_positions).astype(np.float32))
    Assert.allclose(trajectory.unitcell_vectors, unitcell_vectors)


def test_supercell_scale_all(Sr2TiO4, Assert, not_core):
    number_atoms = 7
    scale = 2
    supercell = Sr2TiO4.to_ase(supercell=scale)
    assert len(supercell) == number_atoms * scale**3
    Assert.allclose(supercell.cell.array, scale * Sr2TiO4.ref.lattice_vectors)
    assert list(supercell.symbols) == 16 * ["Sr"] + 8 * ["Ti"] + 32 * ["O"]


def test_supercell_scale_individual(Sr2TiO4, Assert, not_core):
    number_atoms = 7
    scale = (2, 1, 3)
    supercell = Sr2TiO4.to_ase(supercell=scale)
    assert len(supercell) == number_atoms * np.prod(scale)
    Assert.allclose(supercell.cell.array, np.diag(scale) @ Sr2TiO4.ref.lattice_vectors)


def test_supercell_wrong_size(Sr2TiO4, not_a_supercell, not_core):
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4.to_ase(not_a_supercell)


@pytest.mark.parametrize("steps", [-1, 0, slice(1, 3), slice(None)])
def test_lattice_vectors(Sr2TiO4, steps, Assert):
    structure = Sr2TiO4 if steps == -1 else Sr2TiO4[steps]
    Assert.allclose(structure.lattice_vectors(), Sr2TiO4.ref.lattice_vectors[steps])


@pytest.mark.parametrize("steps", [-1, 0, slice(1, 3), slice(None)])
def test_positions(Sr2TiO4, steps, Assert):
    structure = Sr2TiO4 if steps == -1 else Sr2TiO4[steps]
    Assert.allclose(structure.positions(), Sr2TiO4.ref.positions[steps])


def test_cartesian_positions(Sr2TiO4, Fe3O4, Ca3AsBr3, Assert, not_core):
    check_cartesian_positions(Sr2TiO4, Assert)
    check_cartesian_positions(Fe3O4, Assert)
    check_cartesian_positions(Fe3O4[0], Assert)
    check_cartesian_positions(Ca3AsBr3, Assert)


def check_cartesian_positions(structure, Assert):
    Assert.allclose(structure.cartesian_positions(), structure.to_ase().get_positions())


def test_volume_Fe3O4(Fe3O4, Assert):
    reference_volumes = determine_reference_volumes(Fe3O4.ref.lattice_vectors)
    Assert.allclose(Fe3O4.volume(), reference_volumes[-1])
    Assert.allclose(Fe3O4[0].volume(), reference_volumes[0])
    Assert.allclose(Fe3O4[:].volume(), reference_volumes)
    Assert.allclose(Fe3O4[1:3].volume(), reference_volumes[1:3])


def test_volume_Ca3AsO3(Ca3AsBr3, Assert):
    lattice_vectors = Ca3AsBr3.ref.lattice_vectors[np.newaxis]
    reference_volumes = determine_reference_volumes(lattice_vectors)[-1]
    Assert.allclose(Ca3AsBr3.volume(), reference_volumes)


def determine_reference_volumes(lattice_vectors):
    cross_product = np.cross(lattice_vectors[:, 0], lattice_vectors[:, 1])
    return np.abs(np.einsum("ij,ij -> i", cross_product, lattice_vectors[:, 2]))


def test_number_atoms(Sr2TiO4, Ca3AsBr3):
    assert Sr2TiO4.number_atoms() == 7
    assert Ca3AsBr3.number_atoms() == 7


def test_number_steps(Sr2TiO4, Ca3AsBr3):
    assert Sr2TiO4.number_steps() == 1
    assert Sr2TiO4[0].number_steps() == 1
    assert Sr2TiO4[:].number_steps() == len(Sr2TiO4.ref.positions)
    assert Sr2TiO4[1:3].number_steps() == 2
    assert Ca3AsBr3.number_steps() == 1


def test_plot_Sr2TiO4(Sr2TiO4, supercell, Assert):
    check_plot_structure(Sr2TiO4, -1, Assert, supercell)
    for steps in (slice(None), slice(1, 3), 0):
        check_plot_structure(Sr2TiO4[steps], steps, Assert, supercell)


def test_plot_Fe3O4(Fe3O4, Assert):
    check_plot_structure(Fe3O4, -1, Assert)
    for steps in (slice(None), slice(1, 3), 0):
        check_plot_structure(Fe3O4[steps], steps, Assert)


def test_plot_Ca3AsBr3(Ca3AsBr3, Assert):
    check_plot_structure(Ca3AsBr3, slice(None), Assert)


def check_plot_structure(structure, steps, Assert, supercell=None):
    view = structure.plot(supercell) if supercell else structure.plot()
    assert view.elements.ndim == 2
    assert np.all(structure.ref.elements == view.elements)
    assert view.positions.ndim == 3
    Assert.allclose(structure.ref.positions[steps], view.positions)
    assert view.lattice_vectors.ndim == 3
    Assert.allclose(structure.ref.lattice_vectors[steps], view.lattice_vectors)
    if supercell is None:
        supercell = (1, 1, 1)
    if np.atleast_1d(supercell).size == 1:
        supercell = np.full(3, supercell)
    assert view.supercell.dtype == np.int_
    Assert.allclose(view.supercell, supercell)


def test_plot_not_a_supercell(Sr2TiO4, not_a_supercell):
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4.plot(not_a_supercell)


def test_incorrect_step(Sr2TiO4, Ca3AsBr3):
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4[100].read()
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4[[0, 1]].read()
    with pytest.raises(exception.IncorrectUsage):
        Ca3AsBr3[0]


def test_Sr2TiO4_to_lammps(Sr2TiO4, not_core):
    assert re.match(REF_LAMMPS, Sr2TiO4.to_lammps())


def test_ZnS_to_lammps(ZnS, not_core):
    assert re.match(REF_LAMMPS_ZnS, ZnS.to_lammps())
    assert ZnS.to_lammps(standard_form=False) == REF_LAMMPS_ZnS_general


def test_print_final(Sr2TiO4, format_):
    actual, _ = format_(Sr2TiO4)
    assert actual == {"text/plain": REF_POSCAR, "text/html": REF_HTML}


def test_print_specific(Sr2TiO4, format_):
    actual, _ = format_(Sr2TiO4[0])
    ref_plain = REF_POSCAR.replace("Sr2TiO4", "Sr2TiO4 (step 1)")
    ref_html = REF_HTML.replace("Sr2TiO4", "Sr2TiO4 (step 1)")
    assert actual["text/plain"] == ref_plain
    assert actual == {"text/plain": ref_plain, "text/html": ref_html}


def test_print_trajectory(Sr2TiO4, format_):
    actual, _ = format_(Sr2TiO4[1:3])
    ref_plain = REF_POSCAR.replace("Sr2TiO4", "Sr2TiO4 from step 2 to 4")
    ref_html = REF_HTML.replace("Sr2TiO4", "Sr2TiO4 from step 2 to 4")
    assert actual == {"text/plain": ref_plain, "text/html": ref_html}


def test_print_Ca3AsBr3(Ca3AsBr3, format_):
    actual, _ = format_(Ca3AsBr3)
    assert actual["text/plain"] == REF_Ca3AsBr3


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.structure("Sr2TiO4")
    parameters = {"__getitem__": {"steps": slice(None)}}
    check_factory_methods(Structure, data, parameters)

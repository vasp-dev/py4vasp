from unittest.mock import patch
from py4vasp.control import POSCAR
from py4vasp.data import Structure, Magnetism, Topology
import py4vasp.data as data
import py4vasp.exceptions as exception
import pytest
import numpy as np
import types

REF_POSCAR = """
Sr2TiO4
1.0
6.9229 0.0 0.0
4.694503016799998 5.0880434191000035 0.0
-5.808696220500002 -2.544019393599997 2.7773292841999986
Sr Ti O
2 1 4
Direct
0.64529 0.64529 0.0
0.35471 0.35471 0.0
0.0 0.0 0.0
0.84178 0.84178 0.0
0.15823 0.15823 0.0
0.5 0.0 0.5
0.0 0.5 0.5
""".strip()

REF_HTML = """
Sr2TiO4<br>
1.0<br>
<table>
<tr><td>6.9229</td><td>0.0</td><td>0.0</td></tr>
<tr><td>4.694503016799998</td><td>5.0880434191000035</td><td>0.0</td></tr>
<tr><td>-5.808696220500002</td><td>-2.544019393599997</td><td>2.7773292841999986</td></tr>
</table>
Sr Ti O<br>
2 1 4<br>
Direct<br>
<table>
<tr><td>0.64529</td><td>0.64529</td><td>0.0</td></tr>
<tr><td>0.35471</td><td>0.35471</td><td>0.0</td></tr>
<tr><td>0.0</td><td>0.0</td><td>0.0</td></tr>
<tr><td>0.84178</td><td>0.84178</td><td>0.0</td></tr>
<tr><td>0.15823</td><td>0.15823</td><td>0.0</td></tr>
<tr><td>0.5</td><td>0.0</td><td>0.5</td></tr>
<tr><td>0.0</td><td>0.5</td><td>0.5</td></tr>
</table>""".strip()


@pytest.fixture
def Sr2TiO4(raw_data):
    raw_structure = raw_data.structure("Sr2TiO4")
    structure = Structure(raw_structure)
    structure.ref = types.SimpleNamespace()
    structure.ref.lattice_vectors = raw_structure.cell.lattice_vectors
    structure.ref.positions = raw_structure.positions
    return structure


@pytest.fixture
def Fe3O4_collinear(raw_data):
    raw_structure = raw_data.structure("Fe3O4 collinear")
    structure = Structure(raw_structure)
    structure.ref = types.SimpleNamespace()
    structure.ref.moments = np.sum(raw_structure.magnetism.moments[:, 1], axis=2)
    structure.ref.lattice_vectors = raw_structure.cell.lattice_vectors
    structure.ref.positions = np.mod(raw_structure.positions, 1)
    return structure


@pytest.fixture
def Fe3O4_noncollinear(raw_data):
    raw_structure = raw_data.structure("Fe3O4 noncollinear")
    structure = Structure(raw_structure)
    structure.ref = types.SimpleNamespace()
    structure.ref.moments = Magnetism(raw_structure.magnetism)[:].total_moments()
    return structure


@pytest.fixture
def Fe3O4_charge_only(raw_data):
    return Structure(raw_data.structure("Fe3O4 charge_only"))


@pytest.fixture
def Fe3O4_zero_moments(raw_data):
    return Structure(raw_data.structure("Fe3O4 zero_moments"))


def test_read_Sr2TiO4(Sr2TiO4, Assert):
    check_Sr2TiO4_structure(Sr2TiO4.read(), Sr2TiO4.ref, -1, Assert)
    for steps in (slice(None), slice(1, 3), 0):
        check_Sr2TiO4_structure(Sr2TiO4[steps].read(), Sr2TiO4.ref, steps, Assert)


def check_Sr2TiO4_structure(actual, reference, steps, Assert):
    Assert.allclose(actual["lattice_vectors"], reference.lattice_vectors[steps])
    Assert.allclose(actual["positions"], reference.positions[steps])
    assert actual["elements"] == ["Sr", "Sr", "Ti", "O", "O", "O", "O"]
    assert actual["names"] == ["Sr_1", "Sr_2", "Ti_1", "O_1", "O_2", "O_3", "O_4"]
    assert actual["moments"] is None


def test_read_collinear(Fe3O4_collinear, Assert):
    for steps in (slice(None), slice(1, 3), 0):
        moments = Fe3O4_collinear[steps].read()["moments"]
        Assert.allclose(moments, Fe3O4_collinear.ref.moments[steps])
    Assert.allclose(Fe3O4_collinear.read()["moments"], Fe3O4_collinear.ref.moments[-1])


def test_read_charge_only(Fe3O4_charge_only):
    for steps in (slice(None), slice(1, 3), 0):
        assert Fe3O4_charge_only[steps].read()["moments"] is None
    assert Fe3O4_charge_only.read()["moments"] is None


def test_to_poscar(Sr2TiO4, Assert):
    assert Sr2TiO4.to_POSCAR() == REF_POSCAR
    assert Sr2TiO4[0].to_POSCAR() == REF_POSCAR.replace("Sr2TiO4", "Sr2TiO4 (step 1)")
    for steps in (slice(None), slice(1, 3)):
        with pytest.raises(exception.NotImplemented):
            Sr2TiO4[steps].to_POSCAR()


def test_from_poscar(Sr2TiO4, Assert):
    structure = Structure.from_POSCAR(REF_POSCAR)
    check_Sr2TiO4_structure(structure.read(), Sr2TiO4.ref, -1, Assert)


def test_to_ase_Sr2TiO4(Sr2TiO4, Assert):
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


def test_to_ase_collinear(Fe3O4_collinear, Assert):
    check_Fe3O4_ase(Fe3O4_collinear.to_ase(), Fe3O4_collinear.ref, -1, Assert)
    check_Fe3O4_ase(Fe3O4_collinear[0].to_ase(), Fe3O4_collinear.ref, 0, Assert)


def check_Fe3O4_ase(structure, reference, steps, Assert):
    Assert.allclose(structure.cell.array, reference.lattice_vectors[steps])
    Assert.allclose(structure.get_scaled_positions(), reference.positions[steps])
    moments = structure.get_initial_magnetic_moments()
    Assert.allclose(moments, reference.moments[steps])
    assert all(structure.symbols == "Fe3O4")
    assert all(structure.pbc)


def test_from_ase(Sr2TiO4, Assert):
    structure = Structure.from_ase(Sr2TiO4.to_ase())
    check_Sr2TiO4_structure(structure.read(), Sr2TiO4.ref, -1, Assert)


def test_to_mdtraj(Sr2TiO4, Assert):
    for steps in (slice(None), slice(1, 3)):
        trajectory = Sr2TiO4[steps].to_mdtraj()


def check_Sr2TiO4_mdtraj(trajectory, reference, steps, Assert):
    assert trajectory.n_frames == len(reference.positions[steps])
    assert trajectory.n_atoms == len(reference.elements[steps])
    unitcell_vectors = Trajectory.A_to_nm * reference.lattice_vectors[steps]
    cartesian_positions = [
        pos @ cell for pos, cell in zip(reference.positions[steps], unitcell_vectors)
    ]
    Assert.allclose(trajectory.xyz, np.array(cartesian_positions).astype(np.float32))
    Assert.allclose(trajectory.unitcell_vectors, unitcell_vectors)


def test_supercell_scale_all(Sr2TiO4, Assert):
    number_atoms = 7
    scale = 2
    supercell = Sr2TiO4.to_ase(supercell=scale)
    assert len(supercell) == number_atoms * scale ** 3
    Assert.allclose(supercell.cell.array, scale * Sr2TiO4.ref.lattice_vectors)


def test_supercell_scale_individual(Sr2TiO4, Assert):
    number_atoms = 7
    scale = (2, 1, 3)
    supercell = Sr2TiO4.to_ase(supercell=scale)
    assert len(supercell) == number_atoms * np.prod(scale)
    Assert.allclose(supercell.cell.array, np.diag(scale) @ Sr2TiO4.ref.lattice_vectors)


def test_supercell_wrong_size(Sr2TiO4):
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4.to_ase("foo")
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4.to_ase([1, 2])


def test_cartesian_positions(Sr2TiO4, Fe3O4_collinear, Assert):
    check_cartesian_positions(Sr2TiO4, Assert)
    check_cartesian_positions(Fe3O4_collinear, Assert)
    check_cartesian_positions(Fe3O4_collinear[0], Assert)


def check_cartesian_positions(structure, Assert):
    Assert.allclose(structure.cartesian_positions(), structure.to_ase().get_positions())


def test_volume_Fe3O4(Fe3O4_collinear, Assert):
    reference_volumes = determine_reference_volumes(Fe3O4_collinear.ref.lattice_vectors)
    Assert.allclose(Fe3O4_collinear.volume(), reference_volumes[-1])
    Assert.allclose(Fe3O4_collinear[0].volume(), reference_volumes[0])
    Assert.allclose(Fe3O4_collinear[:].volume(), reference_volumes)
    Assert.allclose(Fe3O4_collinear[1:3].volume(), reference_volumes[1:3])


def determine_reference_volumes(lattice_vectors):
    cross_product = np.cross(lattice_vectors[:, 0], lattice_vectors[:, 1])
    return np.abs(np.einsum("ij,ij -> i", cross_product, lattice_vectors[:, 2]))


def test_number_atoms(Sr2TiO4):
    assert Sr2TiO4.number_atoms() == 7


def test_number_steps(Sr2TiO4):
    assert Sr2TiO4.number_steps() == 1
    assert Sr2TiO4[0].number_steps() == 1
    assert Sr2TiO4[:].number_steps() == len(Sr2TiO4.ref.positions)
    assert Sr2TiO4[1:3].number_steps() == 2


def test_plot_Sr2TiO4(Sr2TiO4):
    check_plot_Sr2TiO4(Sr2TiO4)
    for steps in (slice(None), slice(1, 3), 0):
        check_plot_Sr2TiO4(Sr2TiO4[steps])


def check_plot_Sr2TiO4(structure):
    cm_init = patch.object(data.Viewer3d, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(data.Viewer3d, "show_cell")
    with cm_init as init, cm_cell as cell:
        structure.plot()
        init.assert_called_once()
        cell.assert_called_once()


def test_plot_collinear(Fe3O4_collinear, Assert):
    cm_init = patch.object(data.Viewer3d, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(data.Viewer3d, "show_cell")
    cm_arrows = patch.object(data.Viewer3d, "show_arrows_at_atoms")
    with cm_init, cm_cell, cm_arrows as arrows:
        Fe3O4_collinear.plot()
        arrows.assert_called_once()
        args, kwargs = arrows.call_args
    actual_moments = args[0]
    rescale_moments = Structure.length_moments / np.max(Fe3O4_collinear.ref.moments[-1])
    for actual, reference in zip(actual_moments, Fe3O4_collinear.ref.moments[-1]):
        Assert.allclose(actual, [0, 0, reference * rescale_moments])


def test_plot_noncollinear(Fe3O4_noncollinear, Assert):
    cm_init = patch.object(data.Viewer3d, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(data.Viewer3d, "show_cell")
    cm_arrows = patch.object(data.Viewer3d, "show_arrows_at_atoms")
    with cm_init, cm_cell, cm_arrows as arrows:
        Fe3O4_noncollinear.plot()
        arrows.assert_called_once()
        args, kwargs = arrows.call_args
    actual_moments = args[0]
    largest_moment = np.max(np.linalg.norm(Fe3O4_noncollinear.ref.moments[-1], axis=1))
    rescale_moments = Structure.length_moments / largest_moment
    for actual, reference in zip(actual_moments, Fe3O4_noncollinear.ref.moments[-1]):
        Assert.allclose(actual, reference * rescale_moments)


def test_plot_charge_only(Fe3O4_charge_only):
    cm_init = patch.object(data.Viewer3d, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(data.Viewer3d, "show_cell")
    cm_arrows = patch.object(data.Viewer3d, "show_arrows_at_atoms")
    with cm_init, cm_cell, cm_arrows as arrows:
        Fe3O4_charge_only.plot()
        arrows.assert_not_called()


def test_plot_zero_moments(Fe3O4_zero_moments):
    cm_init = patch.object(data.Viewer3d, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(data.Viewer3d, "show_cell")
    cm_arrows = patch.object(data.Viewer3d, "show_arrows_at_atoms")
    with cm_init, cm_cell, cm_arrows as arrows:
        Fe3O4_zero_moments.plot()
        arrows.assert_not_called()


def test_incorrect_step(Sr2TiO4):
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4[100].read()
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4[[0, 1]].read()


def test_print_final(Sr2TiO4, format_):
    actual, _ = format_(Sr2TiO4)
    assert actual == {"text/plain": REF_POSCAR, "text/html": REF_HTML}


def test_print_specific(Sr2TiO4, format_):
    actual, _ = format_(Sr2TiO4[0])
    ref_plain = REF_POSCAR.replace("Sr2TiO4", "Sr2TiO4 (step 1)")
    ref_html = REF_HTML.replace("Sr2TiO4", "Sr2TiO4 (step 1)")
    assert actual == {"text/plain": ref_plain, "text/html": ref_html}


def test_print_trajectory(Sr2TiO4, format_):
    actual, _ = format_(Sr2TiO4[1:3])
    ref_plain = REF_POSCAR.replace("Sr2TiO4", "Sr2TiO4 from step 2 to 4")
    ref_html = REF_HTML.replace("Sr2TiO4", "Sr2TiO4 from step 2 to 4")
    assert actual == {"text/plain": ref_plain, "text/html": ref_html}


def test_descriptor(Sr2TiO4, check_descriptors):
    descriptors = {
        "_to_dict": ["to_dict", "read"],
        "_to_viewer3d": ["to_viewer3d", "plot"],
        "_to_string": ["__str__"],
        "_to_ase": ["to_ase"],
        "_to_mdtraj": ["to_mdtraj"],
        "_to_poscar": ["to_POSCAR"],
        "_cartesian_positions": ["cartesian_positions"],
        "_number_atoms": ["number_atoms"],
        "_number_steps": ["number_steps"],
    }
    check_descriptors(Sr2TiO4, descriptors)


def test_from_file(raw_data, mock_file, check_read):
    raw_structure = raw_data.structure("Sr2TiO4")
    with mock_file("structure", raw_structure) as mocks:
        check_read(Structure, mocks, raw_structure)

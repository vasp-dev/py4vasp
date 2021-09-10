from unittest.mock import patch
from py4vasp.control import POSCAR
from py4vasp.data import Structure, Magnetism
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


@pytest.fixture
def Sr2TiO4(raw_data):
    raw_structure = raw_data.structure("Sr2TiO4")
    structure = Structure(raw_structure)
    structure.ref = types.SimpleNamespace()
    structure.ref.cell = raw_structure.cell.lattice_vectors * raw_structure.cell.scale
    structure.ref.positions = raw_structure.positions
    return structure


@pytest.fixture
def Fe3O4_collinear(raw_data):
    raw_structure = raw_data.structure("Fe3O4 collinear")
    structure = Structure(raw_structure)
    structure.ref = types.SimpleNamespace()
    structure.ref.moments = np.sum(raw_structure.magnetism.moments[-1, 1], axis=1)
    return structure


@pytest.fixture
def Fe3O4_noncollinear(raw_data):
    raw_structure = raw_data.structure("Fe3O4 noncollinear")
    structure = Structure(raw_structure)
    structure.ref = types.SimpleNamespace()
    structure.ref.moments = Magnetism(raw_structure.magnetism).total_moments(-1)
    return structure


@pytest.fixture
def Fe3O4_charge_only(raw_data):
    return Structure(raw_data.structure("Fe3O4 charge_only"))


@pytest.fixture
def Fe3O4_zero_moments(raw_data):
    return Structure(raw_data.structure("Fe3O4 zero_moments"))


def test_read(Sr2TiO4, Assert):
    check_Sr2TiO4_structure(Sr2TiO4, Sr2TiO4.ref, Assert)


def check_Sr2TiO4_structure(Sr2TiO4, reference, Assert):
    actual = Sr2TiO4.read()
    Assert.allclose(actual["lattice_vectors"], reference.cell)
    Assert.allclose(actual["positions"], reference.positions)
    assert actual["elements"] == ["Sr", "Sr", "Ti", "O", "O", "O", "O"]
    assert actual["moments"] is None


def test_to_poscar(Sr2TiO4, Assert):
    assert Sr2TiO4.to_POSCAR() == REF_POSCAR


def test_from_poscar(Sr2TiO4, Assert):
    structure = Structure.from_POSCAR(REF_POSCAR)
    check_Sr2TiO4_structure(structure, Sr2TiO4.ref, Assert)


def test_to_ase(Sr2TiO4, Assert):
    structure = Sr2TiO4.to_ase()
    Assert.allclose(structure.cell.array, Sr2TiO4.ref.cell)
    Assert.allclose(structure.get_scaled_positions(), Sr2TiO4.ref.positions)
    assert all(structure.symbols == "Sr2TiO4")
    assert all(structure.pbc)


def test_from_ase(Sr2TiO4, Assert):
    structure = Structure.from_ase(Sr2TiO4.to_ase())
    check_Sr2TiO4_structure(structure, Sr2TiO4.ref, Assert)


def test_supercell_scale_all(Sr2TiO4, Assert):
    number_atoms = 7
    scale = 2
    supercell = Sr2TiO4.to_ase(supercell=scale)
    assert len(supercell) == number_atoms * scale ** 3
    Assert.allclose(supercell.cell.array, scale * Sr2TiO4.ref.cell)


def test_supercell_scale_individual(Sr2TiO4, Assert):
    number_atoms = 7
    scale = (2, 1, 3)
    supercell = Sr2TiO4.to_ase(supercell=scale)
    assert len(supercell) == number_atoms * np.prod(scale)
    Assert.allclose(supercell.cell.array, np.diag(scale) @ Sr2TiO4.ref.cell)


def test_supercell_wrong_size(Sr2TiO4):
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4.to_ase("foo")
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4.to_ase([1, 2])


def test_cartesian_positions(Sr2TiO4, Assert):
    Assert.allclose(Sr2TiO4.cartesian_positions(), Sr2TiO4.to_ase().get_positions())


def test_length(Sr2TiO4):
    assert len(Sr2TiO4) == 7


def test_plot(Sr2TiO4):
    cm_init = patch.object(data.Viewer3d, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(data.Viewer3d, "show_cell")
    with cm_init as init, cm_cell as cell:
        Sr2TiO4.plot()
        init.assert_called_once()
        cell.assert_called_once()


def test_collinear_read(Fe3O4_collinear, Assert):
    moments = Fe3O4_collinear.read()["moments"]
    Assert.allclose(moments, Fe3O4_collinear.ref.moments)


def test_collinear_to_ase(Fe3O4_collinear, Assert):
    moments = Fe3O4_collinear.to_ase().get_initial_magnetic_moments()
    Assert.allclose(moments, Fe3O4_collinear.ref.moments)


def test_collinear_plot(Fe3O4_collinear, Assert):
    cm_init = patch.object(data.Viewer3d, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(data.Viewer3d, "show_cell")
    cm_arrows = patch.object(data.Viewer3d, "show_arrows_at_atoms")
    with cm_init, cm_cell, cm_arrows as arrows:
        Fe3O4_collinear.plot()
        arrows.assert_called_once()
        args, kwargs = arrows.call_args
    actual_moments = args[0]
    rescale_moments = Structure.length_moments / np.max(Fe3O4_collinear.ref.moments)
    for actual, reference in zip(actual_moments, Fe3O4_collinear.ref.moments):
        Assert.allclose(actual, [0, 0, reference * rescale_moments])


def test_noncollinear_plot(Fe3O4_noncollinear, Assert):
    cm_init = patch.object(data.Viewer3d, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(data.Viewer3d, "show_cell")
    cm_arrows = patch.object(data.Viewer3d, "show_arrows_at_atoms")
    with cm_init, cm_cell, cm_arrows as arrows:
        Fe3O4_noncollinear.plot()
        arrows.assert_called_once()
        args, kwargs = arrows.call_args
    actual_moments = args[0]
    largest_moment = np.max(np.linalg.norm(Fe3O4_noncollinear.ref.moments, axis=1))
    rescale_moments = Structure.length_moments / largest_moment
    for actual, reference in zip(actual_moments, Fe3O4_noncollinear.ref.moments):
        Assert.allclose(actual, reference * rescale_moments)


def test_charge_only_read(Fe3O4_charge_only):
    assert Fe3O4_charge_only.read()["moments"] is None


def test_charge_only_plot(Fe3O4_charge_only):
    cm_init = patch.object(data.Viewer3d, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(data.Viewer3d, "show_cell")
    cm_arrows = patch.object(data.Viewer3d, "show_arrows_at_atoms")
    with cm_init, cm_cell, cm_arrows as arrows:
        Fe3O4_charge_only.plot()
        arrows.assert_not_called()


def test_zero_moments_plot(Fe3O4_zero_moments):
    cm_init = patch.object(data.Viewer3d, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(data.Viewer3d, "show_cell")
    cm_arrows = patch.object(data.Viewer3d, "show_arrows_at_atoms")
    with cm_init, cm_cell, cm_arrows as arrows:
        Fe3O4_zero_moments.plot()
        arrows.assert_not_called()


def test_print(Sr2TiO4, format_):
    actual, _ = format_(Sr2TiO4)
    ref_html = """
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
    assert actual == {"text/plain": REF_POSCAR, "text/html": ref_html}


def test_descriptor(Sr2TiO4, check_descriptors):
    descriptors = {
        "_to_dict": ["to_dict", "read"],
        "_to_viewer3d": ["to_viewer3d", "plot"],
        "_to_string": ["to_POSCAR"],
        "_to_ase": ["to_ase"],
        "_cartesian_positions": ["cartesian_positions"],
    }
    check_descriptors(Sr2TiO4, descriptors)


# def test_from_file(raw_structure, mock_file, check_read):
#     with mock_file("structure", raw_structure) as mocks:
#         check_read(Structure, mocks, raw_structure)

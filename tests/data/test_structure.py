from py4vasp.data import Structure
from py4vasp.exceptions import RefinementException
import py4vasp.raw as raw
import pytest
import numpy as np


@pytest.fixture
def raw_structure():
    number_atoms = 20
    shape = (number_atoms, 3)
    return raw.Structure(
        cell=raw.Cell(scale=1.0, lattice_vectors=np.eye(3)),
        cartesian_positions=np.arange(np.prod(shape)).reshape(shape),
        species=np.array(["C"]*number_atoms, dtype="S2"),
    )


def test_read(raw_structure, Assert):
    actual = Structure(raw_structure).read()
    assert (actual["cell"] == raw_structure.cell.lattice_vectors).all()
    Assert.allclose(actual["cartesian_positions"],
                    raw_structure.cartesian_positions)
    assert (actual["species"] == raw_structure.species).all()


def test_to_pymatgen(raw_structure, Assert):
    structure = Structure(raw_structure)
    mg_structure = structure.to_pymatgen()
    a, b, c = mg_structure.lattice.as_dict()["matrix"]
    assert a == [1, 0, 0]
    assert b == [0, 1, 0]
    assert c == [0, 0, 1]


def test_plot(raw_structure, Assert):
    structure = Structure(raw_structure)
    assert structure.structure_viewer is None
    view = structure.plot()
    assert structure.structure_viewer is not None

    view = structure.plot(show_cell=False)
    assert [(msg["methodName"], msg["args"])
            for msg in view.get_state()["_ngl_msg_archive"]][1:] == []

    view = structure.plot(show_cell=True)
    assert [(msg["methodName"], msg["args"]) for msg in view.get_state()[
        "_ngl_msg_archive"]][1:] == [('addRepresentation', ['unitcell'])]

    view = structure.plot(show_cell=False, show_axes=False)
    assert [(msg["methodName"], msg["args"])
            for msg in view.get_state()["_ngl_msg_archive"]][1:] == []

    view = structure.plot(show_cell=False, show_axes=True, axes_length=5)
    assert [(msg["methodName"], msg["args"]) for msg in view.get_state()["_ngl_msg_archive"]][1:] == [
        ('addShape', [
         'shape', [('arrow', [0, 0, 0], [5, 0, 0], [1, 0, 0], 0.2)]]),
        ('addShape', [
         'shape', [('arrow', [0, 0, 0], [0, 5, 0], [0, 1, 0], 0.2)]]),
        ('addShape', [
         'shape', [('arrow', [0, 0, 0], [0, 0, 5], [0, 0, 1], 0.2)]]),
    ]


def test_plot_arrows(raw_structure, Assert):
    structure = Structure(raw_structure)
    assert structure.structure_viewer is None
    view = structure.plot_arrows([(0, 0, 1) for i in range(len(structure))])
    assert structure.structure_viewer is not None
    assert [(msg["methodName"], msg["args"]) for msg in view.get_state()[
        "_ngl_msg_archive"]][1] == ('addRepresentation', ['unitcell'])
    assert [(msg["methodName"], msg["args"]) for msg in view.get_state()[
        "_ngl_msg_archive"]][2] == ('addShape', ['shape', [('arrow', [0.0, 1.0, 2.0], [0.0, 1.0, 3.0], [0.1, 0.1, 0.8], 0.2)]])
    assert sum(msg["methodName"] == 'addShape' for msg in view.get_state()[
        "_ngl_msg_archive"]) == len(structure)

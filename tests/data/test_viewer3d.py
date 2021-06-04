from unittest.mock import patch
from py4vasp.data import Structure, Trajectory, Viewer3d
from py4vasp.data.viewer3d import _Arrow3d, _x_axis, _y_axis, _z_axis
from .test_structure import raw_structure, raw_topology
from .test_density import raw_density
from .test_trajectory import raw_trajectory
import py4vasp.exceptions as exception
import ipykernel.jsonutil as json
import numpy as np
import pytest
import nglview


@pytest.fixture
def viewer3d(raw_structure):
    return make_viewer(raw_structure)


def make_viewer(raw_structure, supercell=None):
    structure = Structure(raw_structure)
    viewer = structure.plot(supercell)
    viewer.raw_structure = raw_structure
    viewer.default_messages = count_messages(viewer, setup=True)
    return viewer


def count_messages(viewer, setup=False):
    n = viewer.default_messages if not setup else 0
    return len(viewer._ngl.get_state()["_ngl_msg_archive"]) - n


def last_messages(viewer, n=1, get_msg_kwargs=False):
    num_messages = count_messages(viewer)
    if num_messages >= n:
        index = viewer.default_messages + num_messages - n
        messages = viewer._ngl.get_state()["_ngl_msg_archive"][index:]
        if get_msg_kwargs:
            get_msg_tuple = lambda msg: (msg["methodName"], msg["args"], msg["kwargs"])
        else:
            get_msg_tuple = lambda msg: (msg["methodName"], msg["args"])
        return [get_msg_tuple(msg) for msg in messages]
    else:
        return []


def test_ipython(viewer3d):
    cm_display = patch.object(nglview.NGLWidget, "_ipython_display_", autospec=True)
    with cm_display as display:
        assert viewer3d._ngl is not None
        viewer3d._ipython_display_()
        display.assert_called_once()


def test_cell(viewer3d):
    viewer3d.show_cell()
    assert_add_unitcell(*last_messages(viewer3d))
    viewer3d.hide_cell()
    assert last_messages(viewer3d) == [("removeRepresentationsByName", ["unitcell", 0])]


def test_axes(viewer3d, assert_arrow_message):
    viewer3d.show_axes()
    messages = last_messages(viewer3d, n=3)
    assert_arrow_message(messages[0], _x_axis)
    assert_arrow_message(messages[1], _y_axis)
    assert_arrow_message(messages[2], _z_axis)
    viewer3d.show_axes()
    viewer3d.hide_axes()
    # ngl deletes the sent messages to indicate removal of the shapes
    assert count_messages(viewer3d) == 0
    viewer3d.hide_axes()


@pytest.fixture
def assert_arrow_message(Assert):
    def _assert_arrow_message(message, arrow):
        assert message[0] == "addShape"
        args = message[1]
        assert args[0] == "shape"
        params = args[1][0]
        assert params[0] == "arrow"
        Assert.allclose(params[1], arrow.tail)
        Assert.allclose(params[2], arrow.tip)
        assert params[3] == arrow.color
        assert params[4] == arrow.radius

    return _assert_arrow_message


def test_arrows(viewer3d, assert_arrow_message):
    positions = viewer3d.raw_structure.positions @ viewer3d.raw_structure.actual_cell
    number_atoms = len(positions)
    color = [0.1, 0.1, 0.8]
    arrows = create_arrows(viewer3d, number_atoms)
    messages = last_messages(viewer3d, number_atoms)
    assert len(messages) == number_atoms
    for message, tail, arrow in zip(messages, positions, arrows):
        tip = tail + arrow
        assert_arrow_message(message, _Arrow3d(tail, tip, color))
    viewer3d.show_arrows_at_atoms(arrows)
    assert count_messages(viewer3d) == 2 * number_atoms
    viewer3d.hide_arrows_at_atoms()
    # ngl deletes the sent messages to indicate removal of the shapes
    assert count_messages(viewer3d) == 0
    viewer3d.hide_arrows_at_atoms()


def test_supercell(raw_structure, assert_arrow_message):
    number_atoms = len(raw_structure.positions)
    supercell = (1, 2, 3)
    viewer = make_viewer(raw_structure, supercell)
    create_arrows(viewer, number_atoms)
    assert count_messages(viewer) == np.prod(supercell) * number_atoms
    supercell = 2  # meaning 2 along every direction
    viewer = make_viewer(raw_structure, supercell)
    create_arrows(viewer, number_atoms)
    assert count_messages(viewer) == supercell ** 3 * number_atoms


def test_bare_ngl_cannot_add_arrows_at_atoms(viewer3d):
    viewer = Viewer3d(viewer3d._ngl)
    with pytest.raises(exception.RefinementError):
        create_arrows(viewer, 1)


def create_arrows(viewer, number_atoms):
    arrows = np.repeat([(0, 0, 1)], number_atoms, axis=0)
    viewer.show_arrows_at_atoms(arrows)
    return arrows


def test_serializable():
    arrow = _Arrow3d(np.zeros(3), np.ones(3), np.ones(1))
    for element in arrow.to_serializable():
        json.json_clean(element)


def test_standard_form(raw_structure, Assert):
    x = np.sqrt(0.5)
    raw_structure.cell.lattice_vectors = np.array([[x, x, 0], [-x, x, 0], [0, 0, 1]])
    viewer = make_viewer(raw_structure)
    expected_positions = raw_structure.cell.scale * raw_structure.positions
    Assert.allclose(viewer._positions, expected_positions)


def test_isosurface(raw_density):
    viewer = make_viewer(raw_density.structure)
    viewer.show_isosurface(raw_density.charge)
    messages = last_messages(viewer, n=1, get_msg_kwargs=True)
    assert_load_file(messages[0], binary=True, default=True)
    #
    kwargs = {"isolevel": 0.1, "color": "red"}
    viewer.show_isosurface(raw_density.charge, **kwargs)
    messages = last_messages(viewer, n=2, get_msg_kwargs=True)
    assert_load_file(messages[0], binary=True, default=False)
    assert_add_surface(messages[1], kwargs)


def test_trajectory(raw_trajectory):
    trajectory = Trajectory(raw_trajectory)
    viewer = trajectory.plot()
    viewer.default_messages = 0
    n = count_messages(viewer)
    assert n == 2
    messages = last_messages(viewer, n, get_msg_kwargs=True)
    assert_load_file(messages[0], binary=False, default=True)
    assert_add_unitcell(messages[1])


def assert_load_file(message, binary, default):
    assert message[0] == "loadFile"
    assert message[1][0]["binary"] == binary
    assert message[2]["defaultRepresentation"] == default


def assert_add_unitcell(message):
    assert message[0] == "addRepresentation"
    assert message[1][0] == "unitcell"


def assert_add_surface(message, kwargs):
    assert message[0] == "addRepresentation"
    assert message[1][0] == "surface"
    for key, val in kwargs.items():
        assert message[2][key] == val

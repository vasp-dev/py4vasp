from unittest.mock import patch
from py4vasp.data import Structure, Viewer3d
from py4vasp.data.viewer3d import _Arrow3d, _x_axis, _y_axis, _z_axis
from .test_structure import raw_structure
import numpy as np
import pytest
import nglview


@pytest.fixture
def viewer3d(raw_structure):
    structure = Structure(raw_structure)
    viewer = structure.plot()
    viewer.default_messages = count_messages(viewer, setup=True)
    viewer.positions = raw_structure.cartesian_positions
    return viewer


def count_messages(viewer, setup=False):
    n = viewer.default_messages if not setup else 0
    return len(viewer._ngl.get_state()["_ngl_msg_archive"]) - n


def last_messages(viewer, n=1):
    num_messages = count_messages(viewer)
    if num_messages >= n:
        index = viewer.default_messages + num_messages - n
        messages = viewer._ngl.get_state()["_ngl_msg_archive"][index:]
        return [(msg["methodName"], msg["args"]) for msg in messages]
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
    assert last_messages(viewer3d) == [("addRepresentation", ["unitcell"])]
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
    direction = np.array((0, 0, 1))
    number_atoms = len(viewer3d._structure)
    arrows = np.repeat([direction], number_atoms, axis=0)
    viewer3d.show_arrows_at_atoms(arrows)
    messages = last_messages(viewer3d, number_atoms)
    color = [0.1, 0.1, 0.8]
    assert len(messages) == number_atoms
    for message, tail, arrow in zip(messages, viewer3d.positions, arrows):
        tip = tail + arrow
        assert_arrow_message(message, _Arrow3d(tail, tip, color))
    viewer3d.show_arrows_at_atoms(arrows)
    assert count_messages(viewer3d) == 2 * number_atoms
    viewer3d.hide_arrows_at_atoms()
    # ngl deletes the sent messages to indicate removal of the shapes
    assert count_messages(viewer3d) == 0
    viewer3d.hide_arrows_at_atoms()

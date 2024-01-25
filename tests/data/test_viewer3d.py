# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._third_party.viewer.viewer3d import Viewer3d, _Arrow3d, _x_axis, _y_axis, _z_axis
from py4vasp._util import import_
from py4vasp.data import Structure

json = import_.optional("ipykernel.jsonutil")
nglview = import_.optional("nglview")


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


@pytest.fixture
def viewer3d(raw_data, not_core):
    structure = Structure.from_data(raw_data.structure("Sr2TiO4"))
    return make_viewer(structure)


@pytest.fixture
def nonstandard_form(raw_data, not_core):
    raw_structure = raw_data.structure("Sr2TiO4")
    x = np.sqrt(0.5)
    raw_structure.cell.lattice_vectors = np.array([[[x, x, 0], [-x, x, 0], [0, 0, 1]]])
    raw_structure.positions += 0.1  # shift to avoid small comparisons
    viewer = make_viewer(Structure.from_data(raw_structure))
    viewer.ref.transformation = np.array([[x, x, 0], [-x, x, 0], [0, 0, 1]])
    return viewer


def make_viewer(structure, supercell=None):
    viewer = structure.plot(supercell)
    viewer.ref = types.SimpleNamespace()
    viewer.ref.positions = structure.cartesian_positions()
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


def test_arrows(viewer3d, assert_arrow_message):
    number_atoms = len(viewer3d.ref.positions)
    color = [0.1, 0.1, 0.8]
    arrows = create_arrows(viewer3d, number_atoms)
    messages = last_messages(viewer3d, number_atoms)
    assert len(messages) == number_atoms
    for message, tail, arrow in zip(messages, viewer3d.ref.positions, arrows):
        tip = tail + arrow
        assert_arrow_message(message, _Arrow3d(tail, tip, color))
    viewer3d.show_arrows_at_atoms(arrows)
    assert count_messages(viewer3d) == 2 * number_atoms
    viewer3d.hide_arrows_at_atoms()
    # ngl deletes the sent messages to indicate removal of the shapes
    assert count_messages(viewer3d) == 0
    viewer3d.hide_arrows_at_atoms()


def test_supercell(raw_data, not_core):
    structure = Structure.from_data(raw_data.structure("Sr2TiO4"))
    number_atoms = structure.number_atoms()
    supercell = (1, 2, 3)
    viewer = make_viewer(structure, supercell)
    create_arrows(viewer, number_atoms)
    assert count_messages(viewer) == np.prod(supercell) * number_atoms
    supercell = 2  # meaning 2 along every direction
    viewer = make_viewer(structure, supercell)
    create_arrows(viewer, number_atoms)
    assert count_messages(viewer) == supercell**3 * number_atoms


def test_bare_ngl_cannot_add_arrows_at_atoms(viewer3d):
    viewer = Viewer3d(viewer3d._ngl)
    with pytest.raises(exception.RefinementError):
        create_arrows(viewer, 1)


def create_arrows(viewer, number_atoms):
    arrows = np.repeat([(0.2, 0.4, 0.6)], number_atoms, axis=0)
    viewer.show_arrows_at_atoms(arrows)
    return arrows


def test_serializable(not_core):
    arrow = _Arrow3d(np.zeros(3), np.ones(3), np.ones(1))
    for element in arrow.to_serializable():
        json.json_clean(element)


def test_nonstandard_form(nonstandard_form, Assert, assert_arrow_message):
    viewer = nonstandard_form
    Assert.allclose(viewer._positions, viewer.ref.positions)
    viewer.show_axes()
    messages = last_messages(viewer, n=3)
    assert_arrow_message(messages[0], rotate(_x_axis, viewer.ref.transformation))
    assert_arrow_message(messages[1], rotate(_y_axis, viewer.ref.transformation))
    assert_arrow_message(messages[2], rotate(_z_axis, viewer.ref.transformation))
    #
    color = [0.1, 0.1, 0.8]
    number_atoms = len(viewer.ref.positions)
    arrows = create_arrows(viewer, number_atoms)
    messages = last_messages(viewer, number_atoms)
    for message, tail, arrow in zip(messages, viewer.ref.positions, arrows):
        tip = tail + viewer.ref.transformation @ arrow
        assert_arrow_message(message, _Arrow3d(tail, tip, color))


def rotate(arrow, transformation):
    return _Arrow3d(
        transformation @ arrow.tail, transformation @ arrow.tip, arrow.color
    )


def test_isosurface(raw_data, not_core):
    raw_density = raw_data.density("Fe3O4 collinear")
    viewer = make_viewer(Structure.from_data(raw_density.structure))
    viewer.show_isosurface(raw_density.charge)
    messages = last_messages(viewer, n=1, get_msg_kwargs=True)
    assert_load_file(messages[0], binary=True, default=True)
    #
    kwargs = {"isolevel": 0.1, "color": "red"}
    viewer.show_isosurface(raw_density.charge, **kwargs)
    messages = last_messages(viewer, n=2, get_msg_kwargs=True)
    assert_load_file(messages[0], binary=True, default=False)
    assert_add_surface(messages[1], kwargs)


def test_trajectory(raw_data, not_core):
    structure = Structure.from_data(raw_data.structure("Sr2TiO4"))
    viewer = structure[:].plot()
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

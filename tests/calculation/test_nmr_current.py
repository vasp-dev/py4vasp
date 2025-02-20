# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import numpy as np
import pytest

from py4vasp._calculation.nmr_current import NmrCurrent
from py4vasp._calculation.structure import Structure


@pytest.fixture(params=("all", "x", "y", "z"))
def nmr_current(request, raw_data):
    return make_reference_current(request.param, raw_data)


@pytest.fixture
def all_nmr_current(raw_data):
    return make_reference_current("all", raw_data)


def make_reference_current(selection, raw_data):
    raw_current = raw_data.nmr_current(selection)
    current = NmrCurrent.from_data(raw_current)
    current.ref = types.SimpleNamespace()
    current.ref.structure = Structure.from_data(raw_current.structure)
    # the effect is that all is equivalent to z for plotting
    if selection in ("x", "all"):
        current.ref.current_Bx = np.transpose(raw_current.nmr_current[0])
        current.ref.default_current = current.ref.current_Bx
        current.ref.default_direction = "x"
    if selection in ("y", "all"):
        index_y = raw_current.valid_indices.index("y")
        current.ref.current_By = np.transpose(raw_current.nmr_current[index_y])
        current.ref.default_current = current.ref.current_By
        current.ref.default_direction = "y"
    if selection in ("z", "all"):
        current.ref.current_Bz = np.transpose(raw_current.nmr_current[-1])
        current.ref.default_current = current.ref.current_Bz
        current.ref.default_direction = "z"
    return current


def test_read(nmr_current, Assert):
    actual = nmr_current.read()
    Assert.same_structure(actual["structure"], nmr_current.ref.structure.read())
    for axis in "xyz":
        label = f"nmr_current_B{axis}"
        reference_current = getattr(nmr_current.ref, f"current_B{axis}", None)
        if reference_current is not None:
            Assert.allclose(actual[label], reference_current)
        else:
            assert label not in actual


def test_to_quiver(nmr_current, Assert):
    expected_data = nmr_current.ref.default_current[:, :, 10, :2]
    reference_structure = nmr_current.ref.structure
    expected_lattice_vectors = reference_structure.lattice_vectors()[:2, :2]
    graph = nmr_current.to_quiver(c=0.7)
    assert len(graph) == 1
    series = graph.series[0]
    Assert.allclose(series.data, 0.003 * np.moveaxis(expected_data, -1, 0))
    Assert.allclose(series.lattice.vectors, expected_lattice_vectors)
    assert series.label == f"nmr_current_B{nmr_current.ref.default_direction}"


def test_to_quiver_supercell(nmr_current, Assert):
    graph = nmr_current.to_quiver(a=0, supercell=2)
    Assert.allclose(graph.series[0].supercell, (2, 2))
    graph = nmr_current.to_quiver(a=0, supercell=(2, 1))
    Assert.allclose(graph.series[0].supercell, (2, 1))


@pytest.mark.parametrize(
    "normal, rotation",
    [
        ("auto", np.eye(2)),
        ("x", np.array([[0, -1], [1, 0]])),
        ("y", np.diag((1, -1))),
        ("z", np.eye(2)),
    ],
)
def test_to_quiver_normal(nmr_current, normal, rotation, Assert):
    unrotated_graph = nmr_current.to_quiver(c=0.5)
    rotated_graph = nmr_current.to_quiver(c=0.5, normal=normal)
    expected_lattice = unrotated_graph.series[0].lattice.vectors @ rotation
    Assert.allclose(rotated_graph.series[0].lattice.vectors, expected_lattice)
    expected_data = (unrotated_graph.series[0].data.T @ rotation).T
    Assert.allclose(rotated_graph.series[0].data, expected_data)


@pytest.mark.parametrize("selection", ("x", "y", "z"))
def test_to_quiver_selection(all_nmr_current, selection, Assert):
    expected_data = getattr(all_nmr_current.ref, f"current_B{selection}")[:, :, 4, :2]
    reference_structure = all_nmr_current.ref.structure
    expected_lattice_vectors = reference_structure.lattice_vectors()[:2, :2]
    graph = all_nmr_current.to_quiver(selection, c=0.3)
    assert len(graph) == 1
    series = graph.series[0]
    Assert.allclose(series.data, 0.003 * np.moveaxis(expected_data, -1, 0))
    Assert.allclose(series.lattice.vectors, expected_lattice_vectors)
    assert series.label == f"nmr_current_B{selection}"


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.nmr_current("x")
    parameters = {"to_quiver": {"a": 0.3}}
    check_factory_methods(NmrCurrent, data, parameters)

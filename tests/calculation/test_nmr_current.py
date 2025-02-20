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


@pytest.fixture(params=("x", "y", "z"))
def single_nmr_current(request, raw_data):
    return make_reference_current(request.param, raw_data)


def make_reference_current(selection, raw_data):
    raw_current = raw_data.nmr_current(selection)
    current = NmrCurrent.from_data(raw_current)
    current.ref = types.SimpleNamespace()
    current.ref.structure = Structure.from_data(raw_current.structure)
    if selection in ("x", "all"):
        current.ref.current_Bx = np.transpose(raw_current.nmr_current[0])
    if selection in ("y", "all"):
        index_y = raw_current.valid_indices.index("y")
        current.ref.current_By = np.transpose(raw_current.nmr_current[index_y])
    if selection in ("z", "all"):
        current.ref.current_Bz = np.transpose(raw_current.nmr_current[-1])
    if selection != "all":
        current.ref.single_current = getattr(current.ref, f"current_B{selection}")
        current.ref.direction = selection
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


def test_to_quiver(single_nmr_current, Assert):
    expected_data = single_nmr_current.ref.single_current[:, :, 10, :2]
    reference_structure = single_nmr_current.ref.structure
    expected_lattice_vectors = reference_structure.lattice_vectors()[:2, :2]
    graph = single_nmr_current.to_quiver(c=0.7)
    assert len(graph) == 1
    series = graph.series[0]
    Assert.allclose(series.data, 0.003 * np.moveaxis(expected_data, -1, 0))
    Assert.allclose(series.lattice.vectors, expected_lattice_vectors)
    assert series.label == f"nmr_current_B{single_nmr_current.ref.direction}"


def test_to_quiver_supercell(single_nmr_current, Assert):
    graph = single_nmr_current.to_quiver(a=0, supercell=2)
    Assert.allclose(graph.series[0].supercell, (2, 2))
    graph = single_nmr_current.to_quiver(a=0, supercell=(2, 1))
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
def test_to_quiver_normal(single_nmr_current, normal, rotation, Assert):
    unrotated_graph = single_nmr_current.to_quiver(c=0.5)
    rotated_graph = single_nmr_current.to_quiver(c=0.5, normal=normal)
    expected_lattice = unrotated_graph.series[0].lattice.vectors @ rotation
    Assert.allclose(rotated_graph.series[0].lattice.vectors, expected_lattice)
    expected_data = (unrotated_graph.series[0].data.T @ rotation).T
    Assert.allclose(rotated_graph.series[0].data, expected_data)


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.nmr_current("x")
    parameters = {"to_quiver": {"a": 0.3}}
    check_factory_methods(NmrCurrent, data, parameters)

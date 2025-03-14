# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
import types

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._calculation.current_density import CurrentDensity
from py4vasp._calculation.structure import Structure


@pytest.fixture(params=("all", "x", "y", "z"))
def current_density(request, raw_data):
    return make_reference_current_density(request.param, raw_data)


@pytest.fixture
def multiple_current_densities(raw_data):
    return make_reference_current_density("all", raw_data)


@dataclasses.dataclass
class Normal:
    normal: str
    expected_rotation: np.ndarray


@pytest.fixture(
    params=[
        Normal(normal="auto", expected_rotation=np.eye(2)),
        Normal(normal="x", expected_rotation=np.array([[0, -1], [1, 0]])),
        Normal(normal="y", expected_rotation=np.diag((1, -1))),
        Normal(normal="z", expected_rotation=np.eye(2)),
    ]
)
def normal_vector(request):
    return request.param


def make_reference_current_density(selection, raw_data):
    raw_current = raw_data.current_density(selection)
    current = CurrentDensity.from_data(raw_current)
    current.ref = types.SimpleNamespace()
    current.ref.structure = Structure.from_data(raw_current.structure)
    # the effect is that all is equivalent to z for plotting
    if selection in ("x", "all"):
        current.ref.current_x = np.transpose(raw_current.current_density[0])
        current.ref.default_current = current.ref.current_x
        current.ref.default_direction = "x"
    if selection in ("y", "all"):
        index_y = raw_current.valid_indices.index("y")
        current.ref.current_y = np.transpose(raw_current.current_density[index_y])
        current.ref.default_current = current.ref.current_y
        current.ref.default_direction = "y"
    if selection in ("z", "all"):
        current.ref.current_z = np.transpose(raw_current.current_density[-1])
        current.ref.default_current = current.ref.current_z
        current.ref.default_direction = "z"
    selections = "x, y, z" if selection == "all" else selection
    current.ref.string = f"""\
current density:
    structure: Fe3O4
    grid: 10, 12, 14
    selections: {selections}"""
    return current


def test_read(current_density, Assert):
    actual = current_density.read()
    Assert.same_structure(actual["structure"], current_density.ref.structure.read())
    for axis in "xyz":
        label = f"current_{axis}"
        reference_current = getattr(current_density.ref, f"current_{axis}", None)
        if reference_current is not None:
            Assert.allclose(actual[label], reference_current)
        else:
            assert label not in actual


def test_to_quiver(current_density, Assert):
    expected_data = current_density.ref.default_current[:, :, 10, :2]
    reference_structure = current_density.ref.structure
    expected_lattice_vectors = reference_structure.lattice_vectors()[:2, :2]
    graph = current_density.to_quiver(c=0.7)
    assert len(graph) == 1
    series = graph.series[0]
    Assert.allclose(series.data, 0.003 * np.moveaxis(expected_data, -1, 0))
    Assert.allclose(series.lattice.vectors, expected_lattice_vectors)
    assert series.label == f"current_{current_density.ref.default_direction}"


def test_to_quiver_supercell(current_density, Assert):
    graph = current_density.to_quiver(a=0, supercell=2)
    Assert.allclose(graph.series[0].supercell, (2, 2))
    graph = current_density.to_quiver(a=0, supercell=(2, 1))
    Assert.allclose(graph.series[0].supercell, (2, 1))


def test_to_quiver_normal(current_density, normal_vector, Assert):
    unrotated_graph = current_density.to_quiver(c=0.5)
    rotated_graph = current_density.to_quiver(c=0.5, normal=normal_vector.normal)
    rotation = normal_vector.expected_rotation
    expected_lattice = unrotated_graph.series[0].lattice.vectors @ rotation
    Assert.allclose(rotated_graph.series[0].lattice.vectors, expected_lattice)
    expected_data = (unrotated_graph.series[0].data.T @ rotation).T
    Assert.allclose(rotated_graph.series[0].data, expected_data)


@pytest.mark.parametrize("selection", ("x", "y", "z"))
def test_to_quiver_selection(multiple_current_densities, selection, Assert):
    expected_data = getattr(multiple_current_densities.ref, f"current_{selection}")
    expected_data = expected_data[:, :, 4, :2]
    reference_structure = multiple_current_densities.ref.structure
    expected_lattice_vectors = reference_structure.lattice_vectors()[:2, :2]
    graph = multiple_current_densities.to_quiver(selection, c=0.3)
    assert len(graph) == 1
    series = graph.series[0]
    Assert.allclose(series.data, np.moveaxis(expected_data, -1, 0))
    Assert.allclose(series.lattice.vectors, expected_lattice_vectors)
    assert series.label == f"current_{selection}"


@pytest.mark.parametrize(
    "kwargs, index, position",
    (({"a": 0.1}, 0, 1), ({"b": 0.7}, 1, 8), ({"c": 1.3}, 2, 4)),
)
def test_to_contour(current_density, kwargs, index, position, Assert):
    graph = current_density.to_contour(**kwargs)
    slice_ = [slice(None), slice(None), slice(None)]
    slice_[index] = position
    vector_data = current_density.ref.default_current[tuple(slice_)]
    scalar_data = np.linalg.norm(vector_data, axis=-1)
    assert len(graph) == 1
    series = graph.series[0]
    Assert.allclose(series.data, scalar_data)


def test_to_contour_supercell(current_density, Assert):
    graph = current_density.to_contour(b=0, supercell=2)
    Assert.allclose(graph.series[0].supercell, (2, 2))
    graph = current_density.to_contour(b=0, supercell=(2, 1))
    Assert.allclose(graph.series[0].supercell, (2, 1))


def test_to_contour_normal(current_density, normal_vector, Assert):
    graph = current_density.to_contour(c=0.5, normal=normal_vector.normal)
    rotation = normal_vector.expected_rotation
    lattice_vectors = current_density.ref.structure.lattice_vectors()
    expected_lattice = lattice_vectors[:2, :2] @ rotation
    Assert.allclose(graph.series[0].lattice.vectors, expected_lattice)


@pytest.mark.parametrize("selection", ("x", "y", "z"))
def test_to_contour_selection(multiple_current_densities, selection, Assert):
    expected_data = getattr(multiple_current_densities.ref, f"current_{selection}")
    expected_data = np.linalg.norm(expected_data[:, :, -1], axis=-1)
    reference_structure = multiple_current_densities.ref.structure
    expected_lattice_vectors = reference_structure.lattice_vectors()[:2, :2]
    graph = multiple_current_densities.to_contour(selection, c=-0.1)
    assert len(graph) == 1
    series = graph.series[0]
    Assert.allclose(series.data, expected_data)
    Assert.allclose(series.lattice.vectors, expected_lattice_vectors)
    assert series.label == f"current_{selection}"


def test_print(current_density, format_):
    actual, _ = format_(current_density)
    assert actual == {"text/plain": current_density.ref.string}


@pytest.mark.parametrize("args, kwargs", ([(), {}], [(), {"a": 1, "b": 2}], [(3,), {}]))
def test_incorrect_slice_raises_error(current_density, args, kwargs):
    with pytest.raises(exception.IncorrectUsage):
        current_density.to_contour(*args, **kwargs)
    with pytest.raises(exception.IncorrectUsage):
        current_density.to_quiver(*args, **kwargs)


def test_incorrect_selection_raises_error(raw_data):
    current_density = make_reference_current_density("x", raw_data)
    with pytest.raises(exception.IncorrectUsage):
        current_density.to_contour("y", a=0)
    with pytest.raises(exception.IncorrectUsage):
        current_density.to_quiver("y", b=0)
    with pytest.raises(exception.IncorrectUsage):
        current_density.to_contour("foo", c=0)
    with pytest.raises(exception.IncorrectUsage):
        current_density.to_quiver("foo", a=0)


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.current_density("x")
    check_factory_methods(CurrentDensity, data)

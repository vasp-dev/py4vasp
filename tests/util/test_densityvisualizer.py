from types import SimpleNamespace

import numpy as np
import pytest

from py4vasp import raw
from py4vasp._calculation.density import Density
from py4vasp._calculation.structure import Structure
from py4vasp._util import density, index, slicing


def _make_visualizer(raw_data, request, data_ndim: int):
    structure = Structure.from_data(raw_data.structure("Sr2TiO4"))
    raw_density = raw_data.density("Sr2TiO4")
    if request.param == "simple":
        if data_ndim == 3:
            data3d = np.ones(shape=(3, 4, 5))
        elif data_ndim == 4:
            data3d = np.ones(shape=(3, 4, 5, 3))
        else:
            raise NotImplementedError(f"ndim {data_ndim} not implemented.")
        ref_data = [data3d.T]
        selector = index.Selector({}, data3d)
        selections = [()]
    elif request.param == "with_selections":
        if data_ndim == 3:
            data3d = np.ones(shape=(3, 4, 5, 10))
        elif data_ndim == 4:
            data3d = np.ones(shape=(3, 4, 5, 3, 10))
        else:
            raise NotImplementedError(f"ndim {data_ndim} not implemented.")
        ref_data = [data3d.T[0], data3d.T[1]]
        selector = index.Selector({-1: {"spin up": 0, "spin down": 1}}, data3d)
        selections = [("spin up",), ("spin down",)]
    else:
        raise NotImplementedError(f"Requested param {request.param} not implemented.")

    visualizer = density.Visualizer(structure, selector)
    visualizer.ref = SimpleNamespace()
    visualizer.ref.structure = structure
    visualizer.ref.data3d = ref_data
    visualizer.ref.selections = selections
    visualizer.ref.density = Density.from_data(raw_density)
    return visualizer


@pytest.fixture(params=["simple", "with_selections"])
def visualizer(raw_data, request):
    return _make_visualizer(raw_data, request, 3)


@pytest.fixture(params=["simple", "with_selections"])
def visualizer_quiver(raw_data, request):
    return _make_visualizer(raw_data, request, 4)


def test_view(visualizer, Assert):
    view = visualizer.to_view(visualizer.ref.selections)

    Assert.same_structure_view(visualizer.ref.structure.to_view(), view)
    assert len(view.grid_scalars) == len(visualizer.ref.selections)
    for sel, scalar, data in zip(
        visualizer.ref.selections, view.grid_scalars, visualizer.ref.data3d
    ):
        if len(sel) > 0:
            expected_label = sel[0]
        else:
            expected_label = ""
        assert scalar.label == expected_label
        Assert.allclose(scalar.quantity, data)


@pytest.mark.parametrize("supercell", [(2, 3, 2), 3, (2, 5, 1)])
def test_view_supercell(visualizer, supercell, Assert):
    view = visualizer.to_view(visualizer.ref.selections, supercell=supercell)

    Assert.same_structure_view(
        visualizer.ref.structure.to_view(supercell=supercell), view
    )
    assert len(view.grid_scalars) == len(visualizer.ref.selections)
    for sel, scalar, data in zip(
        visualizer.ref.selections, view.grid_scalars, visualizer.ref.data3d
    ):
        if len(sel) > 0:
            expected_label = sel[0]
        else:
            expected_label = ""
        assert scalar.label == expected_label
        Assert.allclose(scalar.quantity, data)


@pytest.mark.parametrize(
    "kwargs, index, position",
    (({"a": 0.2}, 0, 1), ({"b": 0.5}, 1, 2), ({"c": 1.3}, 2, 1)),
)
def test_contour(visualizer, kwargs, index, position, Assert):
    graph = visualizer.to_contour(visualizer.ref.selections, **kwargs)

    assert len(graph) == len(visualizer.ref.selections)
    for sel, series, data in zip(
        visualizer.ref.selections, graph.series, visualizer.ref.data3d
    ):
        slice_ = [slice(None), slice(None), slice(None)]
        slice_[index] = position
        expected_data = np.array(data)[tuple(slice_)]
        Assert.allclose(series.data, expected_data)

        lattice_vectors = visualizer.ref.structure.lattice_vectors()
        lattice_vectors = np.delete(lattice_vectors, index, axis=0)
        expected_products = lattice_vectors @ lattice_vectors.T
        actual_products = series.lattice.vectors @ series.lattice.vectors.T
        Assert.allclose(actual_products, expected_products)
        if len(sel) > 0:
            expected_label = sel[0]
        else:
            expected_label = ""
        assert series.label == expected_label


@pytest.mark.parametrize("supercell", [(2, 3), 3, (2, 5)])
def test_contour_supercell(visualizer, supercell, Assert):
    kwargs, index = ({"c": 1.3}, 2)
    graph = visualizer.to_contour(
        visualizer.ref.selections, supercell=supercell, **kwargs
    )

    assert len(graph) == len(visualizer.ref.selections)
    for series in graph.series:
        lattice_vectors = visualizer.ref.structure.lattice_vectors()
        lattice_vectors = np.delete(lattice_vectors, index, axis=0)
        expected_products = lattice_vectors @ lattice_vectors.T
        actual_products = series.lattice.vectors @ series.lattice.vectors.T
        Assert.allclose(actual_products, expected_products)
        expected_supercell = (
            supercell if isinstance(supercell, tuple) else (supercell, supercell)
        )
        Assert.allclose(series.supercell, expected_supercell)


@pytest.mark.parametrize("normal", [None, "auto", "x"])
def test_contour_normal(visualizer, normal, Assert):
    graph = visualizer.to_contour(visualizer.ref.selections, normal=normal, c=1.3)

    assert len(graph) == len(visualizer.ref.selections)
    for series in graph.series:
        expected_plane = slicing.plane(
            visualizer.ref.structure.lattice_vectors(), "c", normal
        )
        Assert.allclose(series.lattice.cell, expected_plane.cell)
        Assert.allclose(series.lattice.vectors, expected_plane.vectors)


def test_quiver(visualizer_quiver, Assert):
    kwargs, index, position = ({"c": 1.3}, 2, 1)
    graph = visualizer_quiver.to_quiver(visualizer_quiver.ref.selections, **kwargs)

    assert len(graph) == len(visualizer_quiver.ref.selections)
    for sel, series, data in zip(
        visualizer_quiver.ref.selections, graph.series, visualizer_quiver.ref.data3d
    ):
        slice_ = [slice(0, 2), slice(None), slice(None), slice(None)]
        slice_[index + 1] = position
        expected_data = np.array(data)[tuple(slice_)]
        Assert.allclose(series.data, expected_data)

        lattice_vectors = visualizer_quiver.ref.structure.lattice_vectors()
        lattice_vectors = np.delete(lattice_vectors, index, axis=0)
        expected_products = lattice_vectors @ lattice_vectors.T
        actual_products = series.lattice.vectors @ series.lattice.vectors.T
        Assert.allclose(actual_products, expected_products)
        if len(sel) > 0:
            expected_label = sel[0]
        else:
            expected_label = ""
        assert series.label == expected_label


@pytest.mark.parametrize("supercell", [(2, 3), 3, (2, 5)])
def test_quiver_supercell(visualizer_quiver, supercell, Assert):
    kwargs, index = ({"c": 1.3}, 2)
    graph = visualizer_quiver.to_quiver(
        visualizer_quiver.ref.selections, supercell=supercell, **kwargs
    )

    assert len(graph) == len(visualizer_quiver.ref.selections)
    for series in graph.series:
        lattice_vectors = visualizer_quiver.ref.structure.lattice_vectors()
        lattice_vectors = np.delete(lattice_vectors, index, axis=0)
        expected_products = lattice_vectors @ lattice_vectors.T
        actual_products = series.lattice.vectors @ series.lattice.vectors.T
        Assert.allclose(actual_products, expected_products)
        expected_supercell = (
            supercell if isinstance(supercell, tuple) else (supercell, supercell)
        )
        Assert.allclose(series.supercell, expected_supercell)


@pytest.mark.parametrize("normal", [None, "auto", "x"])
def test_quiver_normal(visualizer_quiver, normal, Assert):
    graph = visualizer_quiver.to_quiver(
        visualizer_quiver.ref.selections, normal=normal, c=1.3
    )

    assert len(graph) == len(visualizer_quiver.ref.selections)
    for series in graph.series:
        expected_plane = slicing.plane(
            visualizer_quiver.ref.structure.lattice_vectors(), "c", normal
        )
        Assert.allclose(series.lattice.cell, expected_plane.cell)
        Assert.allclose(series.lattice.vectors, expected_plane.vectors)


def test_quiver_collinear(visualizer, Assert):
    graph = visualizer.to_quiver(visualizer.ref.selections, c=1.3)

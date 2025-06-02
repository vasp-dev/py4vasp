from types import SimpleNamespace

import numpy as np
import pytest

from py4vasp import raw
from py4vasp._calculation.density import Density
from py4vasp._calculation.structure import Structure
from py4vasp._util import density, index, slicing
from py4vasp.exception import IncorrectUsage


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

    visualizer = density.Visualizer(structure)
    visualizer.ref = SimpleNamespace()
    visualizer.ref.structure = structure
    visualizer.ref.selector = selector
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
    dataDict = {
        (selection[0] if selection else ""): (visualizer.ref.selector[selection].T)[
            np.newaxis
        ]
        for selection in visualizer.ref.selections
    }
    view = visualizer.to_view(dataDict)

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
    dataDict = {
        (selection[0] if selection else ""): (visualizer.ref.selector[selection].T)[
            np.newaxis
        ]
        for selection in visualizer.ref.selections
    }
    view = visualizer.to_view(dataDict, supercell=supercell)

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
    "slice_args, index, position",
    (
        (density.SliceArguments(a=0.2), 0, 1),
        (density.SliceArguments(b=0.5), 1, 2),
        (density.SliceArguments(c=1.3), 2, 1),
    ),
)
def test_contour(visualizer, slice_args, index, position, Assert):
    dataDict = {
        (selection[0] if selection else ""): visualizer.ref.selector[selection].T
        for selection in visualizer.ref.selections
    }
    graph = visualizer.to_contour(dataDict, slice_args)
    _check_contour(graph, visualizer, index, position, Assert)


def _check_contour(graph, visualizer, index, position, Assert):
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
    dataDict = {
        (selection[0] if selection else ""): visualizer.ref.selector[selection].T
        for selection in visualizer.ref.selections
    }
    slice_args, index = (density.SliceArguments(c=1.3, supercell=supercell), 2)
    graph = visualizer.to_contour(
        dataDict,
        slice_args=slice_args,
    )
    _check_contour_supercell(graph, visualizer, index, supercell, Assert)


def _check_contour_supercell(graph, visualizer, index, supercell, Assert):
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
    dataDict = {
        (selection[0] if selection else ""): visualizer.ref.selector[selection].T
        for selection in visualizer.ref.selections
    }
    slice_args = density.SliceArguments(c=1.3, normal=normal)
    graph = visualizer.to_contour(dataDict, slice_args)
    _check_contour_normal(graph, visualizer, normal, Assert)


def _check_contour_normal(graph, visualizer, normal, Assert):
    assert len(graph) == len(visualizer.ref.selections)
    for series in graph.series:
        expected_plane = slicing.plane(
            visualizer.ref.structure.lattice_vectors(), "c", normal
        )
        Assert.allclose(series.lattice.cell, expected_plane.cell)
        Assert.allclose(series.lattice.vectors, expected_plane.vectors)


def test_quiver(visualizer_quiver, Assert):
    dataDict = {
        (selection[0] if selection else ""): visualizer_quiver.ref.selector[selection].T
        for selection in visualizer_quiver.ref.selections
    }
    slice_args, index, position = (density.SliceArguments(c=1.3), 2, 1)
    graph = visualizer_quiver.to_quiver(dataDict, slice_args)
    _check_quiver(graph, visualizer_quiver, position, index, Assert)


def _check_quiver(graph, visualizer_quiver, position, index, Assert):
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
    dataDict = {
        (selection[0] if selection else ""): visualizer_quiver.ref.selector[selection].T
        for selection in visualizer_quiver.ref.selections
    }
    slice_args, index = (density.SliceArguments(c=1.3, supercell=supercell), 2)
    graph = visualizer_quiver.to_quiver(
        dataDict,
        slice_args,
    )
    _check_quiver_supercell(graph, visualizer_quiver, supercell, index, Assert)


def _check_quiver_supercell(graph, visualizer_quiver, supercell, index, Assert):
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
    dataDict = {
        (selection[0] if selection else ""): visualizer_quiver.ref.selector[selection].T
        for selection in visualizer_quiver.ref.selections
    }
    slice_args = density.SliceArguments(c=1.3, normal=normal)
    graph = visualizer_quiver.to_quiver(dataDict, slice_args)
    _check_quiver_normal(graph, visualizer_quiver, normal, Assert)


def _check_quiver_normal(graph, visualizer_quiver, normal, Assert):
    assert len(graph) == len(visualizer_quiver.ref.selections)
    for series in graph.series:
        expected_plane = slicing.plane(
            visualizer_quiver.ref.structure.lattice_vectors(), "c", normal
        )
        Assert.allclose(series.lattice.cell, expected_plane.cell)
        Assert.allclose(series.lattice.vectors, expected_plane.vectors)


def test_quiver_collinear(visualizer, Assert):
    dataDict = {
        (selection[0] if selection else ""): visualizer.ref.selector[selection].T
        for selection in visualizer.ref.selections
    }
    slice_args = density.SliceArguments(c=1.3)
    graph = visualizer.to_quiver(dataDict, slice_args)


def test_slice_arguments():
    with pytest.raises(IncorrectUsage):
        density.SliceArguments(a=1.0, b=1.0)
    with pytest.raises(IncorrectUsage):
        density.SliceArguments(b=1.0, c=1.0)
    with pytest.raises(IncorrectUsage):
        density.SliceArguments(a=1.0, c=1.0)
    with pytest.raises(IncorrectUsage):
        density.SliceArguments(a=1.0, b=1.0, c=1.0)
    with pytest.raises(IncorrectUsage):
        density.SliceArguments()

    slice_args = density.SliceArguments(b=1.0, supercell=(1, 1, 1))
    assert slice_args.a is None
    assert slice_args.c is None

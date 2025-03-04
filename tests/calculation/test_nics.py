# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import dataclasses
import types

import numpy as np
import pytest

from py4vasp import _config
from py4vasp._calculation.nics import Nics
from py4vasp._calculation.structure import Structure
from py4vasp._third_party import view


@pytest.fixture
def chemical_shift(raw_data):
    raw_nics = raw_data.nics("Sr2TiO4")
    nics = Nics.from_data(raw_nics)
    nics.ref = types.SimpleNamespace()
    transposed_nics = np.array(raw_nics.nics).T
    nics.ref.structure = Structure.from_data(raw_nics.structure)
    nics.ref.output = {"nics": transposed_nics.reshape((10, 12, 14, 3, 3))}
    return nics


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


def test_read(chemical_shift, Assert):
    actual = chemical_shift.read()
    actual_structure = actual.pop("structure")
    Assert.same_structure(actual_structure, chemical_shift.ref.structure.read())
    assert actual.keys() == chemical_shift.ref.output.keys()
    Assert.allclose(actual["nics"], chemical_shift.ref.output["nics"])


def get_3d_tensor_element_from_grid(tensor, element: str):
    if element == "3x3":
        return tensor
    if element == "xx":
        return tensor[:, :, :, 0, 0]
    elif element == "xy":
        return tensor[:, :, :, 0, 1]
    elif element == "xz":
        return tensor[:, :, :, 0, 2]
    elif element == "yx":
        return tensor[:, :, :, 1, 0]
    elif element == "yy":
        return tensor[:, :, :, 1, 1]
    elif element == "yz":
        return tensor[:, :, :, 1, 2]
    elif element == "zx":
        return tensor[:, :, :, 2, 0]
    elif element == "zy":
        return tensor[:, :, :, 2, 1]
    elif element == "zz":
        return tensor[:, :, :, 2, 2]
    elif element == "xx + yy":
        return tensor[:, :, :, 0, 0] + tensor[:, :, :, 1, 1]
    elif element == "xx yy":
        return [tensor[:, :, :, 0, 0], tensor[:, :, :, 1, 1]]
    elif element in [None, "isotropic"]:
        tensor_sum = (
            tensor[:, :, :, 0, 0] + tensor[:, :, :, 1, 1] + tensor[:, :, :, 2, 2]
        )
        return tensor_sum / 3.0
    else:
        raise ValueError(
            f"Element {element} is unknown by get_3d_tensor_element_from_grid."
        )


def test_plot(chemical_shift, selection, Assert):
    tensor = chemical_shift.ref.output["nics"]
    element = get_3d_tensor_element_from_grid(tensor, selection)
    structure_view = chemical_shift.plot(selection)
    expected_view = chemical_shift.ref.structure.plot()
    Assert.same_structure_view(structure_view, expected_view)
    if not (isinstance(element, list)):
        element = [element]
        selection_list = [selection]
    else:
        selection_list = str.split(selection)
    assert len(structure_view.grid_scalars) == len(element)
    for grid_scalar, e, s in zip(structure_view.grid_scalars, element, selection_list):
        assert grid_scalar.label == (f"{s} NICS" if s else "isotropic NICS")
        assert grid_scalar.quantity.ndim == 4
        Assert.allclose(grid_scalar.quantity, e)
        assert len(grid_scalar.isosurfaces) == 2
        assert grid_scalar.isosurfaces == [
            view.Isosurface(1.0, _config.VASP_COLORS["blue"], 0.6),
            view.Isosurface(-1.0, _config.VASP_COLORS["red"], 0.6),
        ]


@pytest.mark.parametrize("supercell", (2, (3, 1, 2)))
def test_plot_supercell(chemical_shift, supercell, Assert):
    view = chemical_shift.plot(supercell=supercell)
    Assert.allclose(view.supercell, supercell)


def test_plot_user_options(chemical_shift):
    view = chemical_shift.plot(isolevel=0.9, opacity=0.2)
    assert len(view.grid_scalars) == 1
    grid_scalar = view.grid_scalars[0]
    assert len(grid_scalar.isosurfaces) == 2
    for idx, isosurface in enumerate(grid_scalar.isosurfaces):
        assert isosurface.isolevel == (-1.0) ** (idx) * 0.9
        assert isosurface.opacity == 0.2


@pytest.mark.parametrize(
    "kwargs, index, position",
    (({"a": 0.1}, 0, 1), ({"b": 0.7}, 1, 8), ({"c": 1.3}, 2, 4)),
)
def test_to_contour(chemical_shift, kwargs, index, position, Assert, selection):
    graph = chemical_shift.to_contour(selection=selection, **kwargs)
    slice_ = [slice(None), slice(None), slice(None)]
    slice_[index] = position
    tensor = chemical_shift.ref.output["nics"]
    scalar_data = get_3d_tensor_element_from_grid(tensor, selection)
    if not (isinstance(scalar_data, list)):
        scalar_data = [scalar_data[tuple(slice_)]]
        selection_list = [selection]
    else:
        scalar_data = [s[tuple(slice_)] for s in scalar_data]
        selection_list = str.split(selection)
    assert len(graph) == len(scalar_data)
    for series, e, s in zip(graph, scalar_data, selection_list):
        assert series.label == (
            f"{s if s else 'isotropic'} NICS contour ({list(kwargs.keys())[0]})"
        )
        Assert.allclose(series.data, e)


def test_to_contour_supercell(chemical_shift, Assert):
    graph = chemical_shift.to_contour(b=0, supercell=2)
    Assert.allclose(graph.series[0].supercell, (2, 2))
    graph = chemical_shift.to_contour(b=0, supercell=(2, 1))
    Assert.allclose(graph.series[0].supercell, (2, 1))


def test_to_contour_normal(chemical_shift, normal_vector, Assert):
    graph = chemical_shift.to_contour(c=0.5, normal=normal_vector.normal)
    rotation = normal_vector.expected_rotation
    lattice_vectors = chemical_shift.ref.structure.lattice_vectors()
    expected_lattice = lattice_vectors[:2, :2] @ rotation
    Assert.allclose(graph.series[0].lattice.vectors, expected_lattice)


@pytest.fixture(
    params=[
        None,
        "xx",
        "xy",
        "xz",
        "yx",
        "yy",
        "yz",
        "zx",
        "zy",
        "zz",
        "xx + yy",
        "xx yy",
        "isotropic",
    ],
)
def selection(request):
    return request.param


def test_to_numpy(selection, chemical_shift, Assert):
    tensor = chemical_shift.ref.output["nics"]
    element = get_3d_tensor_element_from_grid(tensor, selection or "3x3")
    Assert.allclose(chemical_shift.to_numpy(selection), element)


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.nics("Sr2TiO4")
    check_factory_methods(Nics, data)


def test_print(chemical_shift, format_):
    actual, _ = format_(chemical_shift)
    expected_text = """\
nucleus-independent chemical shift:
    structure: Sr2TiO4
    grid: 10, 12, 14
    tensor shape: 3x3"""
    assert actual == {"text/plain": expected_text}

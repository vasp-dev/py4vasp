# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
import types

import numpy as np
import pytest

from py4vasp import _config, calculation, exception, raw
from py4vasp._third_party.view import Isosurface


@pytest.fixture(params=[None, "tau"])
def density_source(request):
    return request.param


@pytest.fixture(params=["Sr2TiO4", "Fe3O4 collinear", "Fe3O4 noncollinear"])
def reference_density(raw_data, density_source, request):
    return make_reference_density(raw_data, request.param, density_source)


@pytest.fixture
def nonpolarized_density(raw_data):
    return make_reference_density(raw_data, "Sr2TiO4")


@pytest.fixture
def collinear_density(raw_data, density_source):
    return make_reference_density(raw_data, "Fe3O4 collinear", density_source)


@pytest.fixture
def noncollinear_density(raw_data, density_source):
    return make_reference_density(raw_data, "Fe3O4 noncollinear", density_source)


@pytest.fixture
def empty_density(raw_data):
    raw_density = raw.Density(raw_data.structure("Sr2TiO4"), charge=raw.VaspData(None))
    return calculation.density.from_data(raw_density)


@dataclasses.dataclass
class Expectation:
    label: str
    density: np.ndarray
    isosurfaces: list


def make_reference_density(raw_data, selection, source=None):
    raw_density = raw_data.density(selection)
    density = calculation.density.from_data(raw_density)
    density.ref = types.SimpleNamespace()
    density.ref.structure = calculation.structure.from_data(raw_density.structure)
    density.ref.output = get_expected_dict(raw_density.charge, source)
    density.ref.string = get_expected_string(selection, source)
    density.ref.selections = get_expected_selections(raw_density.charge)
    density._data_context.selection = source
    density.ref.source = source or "charge"
    return density


def get_expected_dict(charge, source):
    if source:
        return {source: np.array([component.T for component in charge])}
    else:
        if len(charge) == 1:  # nonpolarized
            return {"charge": charge[0].T}
        if len(charge) == 2:  # collinear
            return {"charge": charge[0].T, "magnetization": charge[1].T}
        # noncollinear
        magnetization = np.moveaxis(charge[1:].T, -1, 0)
        return {"charge": charge[0].T, "magnetization": magnetization}


def get_expected_string(selection, source):
    structure, *density = selection.split()
    if source == "tau":
        density = "Kinetic energy"
    elif not density:
        density = "Nonpolarized"
    else:
        density = density[0].capitalize()
    return f"""{density} density:
    structure: {structure}
    grid: 10, 12, 14"""


def get_expected_selections(charge):
    result = {"density": list(raw.selections("density")), "component": ["0"]}
    if len(charge) == 2:  # collinear
        result["component"] += ["3"]
    if len(charge) == 4:  # noncollinear
        result["component"] += ["1", "2", "3"]
    return result


def test_read(reference_density, Assert):
    actual = reference_density.read()
    actual_structure = actual.pop("structure")
    Assert.same_structure(actual_structure, reference_density.ref.structure.read())
    assert actual.keys() == reference_density.ref.output.keys()
    for key in actual:
        Assert.allclose(actual[key], reference_density.ref.output[key])


def test_empty_density(empty_density):
    with pytest.raises(exception.NoData):
        empty_density.read()


@pytest.mark.parametrize("selection", [None, "0", "unity", "sigma_0", "scalar"])
def test_charge_plot(selection, reference_density, Assert):
    source = reference_density.ref.source
    isosurfaces = [Isosurface(isolevel=0.2, color=_config.VASP_CYAN, opacity=0.6)]
    if source == "charge":
        expected = Expectation(
            label=selection if selection else "charge",
            density=reference_density.ref.output[source],
            isosurfaces=isosurfaces,
        )
    else:
        expected = Expectation(
            label=source + (f"({selection})" if selection else ""),
            density=reference_density.ref.output[source][0],
            isosurfaces=isosurfaces,
        )
    if selection:
        check_view(reference_density, expected, Assert, selection=selection)
    else:
        check_view(reference_density, expected, Assert)


def check_view(density, expected, Assert, **kwargs):
    view = density.plot(**kwargs)
    expected_view = density.ref.structure.plot(kwargs.get("supercell"))
    Assert.same_structure_view(view, expected_view)
    assert len(view.grid_scalars) == 1
    grid_scalar = view.grid_scalars[0]
    assert grid_scalar.label == expected.label
    assert grid_scalar.quantity.ndim == 4
    Assert.allclose(grid_scalar.quantity, expected.density)
    assert len(grid_scalar.isosurfaces) == len(expected.isosurfaces)
    assert grid_scalar.isosurfaces == expected.isosurfaces


def test_accessing_spin_raises_error(nonpolarized_density):
    with pytest.raises(exception.NoData):
        nonpolarized_density.plot("3")


@pytest.mark.parametrize(
    "selection", ["3", "sigma_z", "z", "sigma_3", "magnetization", "mag", "m"]
)
def test_collinear_plot(selection, collinear_density, Assert):
    source = collinear_density.ref.source
    isosurfaces = [
        Isosurface(isolevel=0.1, color=_config.VASP_BLUE, opacity=0.6),
        Isosurface(isolevel=-0.1, color=_config.VASP_RED, opacity=0.6),
    ]
    if source == "charge":
        expected = Expectation(
            label=selection,
            density=collinear_density.ref.output["magnetization"],
            isosurfaces=isosurfaces,
        )
    else:
        expected = Expectation(
            label=f"{source}({selection})",
            density=collinear_density.ref.output[source][1],
            isosurfaces=isosurfaces,
        )
        if selection in ("magnetization", "mag", "m"):
            # magnetization not allowed for tau
            return
    check_view(collinear_density, expected, Assert, selection=selection, isolevel=0.1)


def test_accessing_noncollinear_element_raises_error(collinear_density):
    with pytest.raises(exception.NoData):
        collinear_density.plot("1")


@pytest.mark.parametrize(
    "selections",
    [
        ("1", "2", "3"),
        ("sigma_x", "sigma_y", "sigma_z"),
        ("x", "y", "z"),
        ("sigma_1", "sigma_2", "sigma_3"),
        (
            "m(1)",
            "mag(2)",
            "magnetization(3)",
        ),  # the magnetization label should be ignored
    ],
)
def test_plotting_noncollinear_density(selections, noncollinear_density, Assert):
    source = noncollinear_density.ref.source
    if source == "charge":
        if "(" in selections[0]:  # magnetization filtered from selections
            expected_labels = ("1", "2", "3")
        else:
            expected_labels = selections

        expected_density = noncollinear_density.ref.output["magnetization"]
    else:
        expected_labels = (f"{source}({selection})" for selection in selections)
        expected_density = noncollinear_density.ref.output[source][1:]
        if "(" in selections[0]:  # magnetization not allowed for tau
            return
    isosurfaces = [
        Isosurface(isolevel=0.2, color=_config.VASP_BLUE, opacity=0.3),
        Isosurface(isolevel=-0.2, color=_config.VASP_RED, opacity=0.3),
    ]
    for selection, density, label in zip(selections, expected_density, expected_labels):
        expected = Expectation(label, density, isosurfaces)
        kwargs = {"selection": selection, "opacity": 0.3}
        check_view(noncollinear_density, expected, Assert, **kwargs)


def test_adding_components(noncollinear_density, Assert):
    source = noncollinear_density.ref.source
    if source == "charge":
        expected_label = "1 + 2"
        expected_density = noncollinear_density.ref.output["magnetization"]
    else:
        expected_label = f"{source}(1 + 2)"
        expected_density = noncollinear_density.ref.output[source][1:]
    expected = Expectation(
        label=expected_label,
        density=expected_density[0] + expected_density[1],
        isosurfaces=[Isosurface(isolevel=0.4, color=_config.VASP_CYAN, opacity=0.6)],
    )
    check_view(noncollinear_density, expected, Assert, selection="1 + 2", isolevel=0.4)


@pytest.mark.parametrize("supercell", [2, (3, 2, 1)])
def test_plotting_supercell(supercell, reference_density, Assert):
    source = reference_density.ref.source
    isosurfaces = [Isosurface(isolevel=0.2, color=_config.VASP_CYAN, opacity=0.6)]
    if source == "charge":
        expected = Expectation(
            label=source,
            density=reference_density.ref.output[source],
            isosurfaces=isosurfaces,
        )
    else:
        expected = Expectation(
            label=source,
            density=reference_density.ref.output[source][0],
            isosurfaces=isosurfaces,
        )
    check_view(reference_density, expected, Assert, supercell=supercell)


@pytest.mark.parametrize(
    "kwargs, index, position",
    (({"a": 0.1}, 0, 1), ({"b": 0.7}, 1, 8), ({"c": 1.3}, 2, 4)),
)
def test_contour_of_charge(nonpolarized_density, kwargs, index, position, Assert):
    graph = nonpolarized_density.to_contour(**kwargs)
    slice_ = [slice(None), slice(None), slice(None)]
    slice_[index] = position
    data = nonpolarized_density.ref.output["charge"][tuple(slice_)]
    assert len(graph) == 1
    series = graph.series[0]
    Assert.allclose(series.data, data)
    lattice_vectors = nonpolarized_density.ref.structure.lattice_vectors()
    lattice_vectors = np.delete(lattice_vectors, index, axis=0)
    expected_products = lattice_vectors @ lattice_vectors.T
    actual_products = series.lattice @ series.lattice.T
    Assert.allclose(actual_products, expected_products)
    assert series.label == "charge"


def test_incorrect_slice_raises_error(nonpolarized_density):
    with pytest.raises(exception.IncorrectUsage):
        nonpolarized_density.to_contour()
    with pytest.raises(exception.IncorrectUsage):
        nonpolarized_density.to_contour(a=1, b=2)
    with pytest.raises(exception.IncorrectUsage):
        nonpolarized_density.to_contour(3)


@pytest.mark.parametrize(
    "selection", ["3", "sigma_z", "z", "sigma_3", "magnetization", "mag", "m"]
)
def test_collinear_to_contour(selection, collinear_density, Assert):
    source = collinear_density.ref.source
    if source == "charge":
        expected_label = selection
        expected_data = collinear_density.ref.output["magnetization"][:, :, 7]
    else:
        expected_label = f"{source}({selection})"
        expected_data = collinear_density.ref.output[source][1, :, :, 7]
        if selection in ("magnetization", "mag", "m"):
            # magnetization not allowed for tau
            return
    expected_lattice = collinear_density.ref.structure.lattice_vectors()[:2, :2]
    graph = collinear_density.to_contour(selection, c=-0.5)
    assert len(graph) == 1
    series = graph.series[0]
    Assert.allclose(series.data, expected_data)
    Assert.allclose(series.lattice, expected_lattice)
    assert series.label == expected_label


@pytest.mark.parametrize(
    "selections",
    [
        ("1", "2", "3"),
        ("sigma_x", "sigma_y", "sigma_z"),
        ("x", "y", "z"),
        ("sigma_1", "sigma_2", "sigma_3"),
        (
            "m(1)",
            "mag(2)",
            "magnetization(3)",
        ),  # the magnetization label should be ignored
    ],
)
def test_noncollinear_to_contour(noncollinear_density, selections, Assert):
    source = noncollinear_density.ref.source
    if source == "charge":
        if "(" in selections[0]:  # magnetization filtered from selections
            expected_labels = ("1", "2", "3")
        else:
            expected_labels = selections
        expected_data = noncollinear_density.ref.output["magnetization"][:, :, 5, :]
    else:
        expected_labels = (f"{source}({selection})" for selection in selections)
        expected_data = noncollinear_density.ref.output[source][1:, :, 5, :]
        if "(" in selections[0]:  # magnetization not allowed for tau
            return
    graph = noncollinear_density.to_contour(" ".join(selections), b=0.4)
    expected_lattice = noncollinear_density.ref.structure.lattice_vectors()[::2, ::2]
    assert len(graph) == len(expected_data)
    for density, label, series in zip(expected_data, expected_labels, graph.series):
        Assert.allclose(series.data, density)
        Assert.allclose(series.lattice, expected_lattice)
        assert series.label == label


def test_to_numpy(reference_density, Assert):
    source = reference_density.ref.source
    if source == "charge":
        if reference_density.is_nonpolarized():
            expected_density = [reference_density.ref.output["charge"]]
        elif reference_density.is_collinear():
            expected_density = [
                reference_density.ref.output["charge"],
                reference_density.ref.output["magnetization"],
            ]
        else:
            expected_density = [
                reference_density.ref.output["charge"],
                *reference_density.ref.output["magnetization"],
            ]
    else:
        expected_density = reference_density.ref.output[source]
    Assert.allclose(reference_density.to_numpy(), expected_density)


def test_selections(reference_density):
    assert reference_density.selections() == reference_density.ref.selections


def test_selections_empty_density(empty_density):
    assert empty_density.selections() == {"density": list(raw.selections("density"))}


def test_missing_element(reference_density):
    with pytest.raises(exception.IncorrectUsage):
        reference_density.plot("unknown tag")


def test_color_specified_for_sigma_z(collinear_density):
    with pytest.raises(exception.NotImplemented):
        collinear_density.plot("3", color="brown")


@pytest.mark.parametrize("selection", ("m", "mag", "magnetization"))
def test_magnetization_without_component(selection, raw_data):
    data = raw_data.density("Fe3O4 noncollinear")
    with pytest.raises(exception.IncorrectUsage):
        calculation.density.from_data(data).plot(selection)


def test_print(reference_density, format_):
    actual, _ = format_(reference_density)
    assert actual == {"text/plain": reference_density.ref.string}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.density("Fe3O4 collinear")
    parameters = {"to_contour": {"a": 0.3}}
    check_factory_methods(calculation.density, data, parameters)

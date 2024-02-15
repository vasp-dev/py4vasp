# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import numpy as np
import pytest

from py4vasp import _config, calculation, exception, raw
from py4vasp._third_party.view import Isosurface, View


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
    if source == "charge":
        expected_density = reference_density.ref.output[source].T
    else:
        expected_density = reference_density.ref.output[source][0].T
    if selection:
        view = reference_density.plot(selection)
    else:
        view = reference_density.plot()
    structure_view = reference_density.ref.structure.plot()
    assert np.all(structure_view.elements == view.elements)
    Assert.allclose(structure_view.lattice_vectors, view.lattice_vectors)
    Assert.allclose(structure_view.positions, view.positions)
    assert len(view.grid_scalars) == 1
    grid_scalar = view.grid_scalars[0]
    assert grid_scalar.label == source
    assert grid_scalar.quantity.ndim == 4
    Assert.allclose(grid_scalar.quantity, expected_density)
    assert len(grid_scalar.isosurfaces) == 1
    isosurface = Isosurface(isolevel=0.2, color=_config.VASP_CYAN, opacity=0.6)
    assert grid_scalar.isosurfaces[0] == isosurface


def test_accessing_spin_raises_error(nonpolarized_density, not_core):
    with pytest.raises(exception.NoData):
        nonpolarized_density.plot("3")


@pytest.mark.parametrize(
    "selection", ["3", "sigma_z", "z", "sigma_3", "magnetization", "mag", "m"]
)
def test_collinear_plot(selection, collinear_density, Assert):
    source = collinear_density.ref.source
    if source == "charge":
        source = "magnetization"
        expected_density = collinear_density.ref.output[source].T
    else:
        expected_density = collinear_density.ref.output[source][1].T
        if selection in (
            "magnetization",
            "mag",
            "m",
        ):  # magnetization not allowed for tau
            return
    view = collinear_density.plot(selection, isolevel=0.1)
    structure_view = collinear_density.ref.structure.plot()
    assert np.all(structure_view.elements == view.elements)
    Assert.allclose(structure_view.lattice_vectors, view.lattice_vectors)
    Assert.allclose(structure_view.positions, view.positions)
    assert len(view.grid_scalars) == 1
    grid_scalar = view.grid_scalars[0]
    assert grid_scalar.label == source
    assert grid_scalar.quantity.ndim == 4
    Assert.allclose(grid_scalar.quantity, expected_density)
    isosurfaces = [
        Isosurface(isolevel=0.1, color=_config.VASP_BLUE, opacity=0.6),
        Isosurface(isolevel=-0.1, color=_config.VASP_RED, opacity=0.6),
    ]
    assert grid_scalar.isosurfaces == isosurfaces


def test_accessing_noncollinear_element_raises_error(collinear_density, not_core):
    with pytest.raises(exception.NoData):
        collinear_density.plot("1")


@pytest.mark.xfail
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
def test_plotting_noncollinear_density(
    selections, noncollinear_density, mock_viewer, Assert, not_core
):
    source = noncollinear_density.ref.source
    if source == "charge":
        expected_density = noncollinear_density.ref.output["magnetization"]
    else:
        expected_density = noncollinear_density.ref.output[source][1:]
        if "(" in selections[0]:  # magnetization not allowed for tau
            return
    for component, selection in enumerate(selections):
        result = noncollinear_density.plot(selection, isolevel=0.1, smooth=1)
        assert isinstance(result, viewer3d.Viewer3d)
        calls = mock_viewer["surface"].call_args_list
        check_magnetization_plot(expected_density[component].T, calls, Assert)
        mock_viewer["surface"].reset_mock()


def check_magnetization_plot(magnetization, calls, Assert):
    assert len(calls) == 2
    args, kwargs = calls[0]
    Assert.allclose(args[0], magnetization)
    blue = _config.VASP_BLUE
    assert kwargs == {"isolevel": 0.1, "color": blue, "opacity": 0.6, "smooth": 1}
    args, kwargs = calls[1]
    Assert.allclose(args[0], -magnetization)
    red = _config.VASP_RED
    assert kwargs == {"isolevel": 0.1, "color": red, "opacity": 0.6, "smooth": 1}


@pytest.mark.xfail
def test_adding_components(noncollinear_density, mock_viewer, Assert, not_core):
    source = noncollinear_density.ref.source
    if source == "charge":
        expected_density = noncollinear_density.ref.output["magnetization"]
    else:
        expected_density = noncollinear_density.ref.output[source][1:]
    expected_density = expected_density[0] + expected_density[1]
    result = noncollinear_density.plot("1 + 2", isolevel=0.4)
    assert isinstance(result, viewer3d.Viewer3d)
    mock_viewer["surface"].assert_called_once()
    args, kwargs = mock_viewer["surface"].call_args
    Assert.allclose(args[0], expected_density.T)
    assert kwargs == {"isolevel": 0.4, "color": _config.VASP_CYAN, "opacity": 0.6}


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


def test_missing_element(reference_density, not_core):
    with pytest.raises(exception.IncorrectUsage):
        reference_density.plot("unknown tag")


@pytest.mark.xfail
def test_color_specified_for_sigma_z(collinear_density, not_core):
    with pytest.raises(exception.NotImplemented):
        collinear_density.plot("3", color="brown")


@pytest.mark.parametrize("selection", ("m", "mag", "magnetization"))
def test_magnetization_without_component(selection, raw_data, not_core):
    data = raw_data.density("Fe3O4 noncollinear")
    with pytest.raises(exception.IncorrectUsage):
        calculation.density.from_data(data).plot(selection)


def test_print(reference_density, format_):
    actual, _ = format_(reference_density)
    assert actual == {"text/plain": reference_density.ref.string}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.density("Fe3O4 collinear")
    check_factory_methods(calculation.density, data)

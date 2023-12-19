# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp import exception, raw
from py4vasp._data import viewer3d
from py4vasp.data import Density, Structure


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
    return Density.from_data(raw_density)


@pytest.fixture
def mock_viewer():
    obj = viewer3d.Viewer3d
    cm_init = patch.object(obj, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(obj, "show_cell")
    cm_surface = patch.object(obj, "show_isosurface")
    with cm_init as init, cm_cell as cell, cm_surface as surface:
        yield {"init": init, "cell": cell, "surface": surface}


def make_reference_density(raw_data, selection, source=None):
    raw_density = raw_data.density(selection)
    density = Density.from_data(raw_density)
    density.ref = types.SimpleNamespace()
    density.ref.structure = Structure.from_data(raw_density.structure).read()
    density.ref.output = get_expected_dict(raw_density.charge, source)
    density.ref.string = get_expected_string(raw_density.charge)
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


def get_expected_string(charge):
    if len(charge) == 1:
        return """\
Nonpolarized density:
    structure: Sr2TiO4
    grid: 10, 12, 14"""
    elif len(charge) == 2:
        return """\
Collinear density:
    structure: Fe3O4
    grid: 10, 12, 14"""
    else:
        return """\
Noncollinear density:
    structure: Fe3O4
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
    Assert.same_structure(actual_structure, reference_density.ref.structure)
    assert actual.keys() == reference_density.ref.output.keys()
    for key in actual:
        Assert.allclose(actual[key], reference_density.ref.output[key])


def test_empty_density(empty_density):
    with pytest.raises(exception.NoData):
        empty_density.read()


@pytest.mark.parametrize("selection", [None, "0", "unity", "sigma_0", "scalar"])
def test_charge_plot(selection, reference_density, mock_viewer, Assert, not_core):
    source = reference_density.ref.source
    if source == "charge":
        expected_density = reference_density.ref.output["charge"].T
    else:
        expected_density = reference_density.ref.output[source][0].T
    if selection:
        result = reference_density.plot(selection)
    else:
        result = reference_density.plot()
    assert isinstance(result, viewer3d.Viewer3d)
    mock_viewer["init"].assert_called_once()
    mock_viewer["cell"].assert_called_once()
    mock_viewer["surface"].assert_called_once()
    args, kwargs = mock_viewer["surface"].call_args
    Assert.allclose(args[0], expected_density)
    assert kwargs == {"isolevel": 0.2, "color": "yellow", "opacity": 0.6}


def test_accessing_spin_raises_error(nonpolarized_density):
    with pytest.raises(exception.NoData):
        nonpolarized_density.plot("3")


@pytest.mark.parametrize(
    "selection", ["3", "sigma_z", "z", "sigma_3", "magnetization", "mag", "m"]
)
def test_collinear_plot(selection, collinear_density, mock_viewer, Assert, not_core):
    source = collinear_density.ref.source
    if source == "charge":
        expected_density = collinear_density.ref.output["magnetization"].T
    else:
        expected_density = collinear_density.ref.output[source][1].T
        if selection in ("magnetization", "mag", "m"):
            return
    result = collinear_density.plot(selection, isolevel=0.1, smooth=1)
    assert isinstance(result, viewer3d.Viewer3d)
    calls = mock_viewer["surface"].call_args_list
    check_magnetization_plot(expected_density, calls, Assert)


@pytest.mark.parametrize(
    "selections",
    [
        ("1", "2", "3"),
        ("sigma_x", "sigma_y", "sigma_z"),
        ("x", "y", "z"),
        ("sigma_1", "sigma_2", "sigma_3"),
    ],
)
def test_plotting_noncollinear_density(
    selections, noncollinear_density, mock_viewer, Assert
):
    source = noncollinear_density.ref.source
    if source == "charge":
        expected_density = noncollinear_density.ref.output["magnetization"]
    else:
        expected_density = noncollinear_density.ref.output[source][1:]
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
    assert kwargs == {"isolevel": 0.1, "color": "blue", "opacity": 0.6, "smooth": 1}
    args, kwargs = calls[1]
    Assert.allclose(args[0], -magnetization)
    assert kwargs == {"isolevel": 0.1, "color": "red", "opacity": 0.6, "smooth": 1}


def test_selections(reference_density):
    assert reference_density.selections() == reference_density.ref.selections


def test_selections_empty_density(empty_density):
    assert empty_density.selections() == {"density": list(raw.selections("density"))}


def test_missing_element(reference_density, not_core):
    with pytest.raises(exception.IncorrectUsage):
        reference_density.plot("unknown tag")


def test_color_specified_for_sigma_z(collinear_density, not_core):
    with pytest.raises(exception.NotImplemented):
        collinear_density.plot("3", color="brown")


def test_print(reference_density, format_):
    actual, _ = format_(reference_density)
    assert actual == {"text/plain": reference_density.ref.string}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.density("Fe3O4 collinear")
    check_factory_methods(Density, data)

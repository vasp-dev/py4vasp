# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp import exception, raw
from py4vasp._data import viewer3d
from py4vasp.data import Potential, Structure


@pytest.fixture(params=["total", "ionic", "hartree", "xc", "all"])
def included_parts(request):
    return request.param


@pytest.fixture(params=["Sr2TiO4", "Fe3O4 collinear", "Fe3O4 noncollinear"])
def reference_potential(raw_data, request, included_parts):
    return make_reference_potential(raw_data, f"{request.param} {included_parts}")


def make_reference_potential(raw_data, selection):
    raw_potential = raw_data.potential(selection)
    potential = Potential.from_data(raw_potential)
    potential.ref = types.SimpleNamespace()
    potential.ref.included_parts = included_parts
    potential.ref.output = get_expected_dict(raw_potential)
    return potential


def get_expected_dict(raw_potential):
    return {
        "structure": Structure.from_data(raw_potential.structure).read(),
        **separate_potential("total", raw_potential.total_potential),
        **separate_potential("xc", raw_potential.xc_potential),
        **separate_potential("hartree", raw_potential.hartree_potential),
        **separate_potential("ionic", raw_potential.ionic_potential),
    }


def separate_potential(potential_name, potential):
    if potential.is_none():
        return {}
    if len(potential) == 1:  # nonpolarized
        return {potential_name: potential[0].T}
    if len(potential) == 2:  # spin-polarized
        return {
            potential_name: potential[0].T,
            f"{potential_name}_up": potential[0].T + potential[1].T,
            f"{potential_name}_down": potential[0].T - potential[1].T,
        }
    return {
        potential_name: potential[0].T,
        f"{potential_name}_magnetization": np.moveaxis(potential[1:].T, -1, 0),
    }


def test_read(reference_potential, Assert):
    actual = reference_potential.read()
    assert actual.keys() == reference_potential.ref.output.keys()
    for key in actual:
        if key == "structure":
            continue
        Assert.allclose(actual[key], reference_potential.ref.output[key])


def test_plot_total_potential(reference_potential, Assert, not_core):
    obj = viewer3d.Viewer3d
    cm_init = patch.object(obj, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(obj, "show_cell")
    cm_surface = patch.object(obj, "show_isosurface")
    with cm_init as init, cm_cell as cell, cm_surface as surface:
        reference_potential.plot()
        init.assert_called_once()
        cell.assert_called_once()
        surface.assert_called_once()
        args, kwargs = surface.call_args
    Assert.allclose(args[0], reference_potential.ref.output["total"])
    assert kwargs == {"isolevel": 0.0, "color": "yellow", "opacity": 0.6}


def test_plot_selected_potential(reference_potential, Assert, not_core):
    obj = viewer3d.Viewer3d
    cm_init = patch.object(obj, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(obj, "show_cell")
    cm_surface = patch.object(obj, "show_isosurface")
    if reference_potential.ref.included_parts in ("hartree", "ionic", "xc"):
        selection = reference_potential.ref.included_parts
    else:
        selection = "total"
    with cm_init as init, cm_cell as cell, cm_surface as surface:
        reference_potential.plot(selection, isolevel=0.2)
        init.assert_called_once()
        cell.assert_called_once()
        surface.assert_called_once()
        args, kwargs = surface.call_args
    Assert.allclose(args[0], reference_potential.ref.output[selection])
    assert kwargs == {"isolevel": 0.2, "color": "yellow", "opacity": 0.6}


@pytest.mark.parametrize("selection", ["up", "down"])
def test_plot_spin_potential(raw_data, selection, Assert, not_core):
    potential = make_reference_potential(raw_data, "Fe3O4 collinear total")
    obj = viewer3d.Viewer3d
    cm_init = patch.object(obj, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(obj, "show_cell")
    cm_surface = patch.object(obj, "show_isosurface")
    with cm_init as init, cm_cell as cell, cm_surface as surface:
        potential.plot(selection)
        init.assert_called_once()
        cell.assert_called_once()
        surface.assert_called_once()
        args, kwargs = surface.call_args
    Assert.allclose(args[0], potential.ref.output[f"total_{selection}"])
    assert kwargs == {"isolevel": 0.0, "color": "yellow", "opacity": 0.6}


def test_plot_multiple_selections(raw_data, Assert, not_core):
    potential = make_reference_potential(raw_data, "Fe3O4 collinear all")
    obj = viewer3d.Viewer3d
    cm_init = patch.object(obj, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(obj, "show_cell")
    cm_surface = patch.object(obj, "show_isosurface")
    with cm_init as init, cm_cell as cell, cm_surface as surface:
        potential.plot("up(total) hartree xc(down)")
        init.assert_called_once()
        cell.assert_called_once()
        calls = surface.call_args_list
    assert len(calls) == 3
    args, _ = calls[0]
    Assert.allclose(args[0], potential.ref.output["total_up"])
    args, _ = calls[1]
    Assert.allclose(args[0], potential.ref.output["hartree"])
    args, _ = calls[2]
    Assert.allclose(args[0], potential.ref.output["xc_down"])


def test_incorrect_selection(reference_potential, not_core):
    with pytest.raises(exception.IncorrectUsage):
        reference_potential.plot("random_string")


@pytest.mark.parametrize("selection", ["total", "xc", "ionic", "hartree"])
def test_empty_potential(raw_data, selection, not_core):
    raw_potential = raw_data.potential("Sr2TiO4 total")
    raw_potential.total_potential = raw.VaspData(None)
    potential = Potential.from_data(raw_potential)
    with pytest.raises(exception.NoData):
        potential.plot(selection)


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.potential("Fe3O4 collinear total")
    check_factory_methods(Potential, data)

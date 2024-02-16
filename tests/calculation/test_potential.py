# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
import types

import numpy as np
import pytest

from py4vasp import _config, calculation, exception, raw
from py4vasp._third_party.view import Isosurface


@pytest.fixture(params=["total", "ionic", "hartree", "xc", "all"])
def included_kinds(request):
    return request.param


@pytest.fixture(params=["Sr2TiO4", "Fe3O4 collinear", "Fe3O4 noncollinear"])
def reference_potential(raw_data, request, included_kinds):
    return make_reference_potential(raw_data, request.param, included_kinds)


@dataclasses.dataclass
class Expectation:
    label: str
    potential: np.ndarray
    isosurface: Isosurface


def make_reference_potential(raw_data, system, included_kinds):
    selection = f"{system} {included_kinds}"
    raw_potential = raw_data.potential(selection)
    potential = calculation.potential.from_data(raw_potential)
    potential.ref = types.SimpleNamespace()
    potential.ref.included_kinds = included_kinds
    potential.ref.output = get_expected_dict(raw_potential)
    potential.ref.string = get_expected_string(raw_potential, included_kinds)
    potential.ref.structure = calculation.structure.from_data(raw_potential.structure)
    return potential


def get_expected_dict(raw_potential):
    return {
        "structure": calculation.structure.from_data(raw_potential.structure).read(),
        **separate_potential("total", raw_potential.total_potential),
        **separate_potential("xc", raw_potential.xc_potential),
        **separate_potential("hartree", raw_potential.hartree_potential),
        **separate_potential("ionic", raw_potential.ionic_potential),
    }


def get_expected_string(raw_potential, included_parts):
    if len(raw_potential.total_potential) == 1:
        header = """\
nonpolarized potential:
    structure: Sr2TiO4"""
    elif len(raw_potential.total_potential) == 2:
        header = """\
collinear potential:
    structure: Fe3O4"""
    else:
        header = """\
noncollinear potential:
    structure: Fe3O4"""
    grid = "    grid: 10, 12, 14"
    if included_parts == "all":
        available = "    available: total, ionic, xc, hartree"
    elif included_parts == "total":
        available = "    available: total"
    else:
        available = f"    available: total, {included_parts}"
    return "\n".join([header, grid, available])


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


def test_plot_total_potential(reference_potential, Assert):
    view = reference_potential.plot()
    expected_view = reference_potential.ref.structure.plot()
    Assert.same_structure_view(view, expected_view)
    assert len(view.grid_scalars) == 1
    grid_scalar = view.grid_scalars[0]
    assert grid_scalar.label == "total potential"
    assert grid_scalar.quantity.ndim == 4
    Assert.allclose(grid_scalar.quantity, reference_potential.ref.output["total"].T)
    assert len(grid_scalar.isosurfaces) == 1
    isosurface = Isosurface(isolevel=0, color=_config.VASP_CYAN, opacity=0.6)
    assert grid_scalar.isosurfaces[0] == isosurface


def test_plot_selected_potential(reference_potential, Assert):
    if reference_potential.ref.included_kinds in ("hartree", "ionic", "xc"):
        selection = reference_potential.ref.included_kinds
    else:
        selection = "total"
    view = reference_potential.plot(selection, isolevel=0.2)
    expected_view = reference_potential.ref.structure.plot()
    Assert.same_structure_view(view, expected_view)
    assert len(view.grid_scalars) == 1
    grid_scalar = view.grid_scalars[0]
    assert grid_scalar.label == f"{selection} potential"
    assert grid_scalar.quantity.ndim == 4
    Assert.allclose(grid_scalar.quantity, reference_potential.ref.output[selection].T)
    assert len(grid_scalar.isosurfaces) == 1
    isosurface = Isosurface(isolevel=0.2, color=_config.VASP_CYAN, opacity=0.6)
    assert grid_scalar.isosurfaces[0] == isosurface


@pytest.mark.parametrize("selection", ["up", "down"])
def test_plot_spin_potential(raw_data, selection, Assert):
    potential = make_reference_potential(raw_data, "Fe3O4 collinear", "total")
    view = potential.plot(selection, opacity=0.3)
    expected_view = potential.ref.structure.plot()
    Assert.same_structure_view(view, expected_view)
    assert len(view.grid_scalars) == 1
    grid_scalar = view.grid_scalars[0]
    assert grid_scalar.label == f"total potential({selection})"
    assert grid_scalar.quantity.ndim == 4
    Assert.allclose(grid_scalar.quantity, potential.ref.output[f"total_{selection}"].T)
    assert len(grid_scalar.isosurfaces) == 1
    isosurface = Isosurface(isolevel=0.0, color=_config.VASP_CYAN, opacity=0.3)
    assert grid_scalar.isosurfaces[0] == isosurface


def test_plot_multiple_selections(raw_data, Assert):
    potential = make_reference_potential(raw_data, "Fe3O4 collinear", "all")
    view = potential.plot("up(total) hartree xc(down)")
    isosurface = Isosurface(isolevel=0.0, color=_config.VASP_CYAN, opacity=0.6)
    expectations = [
        Expectation(
            label="total potential(up)",
            potential=potential.ref.output["total_up"],
            isosurface=isosurface,
        ),
        Expectation(
            label="hartree potential",
            potential=potential.ref.output["hartree"],
            isosurface=isosurface,
        ),
        Expectation(
            label="xc potential(down)",
            potential=potential.ref.output["xc_down"],
            isosurface=isosurface,
        ),
    ]
    expected_view = potential.ref.structure.plot()
    Assert.same_structure_view(view, expected_view)
    assert len(view.grid_scalars) == len(expectations)
    for grid_scalar, expected in zip(view.grid_scalars, expectations):
        assert grid_scalar.label == expected.label
        assert grid_scalar.quantity.ndim == 4
        Assert.allclose(grid_scalar.quantity, expected.potential.T)
        assert len(grid_scalar.isosurfaces) == 1
        assert grid_scalar.isosurfaces[0] == expected.isosurface


def test_incorrect_selection(reference_potential):
    with pytest.raises(exception.IncorrectUsage):
        reference_potential.plot("random_string")


@pytest.mark.parametrize("selection", ["total", "xc", "ionic", "hartree"])
def test_empty_potential(raw_data, selection):
    raw_potential = raw_data.potential("Sr2TiO4 total")
    raw_potential.total_potential = raw.VaspData(None)
    potential = calculation.potential.from_data(raw_potential)
    with pytest.raises(exception.NoData):
        potential.plot(selection)


def test_print(reference_potential, format_):
    actual, _ = format_(reference_potential)
    assert actual == {"text/plain": reference_potential.ref.string}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.potential("Fe3O4 collinear total")
    check_factory_methods(calculation.potential, data)

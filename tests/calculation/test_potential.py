# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
import types

import numpy as np
import pytest

from py4vasp import _config, exception, raw
from py4vasp._calculation.potential import Potential
from py4vasp._calculation.structure import Structure
from py4vasp._third_party.view import Isosurface
from py4vasp._util import slicing


@pytest.fixture(params=["total", "ionic", "hartree", "xc", "all"])
def included_kinds(request):
    return request.param


@pytest.fixture(params=["Sr2TiO4", "Fe3O4 collinear", "Fe3O4 noncollinear"])
def reference_potential(raw_data, request, included_kinds):
    return make_reference_potential(raw_data, request.param, included_kinds)


@pytest.fixture
def nonpolarized_potential(raw_data):
    return make_reference_potential(raw_data, "Sr2TiO4", "all")


@pytest.fixture
def collinear_potential(raw_data):
    return make_reference_potential(raw_data, "Fe3O4 collinear", "all")


@pytest.fixture
def noncollinear_potential(raw_data):
    return make_reference_potential(raw_data, "Fe3O4 noncollinear", "all")


@dataclasses.dataclass
class Expectation:
    label: str
    potential: np.ndarray
    isosurface: Isosurface


def make_reference_potential(raw_data, system, included_kinds):
    selection = f"{system} {included_kinds}"
    raw_potential = raw_data.potential(selection)
    potential = Potential.from_data(raw_potential)
    potential.ref = types.SimpleNamespace()
    potential.ref.included_kinds = included_kinds
    potential.ref.output = get_expected_dict(raw_potential)
    potential.ref.string = get_expected_string(raw_potential, included_kinds)
    potential.ref.structure = Structure.from_data(raw_potential.structure)
    return potential


def get_expected_dict(raw_potential):
    return {
        "structure": Structure.from_data(raw_potential.structure).read(),
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
    color = _config.VASP_COLORS["cyan"]
    expectation = Expectation(
        label="total potential",
        potential=reference_potential.ref.output["total"],
        isosurface=Isosurface(isolevel=0, color=color, opacity=0.6),
    )
    check_view(reference_potential, view, [expectation], Assert)


def test_plot_selected_potential(reference_potential, Assert):
    if reference_potential.ref.included_kinds in ("hartree", "ionic", "xc"):
        selection = reference_potential.ref.included_kinds
    else:
        selection = "total"
    view = reference_potential.plot(selection, isolevel=0.2)
    color = _config.VASP_COLORS["cyan"]
    expectation = Expectation(
        label=f"{selection} potential",
        potential=reference_potential.ref.output[selection],
        isosurface=Isosurface(isolevel=0.2, color=color, opacity=0.6),
    )
    check_view(reference_potential, view, [expectation], Assert)


@pytest.mark.parametrize("selection", ["up", "down"])
def test_plot_spin_potential(raw_data, selection, Assert):
    potential = make_reference_potential(raw_data, "Fe3O4 collinear", "total")
    view = potential.plot(selection, opacity=0.3)
    color = _config.VASP_COLORS["cyan"]
    expectation = Expectation(
        label=f"total potential({selection})",
        potential=potential.ref.output[f"total_{selection}"],
        isosurface=Isosurface(isolevel=0.0, color=color, opacity=0.3),
    )
    check_view(potential, view, [expectation], Assert)


def test_plot_multiple_selections(raw_data, Assert):
    potential = make_reference_potential(raw_data, "Fe3O4 collinear", "all")
    view = potential.plot("up(total) hartree xc(down)")
    color = _config.VASP_COLORS["cyan"]
    isosurface = Isosurface(isolevel=0.0, color=color, opacity=0.6)
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
    check_view(potential, view, expectations, Assert)


@pytest.mark.parametrize("supercell", [2, (3, 2, 1)])
def test_plot_supercell(raw_data, supercell, Assert):
    potential = make_reference_potential(raw_data, "Sr2TiO4", "total")
    view = potential.plot(supercell=supercell)
    color = _config.VASP_COLORS["cyan"]
    expectation = Expectation(
        label="total potential",
        potential=potential.ref.output[f"total"],
        isosurface=Isosurface(isolevel=0, color=color, opacity=0.6),
    )
    check_view(potential, view, [expectation], Assert, supercell=supercell)


def check_view(potential, view, expectations, Assert, supercell=None):
    expected_view = potential.ref.structure.plot(supercell)
    Assert.same_structure_view(view, expected_view)
    assert len(view.grid_scalars) == len(expectations)
    for grid_scalar, expected in zip(view.grid_scalars, expectations):
        assert grid_scalar.label == expected.label
        assert grid_scalar.quantity.ndim == 4
        Assert.allclose(grid_scalar.quantity, expected.potential)
        assert len(grid_scalar.isosurfaces) == 1
        assert grid_scalar.isosurfaces[0] == expected.isosurface


def test_incorrect_selection(reference_potential):
    with pytest.raises(exception.IncorrectUsage):
        reference_potential.plot("random_string")


@pytest.mark.parametrize("selection", ["total", "xc", "ionic", "hartree"])
def test_empty_potential(raw_data, selection):
    raw_potential = raw_data.potential("Sr2TiO4 total")
    raw_potential.total_potential = raw.VaspData(None)
    potential = Potential.from_data(raw_potential)
    with pytest.raises(exception.NoData):
        potential.plot(selection)


def test_to_contour(reference_potential, Assert):
    reference = reference_potential.ref
    expected_data = reference.output["total"][:, :, 13]
    expected_lattice_vectors = reference.structure.lattice_vectors()[:2, :2]
    graph = reference_potential.to_contour(c=0.9)
    assert len(graph) == 1
    contour = graph.series[0]
    Assert.allclose(contour.data, expected_data)
    Assert.allclose(contour.lattice.vectors, expected_lattice_vectors)
    assert contour.label == "total potential"


@pytest.mark.parametrize(
    "selection", ("sigma_z", "ionic(0)", "xc(sigma_1)", "y(total)")
)
def test_to_contour_selections(noncollinear_potential, selection, Assert):
    if "0" in selection:
        expected_data = noncollinear_potential.ref.output["ionic"]
    elif "1" in selection:
        expected_data = noncollinear_potential.ref.output["xc_magnetization"][0]
    elif "y" in selection:
        expected_data = noncollinear_potential.ref.output["total_magnetization"][1]
    else:
        expected_data = noncollinear_potential.ref.output["total_magnetization"][2]
    expected_data = expected_data[4, :, :]
    graph = noncollinear_potential.to_contour(selection, a=0.4)
    assert len(graph) == 1
    contour = graph.series[0]
    Assert.allclose(contour.data, expected_data)


def test_to_quiver(noncollinear_potential, Assert):
    reference = noncollinear_potential.ref
    expected_data = reference.output["total_magnetization"][:2, :, :, 4]
    expected_lattice_vectors = reference.structure.lattice_vectors()[:2, :2]
    graph = noncollinear_potential.to_quiver(c=0.3)
    assert len(graph) == 1
    quiver = graph.series[0]
    Assert.allclose(quiver.data, expected_data)
    Assert.allclose(quiver.lattice.vectors, expected_lattice_vectors)
    assert quiver.label == "total potential"


def test_to_quiver_multiple(noncollinear_potential, Assert):
    reference = noncollinear_potential.ref
    expected_total = reference.output["total_magnetization"][:2, :, :, 10]
    expected_xc = reference.output["xc_magnetization"][:2, :, :, 10]
    graph = noncollinear_potential.to_quiver("total, xc", c=0.7)
    assert len(graph) == 2
    total_quiver, xc_quiver = graph.series
    Assert.allclose(total_quiver.data, expected_total)
    Assert.allclose(xc_quiver.data, expected_xc)
    assert xc_quiver.label == "xc potential"


def test_to_quiver_collinear(collinear_potential, Assert):
    reference = collinear_potential.ref
    expected_data = reference.output["total_up"] - reference.output["total"]
    length_a = np.linalg.norm(reference.structure.lattice_vectors()[1])
    length_b = np.linalg.norm(reference.structure.lattice_vectors()[2])
    expected_lattice_vectors = np.diag([length_a, length_b])
    graph = collinear_potential.to_quiver(a=-0.2)
    assert len(graph) == 1
    quiver = graph.series[0]
    Assert.allclose(quiver.data[0], 0)
    Assert.allclose(quiver.data[1], expected_data)
    Assert.allclose(quiver.lattice.vectors, expected_lattice_vectors)


@pytest.mark.parametrize("supercell", [3, (2, 3)])
def test_to_quiver_supercell(noncollinear_potential, supercell, Assert):
    graph = noncollinear_potential.to_quiver(b=0.3, supercell=supercell)
    assert len(graph) == 1
    assert len(graph.series[0].supercell) == 2
    Assert.allclose(graph.series[0].supercell, supercell)


@pytest.mark.parametrize("normal", ("x", "y", "z", "auto"))
def test_to_quiver_normal(collinear_potential, normal, Assert):
    lattice_vectors = collinear_potential.ref.structure.lattice_vectors()
    plane = slicing.plane(lattice_vectors, "a", normal)
    graph = collinear_potential.to_quiver(a=0.5, normal=normal)
    assert len(graph) == 1
    Assert.allclose(graph.series[0].lattice, plane)


def test_to_quiver_fails_for_nonpolarized(nonpolarized_potential):
    with pytest.raises(exception.DataMismatch):
        nonpolarized_potential.to_quiver(c=0)


def test_to_quiver_fails_for_ionic(collinear_potential):
    with pytest.raises(exception.IncorrectUsage):
        collinear_potential.to_quiver("ionic", c=0)


def test_print(reference_potential, format_):
    actual, _ = format_(reference_potential)
    assert actual == {"text/plain": reference_potential.ref.string}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.potential("Fe3O4 collinear total")
    check_factory_methods(Potential, data)

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import re
import types
from dataclasses import dataclass

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._calculation.effective_coulomb import EffectiveCoulomb
from py4vasp._third_party import numeric
from py4vasp._util import check, convert


@dataclass
class Setup:
    has_frequencies: bool
    has_positions: bool
    is_nonpolarized: bool
    is_collinear: bool
    number_wannier: int


@pytest.fixture(params=["crpa", "crpa_two_center", "crpar", "crpar_two_center"])
def effective_coulomb(raw_data, request):
    return create_coulomb_reference(raw_data, request.param)


@pytest.fixture
def collinear_crpa(raw_data):
    return create_coulomb_reference(raw_data, "crpa_two_center")


@pytest.fixture
def nonpolarized_crpar(raw_data):
    return create_coulomb_reference(raw_data, "crpar_two_center")


@pytest.fixture
def collinear_crpar(raw_data):
    return create_coulomb_reference(raw_data, "crpar")


def create_coulomb_reference(raw_data, param):
    raw_coulomb = raw_data.effective_coulomb(param)
    coulomb = EffectiveCoulomb.from_data(raw_coulomb)
    coulomb.ref = types.SimpleNamespace()
    setup = determine_setup(raw_coulomb)
    coulomb.ref.read_data = setup_read_data(setup, raw_coulomb)
    coulomb.ref.omega_data = setup_omega_data(setup, coulomb.ref.read_data)
    coulomb.ref.radial_data = setup_radial_data(setup, coulomb.ref.read_data)
    coulomb.ref.overview_data = setup_overview_data(setup, coulomb.ref.read_data)
    return coulomb


def determine_setup(raw_coulomb):
    return Setup(
        has_frequencies=len(raw_coulomb.frequencies) > 1,
        has_positions=not check.is_none(raw_coulomb.positions),
        is_nonpolarized=(len(raw_coulomb.bare_potential_low_cutoff) == 1),
        is_collinear=(len(raw_coulomb.bare_potential_low_cutoff) == 3),
        number_wannier=raw_coulomb.number_wannier_states,
    )


def setup_read_data(setup, raw_coulomb):
    axis_U = 2 if setup.has_frequencies else 1
    C = unpack(setup.number_wannier, raw_coulomb.bare_potential_low_cutoff, axis=1)
    V = unpack(setup.number_wannier, raw_coulomb.bare_potential_high_cutoff, axis=1)
    U = unpack(setup.number_wannier, raw_coulomb.screened_potential, axis=axis_U)
    if setup.has_positions:
        V = np.moveaxis(V, -1, 0)
        U = np.moveaxis(U, -1, 0)
    if setup.has_frequencies:
        U = np.moveaxis(U, 1 if setup.has_positions else 0, -1)
        V = V[..., np.newaxis]
        C = C[..., np.newaxis]
    result = {
        "bare_high_cutoff": V,
        "bare_low_cutoff": C,
        "screened": U,
    }
    if setup.has_frequencies:
        result["frequencies"] = convert.to_complex(raw_coulomb.frequencies[:])
    if not check.is_none(raw_coulomb.positions):
        if check.is_none(raw_coulomb.cell.scale):
            lattice_vectors = raw_coulomb.cell.lattice_vectors
        else:
            lattice_vectors = raw_coulomb.cell.scale * raw_coulomb.cell.lattice_vectors
        result["lattice_vectors"] = lattice_vectors
        result["positions"] = raw_coulomb.positions
    return result


def unpack(num_wannier, data, axis):
    data = convert.to_complex(data[:])
    shape = (
        data.shape[:axis]
        + (num_wannier, num_wannier, num_wannier, num_wannier)
        + data.shape[axis + 1 :]
    )
    return data.reshape(shape)


def setup_omega_data(setup, read_data):
    if not setup.has_frequencies:
        return None
    weight = (1 if setup.is_nonpolarized else 0.5) / setup.number_wannier
    weight_j = 1 / (setup.number_wannier - 1)
    if setup.has_positions:
        screened_potential = read_data["screened"][0]
        bare_potential = read_data["bare_high_cutoff"][0]
        U_for_both = weight * np.einsum(f"rsiiiiw->rw", read_data["screened"])
    else:
        screened_potential = read_data["screened"]
        bare_potential = read_data["bare_high_cutoff"]
        U_for_both = None
    U = weight * np.einsum(f"siiiiw->sw", screened_potential)
    u = weight_j * (weight * np.einsum(f"sijjiw->sw", screened_potential) - U)
    J = weight_j * (weight * np.einsum(f"sijijw->sw", screened_potential) - U)
    V = weight * np.einsum(f"siiiiw->sw", bare_potential)
    v = weight_j * (weight * np.einsum(f"sijjiw->sw", bare_potential) - V)
    Vj = weight_j * (weight * np.einsum(f"sijijw->sw", bare_potential) - V)
    return {
        "frequencies": read_data.get("frequencies"),
        "screened U": U,
        "screened u": u,
        "screened J": J,
        "screened V": U,
        "screened v": u,
        "bare V": V,
        "bare v": v,
        "bare J": Vj,
        "bare U": V,
        "bare u": v,
        "U for both": U_for_both,
    }


def setup_radial_data(setup, read_data):
    if not setup.has_positions:
        return None
    weight = (1 if setup.is_nonpolarized else 0.5) / setup.number_wannier
    weight_j = 1 / (setup.number_wannier - 1)
    if setup.has_frequencies:
        screened_potential = read_data["screened"][..., 0]
        bare_potential = read_data["bare_high_cutoff"][..., 0]
    else:
        screened_potential = read_data["screened"]
        bare_potential = read_data["bare_high_cutoff"]
    positions = read_data["lattice_vectors"] @ read_data["positions"][:].T
    U = weight * np.einsum("rsiiii->sr", screened_potential.real)
    J = weight_j * (weight * np.einsum("rsijij->sr", screened_potential.real) - U)
    return {
        "radius": np.linalg.norm(positions, axis=0),
        "screened U": U,
        "screened J": J,
        "bare V": np.einsum("rsiiii->sr", bare_potential.real) * weight,
        "label for both": [
            f"screened U @ {position}" for position in read_data["positions"]
        ],
    }


def setup_overview_data(setup, read_data):
    weight = (1 if setup.is_nonpolarized else 0.5) / setup.number_wannier
    weight_j = 1 / (setup.number_wannier - 1)
    U_ijkl = read_data["screened"]
    V_ijkl = read_data["bare_high_cutoff"]
    if setup.has_frequencies:  # only omega = 0
        U_ijkl = U_ijkl[..., 0]
        V_ijkl = V_ijkl[..., 0]
    if setup.has_positions:  # only r = 0
        U_ijkl = U_ijkl[0]
        V_ijkl = V_ijkl[0]
    # make spin diagonal
    U_ijkl = U_ijkl[:2]
    V_ijkl = V_ijkl[:2]
    U = weight * np.einsum("siiii->", U_ijkl)
    u = weight_j * (weight * np.einsum("sijji->", U_ijkl) - U)
    J = weight_j * (weight * np.einsum("sijij->", U_ijkl) - U)
    V = weight * np.einsum("siiii->", V_ijkl)
    v = weight_j * (weight * np.einsum("sijji->", V_ijkl) - V)
    Vj = weight_j * (weight * np.einsum("sijij->", V_ijkl) - V)
    return {
        "screened_U": U,
        "screened_u": u,
        "screened_J": J,
        "bare_V": V,
        "bare_v": v,
        "bare_J": Vj,
    }


def test_read(effective_coulomb, Assert):
    actual = effective_coulomb.read()
    assert actual.keys() == effective_coulomb.ref.read_data.keys()
    for key in actual.keys():
        assert actual[key].shape == effective_coulomb.ref.read_data[key].shape
        Assert.allclose(actual[key], effective_coulomb.ref.read_data[key])


def test_plot(effective_coulomb, Assert):
    if effective_coulomb.ref.omega_data is not None:
        graph = effective_coulomb.plot(omega=...)
        omega_data = effective_coulomb.ref.omega_data
        expected_labels = ["screened U", "screened J", "bare V"]
        check_plot_has_correct_series(graph, omega_data, expected_labels, Assert)
    else:
        check_plot_raises_error(effective_coulomb)


@pytest.mark.parametrize("selection", ["U u J", "screened(U, u, J)", "bare(V, v, J)"])
def test_plot_component_selection(collinear_crpar, selection, Assert):
    graph = collinear_crpar.plot(selection)
    omega_data = collinear_crpar.ref.omega_data
    if "bare" in selection:
        expected_labels = ["bare V", "bare v", "bare J"]
    else:
        expected_labels = ["screened U", "screened u", "screened J"]
    check_plot_has_correct_series(graph, omega_data, expected_labels, Assert)


@pytest.mark.parametrize(
    "selection, expected_labels",
    [
        ("screened(V, v)", ["screened V", "screened v"]),
        ("bare(U, u)", ["bare U", "bare u"]),
    ],
)
def test_plot_with_unusual_labels(collinear_crpar, selection, expected_labels, Assert):
    graph = collinear_crpar.plot(selection)
    omega_data = collinear_crpar.ref.omega_data
    check_plot_has_correct_series(graph, omega_data, expected_labels, Assert)


def check_plot_has_correct_series(graph, omega_data, expected_labels, Assert):
    assert len(graph) == len(expected_labels)
    assert graph.xlabel == "Im(ω) (eV)"
    assert graph.ylabel == "Coulomb potential (eV)"
    for series, label in zip(graph, expected_labels):
        Assert.allclose(series.x, omega_data["frequencies"].imag)
        potential = omega_data[label]
        Assert.allclose(series.y, np.sum(potential[:2], axis=0).real)
        assert series.label == label


def check_plot_raises_error(effective_coulomb):
    with pytest.raises(exception.DataMismatch):
        effective_coulomb.plot(omega=...)


@pytest.mark.parametrize("selection", ["total", "up~up", "down~down", "up~down"])
def test_plot_selected_spin(collinear_crpar, selection, Assert):
    effective_coulomb = collinear_crpar
    if selection == "total":
        spin_selection = slice(None, 2)
        factor = 1.0
    else:
        spin_map = {"up~up": 0, "down~down": 1, "up~down": 2}
        spin_selection = slice(spin_map[selection], spin_map[selection] + 1)
        factor = 2.0

    # not specifying omega or radius defaults to frequency plot
    graph = effective_coulomb.plot(selection)
    assert len(graph) == 1
    potential = factor * effective_coulomb.ref.omega_data["screened U"]
    Assert.allclose(graph[0].y, np.sum(potential[spin_selection], axis=0).real)
    assert graph[0].label == f"screened {selection}"


@pytest.mark.parametrize(
    "selection", ["up~up", "down~down", "up~down", "invalid_selection"]
)
def test_plot_invalid_selection(nonpolarized_crpar, selection):
    # must not use magnetism-specific selections for nonpolarized data
    effective_coulomb = nonpolarized_crpar
    with pytest.raises(exception.IncorrectUsage):
        effective_coulomb.plot(selection)


def test_plot_with_analytic_continuation(nonpolarized_crpar, not_core, Assert):
    omega_data = nonpolarized_crpar.ref.omega_data
    omega = np.linspace(0, 10, 20)
    expected_U = numeric.analytic_continuation(
        omega_data["frequencies"], omega_data["screened U"], omega
    )
    graph = nonpolarized_crpar.plot("U", omega=omega)
    assert len(graph) == 1
    assert graph.xlabel == "ω (eV)"
    assert graph.ylabel == "Coulomb potential (eV)"
    series = graph[0]
    Assert.allclose(series.x, omega)
    Assert.allclose(series.y, expected_U[0].real, tolerance=100)
    assert series.label == "screened U"


def test_plot_with_analytic_continuation_and_spin_selection(
    collinear_crpar, not_core, Assert
):
    omega_data = collinear_crpar.ref.omega_data
    omega = np.linspace(0, 10, 20)
    expected_output = 2 * numeric.analytic_continuation(
        omega_data["frequencies"], omega_data["screened U"], omega
    )
    graph = collinear_crpar.plot("down~down", omega=omega)
    assert len(graph) == 1
    series = graph[0]
    Assert.allclose(series.x, omega)
    Assert.allclose(series.y, expected_output[1].real, tolerance=100)
    assert series.label == "screened down~down"


def test_plot_radial_potential(effective_coulomb, Assert):
    if effective_coulomb.ref.radial_data is not None:
        check_radial_plot_has_correct_series(effective_coulomb, Assert)
    else:
        check_radial_plot_raises_error(effective_coulomb)


def check_radial_plot_has_correct_series(effective_coulomb, Assert):
    graph = effective_coulomb.plot(radius=...)
    assert len(graph) == 3
    assert graph.xlabel == "Radius (Å)"
    assert graph.ylabel == "Coulomb potential (eV)"
    expected_labels = ["screened U", "screened J", "bare V"]
    for series, label in zip(graph, expected_labels):
        Assert.allclose(series.x, effective_coulomb.ref.radial_data["radius"])
        potential = effective_coulomb.ref.radial_data[label]
        Assert.allclose(series.y, np.sum(potential[:2], axis=0))
        assert series.label == label
        assert series.marker == "*"


def check_radial_plot_raises_error(effective_coulomb):
    with pytest.raises(exception.DataMismatch):
        effective_coulomb.plot(radius=...)


@pytest.mark.parametrize("selection", ["total", "up~up", "down~down", "up~down"])
def test_plot_radial_selected_spin(collinear_crpa, selection, Assert):
    effective_coulomb = collinear_crpa
    if selection == "total":
        spin_selection = slice(None, 2)
        factor = 1.0
    else:
        spin_map = {"up~up": 0, "down~down": 1, "up~down": 2}
        spin_selection = slice(spin_map[selection], spin_map[selection] + 1)
        factor = 2.0
    graph = effective_coulomb.plot(selection, radius=...)
    assert len(graph) == 1
    series = graph[0]
    potential = factor * effective_coulomb.ref.radial_data["screened U"]
    Assert.allclose(series.y, np.sum(potential[spin_selection], axis=0))
    assert series.label == f"screened {selection}"


@pytest.mark.parametrize(
    "selection", ["up~up", "down~down", "up~down", "invalid_selection"]
)
def test_plot_radial_invalid_selection(nonpolarized_crpar, selection):
    # must not use magnetism-specific selections for nonpolarized data
    effective_coulomb = nonpolarized_crpar
    with pytest.raises(exception.IncorrectUsage):
        effective_coulomb.plot(selection, radius=...)


def test_plot_radial_interpolation(nonpolarized_crpar, not_core, Assert):
    radial_data = nonpolarized_crpar.ref.radial_data
    radius = np.linspace(0.0, np.max(radial_data["radius"]), 30)
    ref_graph = nonpolarized_crpar.plot(radius=...)
    graph = nonpolarized_crpar.plot(radius=radius)
    assert len(graph) == len(ref_graph)
    assert graph.xlabel == "Radius (Å)"
    assert graph.ylabel == "Coulomb potential (eV)"
    for series, ref_series in zip(graph, ref_graph):
        Assert.allclose(series.x, radius)
        expected_interpolated_values = interpolate(radial_data, ref_series.y, radius)
        Assert.allclose(series.y, expected_interpolated_values, tolerance=100)
        assert series.label == ref_series.label
        assert series.marker is None


def test_plot_radial_interpolation_spin_selection(collinear_crpa, not_core, Assert):
    effective_coulomb = collinear_crpa
    radial_data = effective_coulomb.ref.radial_data
    radius = np.linspace(0, 10)
    ref_graph = effective_coulomb.plot("up~down(U V)", radius=...)
    graph = effective_coulomb.plot("up~down(U V)", radius=radius)
    expected_labels = ["screened up~down_U", "bare up~down_V"]
    assert len(graph) == len(ref_graph)
    for series, ref_series, label in zip(graph, ref_graph, expected_labels):
        Assert.allclose(series.x, radius)
        Assert.allclose(series.y, interpolate(radial_data, ref_series.y, radius))
        assert series.label == label


def interpolate(radial_data, potential, radius):
    U0 = potential[0]
    interpolation = numeric.interpolate_with_function(
        EffectiveCoulomb.ohno_potential, radial_data["radius"], potential / U0, radius
    )
    return U0 * interpolation


def test_plot_radial_and_frequency(effective_coulomb, Assert):
    omega_data = effective_coulomb.ref.omega_data
    radial_data = effective_coulomb.ref.radial_data
    if radial_data is None or omega_data is None:
        with pytest.raises(exception.DataMismatch):
            effective_coulomb.plot("U", omega=..., radius=...)
        return
    graph = effective_coulomb.plot("U", omega=..., radius=...)
    assert len(graph) == len(radial_data["radius"])
    assert graph.xlabel == "Im(ω) (eV)"
    assert graph.ylabel == "Coulomb potential (eV)"
    expected_labels = radial_data["label for both"]
    expected_lines = omega_data["U for both"].real
    for series, expected_line, label in zip(graph, expected_lines, expected_labels):
        Assert.allclose(series.x, omega_data["frequencies"].imag)
        Assert.allclose(series.y, expected_line)
        assert series.label == label


def test_plot_radial_and_frequency_nondefault_radius(nonpolarized_crpar, Assert):
    with pytest.raises(exception.NotImplemented):
        nonpolarized_crpar.plot(omega=..., radius=np.array([1.0, 2.0]))


@pytest.mark.xfail
def test_selections(effective_coulomb):
    # TODO implement
    assert False


def test_to_database(effective_coulomb, Assert):
    data = effective_coulomb._to_database()["effective_coulomb"]
    assert data.keys() == effective_coulomb.ref.overview_data.keys()
    for key in data.keys():
        Assert.allclose(data[key], effective_coulomb.ref.overview_data[key])


def test_print(effective_coulomb, format_):
    expected_result = r"""averaged bare interaction
bare Hubbard U =\s+[-\d.]+\s+[-\d.]+
bare Hubbard u =\s+[-\d.]+\s+[-\d.]+
bare Hubbard J =\s+[-\d.]+\s+[-\d.]+

averaged interaction parameter
screened Hubbard U =\s+[-\d.]+\s+[-\d.]+
screened Hubbard u =\s+[-\d.]+\s+[-\d.]+
screened Hubbard J =\s+[-\d.]+\s+[-\d.]+"""
    actual, _ = format_(effective_coulomb)
    assert actual.keys() == {"text/plain"}
    assert re.search(expected_result, actual["text/plain"], re.MULTILINE)


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.effective_coulomb("crpa")
    check_factory_methods(EffectiveCoulomb, data)

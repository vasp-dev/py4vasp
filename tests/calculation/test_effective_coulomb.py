# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types
from dataclasses import dataclass

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._calculation.effective_coulomb import EffectiveCoulomb
from py4vasp._third_party.numeric import analytic_continuation
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
    if setup.is_nonpolarized:
        weight = 1 / setup.number_wannier
    else:
        weight = 0.5 / setup.number_wannier
    if setup.has_positions:
        screened_potential = read_data["screened"][0]
        bare_potential = read_data["bare_high_cutoff"][0]
    else:
        screened_potential = read_data["screened"]
        bare_potential = read_data["bare_high_cutoff"]
    return {
        "frequencies": read_data.get("frequencies"),
        "screened": np.einsum(f"siiiiw->sw", screened_potential) * weight,
        "bare": np.einsum(f"siiiiw->sw", bare_potential) * weight,
    }


def setup_radial_data(setup, read_data):
    if not setup.has_positions:
        return None
    if setup.is_nonpolarized:
        weight = 1 / setup.number_wannier
    else:
        weight = 0.5 / setup.number_wannier
    if setup.has_frequencies:
        screened_potential = read_data["screened"][..., 0]
        bare_potential = read_data["bare_high_cutoff"][..., 0]
    else:
        screened_potential = read_data["screened"]
        bare_potential = read_data["bare_high_cutoff"]
    positions = read_data["lattice_vectors"] @ read_data["positions"][:].T
    return {
        "radius": np.linalg.norm(positions, axis=0),
        "screened": np.einsum("rsiiii->sr", screened_potential) * weight,
        "bare": np.einsum("rsiiii->sr", bare_potential) * weight,
    }


def test_read(effective_coulomb, Assert):
    actual = effective_coulomb.read()
    assert actual.keys() == effective_coulomb.ref.read_data.keys()
    for key in actual.keys():
        assert actual[key].shape == effective_coulomb.ref.read_data[key].shape
        Assert.allclose(actual[key], effective_coulomb.ref.read_data[key])


def test_plot(effective_coulomb, Assert):
    if effective_coulomb.ref.omega_data is not None:
        check_plot_has_correct_series(effective_coulomb, Assert)
    else:
        check_plot_raises_error(effective_coulomb)


def check_plot_has_correct_series(effective_coulomb, Assert):
    graph = effective_coulomb.plot()
    assert len(graph) == 2
    assert graph.xlabel == "Im(ω) (eV)"
    assert graph.ylabel == "Coulomb potential (eV)"
    expected_labels = ["screened", "bare"]
    for series, label in zip(graph, expected_labels):
        Assert.allclose(series.x, effective_coulomb.ref.omega_data["frequencies"].imag)
        potential = effective_coulomb.ref.omega_data[label]
        Assert.allclose(series.y, np.sum(potential[:2], axis=0).real)
        assert series.label == label


def check_plot_raises_error(effective_coulomb):
    with pytest.raises(exception.DataMismatch):
        effective_coulomb.plot()


@pytest.mark.parametrize("selection", ["total", "up~up", "down~down", "up~down"])
def test_plot_selected_spin(collinear_crpar, selection, Assert):
    effective_coulomb = collinear_crpar
    if selection == "total":
        spin_selection = slice(None, 2)
        suffix = ""
        factor = 1.0
    else:
        spin_map = {"up~up": 0, "down~down": 1, "up~down": 2}
        spin_selection = slice(spin_map[selection], spin_map[selection] + 1)
        suffix = f"_{selection}"
        factor = 2.0

    graph = effective_coulomb.plot(selection)
    assert len(graph) == 2
    expected_labels = ["screened", "bare"]
    for series, label in zip(graph, expected_labels):
        potential = factor * effective_coulomb.ref.omega_data[label]
        Assert.allclose(series.y, np.sum(potential[spin_selection], axis=0).real)
        assert series.label == f"{label}{suffix}"


@pytest.mark.parametrize(
    "selection", ["up~up", "down~down", "up~down", "invalid_selection"]
)
def test_plot_invalid_selection(nonpolarized_crpar, selection):
    # must not use magnetism-specific selections for nonpolarized data
    effective_coulomb = nonpolarized_crpar
    with pytest.raises(exception.IncorrectUsage):
        effective_coulomb.plot(selection)


def test_plot_with_analytic_continuation(nonpolarized_crpar, Assert):
    omega_data = nonpolarized_crpar.ref.omega_data
    omega = np.linspace(0, 10, 20)
    expected_output = analytic_continuation(
        omega_data["frequencies"], omega_data["screened"], omega
    )
    graph = nonpolarized_crpar.plot(omega=omega)
    assert len(graph) == 2
    assert graph.xlabel == "ω (eV)"
    assert graph.ylabel == "Coulomb potential (eV)"
    expected_lines = [np.squeeze(expected_output), omega_data["bare"]]
    expected_labels = ["screened", "bare"]
    for series, expected_line, label in zip(graph, expected_lines, expected_labels):
        Assert.allclose(series.x, omega)
        Assert.allclose(series.y, expected_line.real, tolerance=100)
        assert series.label == label


def test_plot_with_analytic_continuation_and_spin_selection(collinear_crpar, Assert):
    omega_data = collinear_crpar.ref.omega_data
    omega = np.linspace(0, 10, 20)
    expected_output = 2 * analytic_continuation(
        omega_data["frequencies"], omega_data["screened"], omega
    )
    graph = collinear_crpar.plot("down~down", omega=omega)
    assert len(graph) == 2
    series = graph[0]
    Assert.allclose(series.x, omega)
    Assert.allclose(series.y, expected_output[1].real, tolerance=100)
    assert series.label == "screened_down~down"


def test_plot_radial_potential(effective_coulomb, Assert):
    if effective_coulomb.ref.radial_data is not None:
        check_radial_plot_has_correct_series(effective_coulomb, Assert)
    else:
        check_radial_plot_raises_error(effective_coulomb)


def check_radial_plot_has_correct_series(effective_coulomb, Assert):
    graph = effective_coulomb.plot(omega=0)
    assert len(graph) == 2
    assert graph.xlabel == "Distance (Å)"
    assert graph.ylabel == "Coulomb potential (eV)"
    expected_labels = ["screened", "bare"]
    for series, label in zip(graph, expected_labels):
        Assert.allclose(series.x, effective_coulomb.ref.radial_data["radius"])
        potential = effective_coulomb.ref.radial_data[label].real
        Assert.allclose(series.y, np.sum(potential[:2], axis=0))
        assert series.label == label
        assert series.marker == "*"


def check_radial_plot_raises_error(effective_coulomb):
    with pytest.raises(exception.DataMismatch):
        effective_coulomb.plot(omega=0)


@pytest.mark.parametrize("selection", ["total", "up~up", "down~down", "up~down"])
def test_plot_radial_selected_spin(collinear_crpa, selection, Assert):
    effective_coulomb = collinear_crpa
    if selection == "total":
        spin_selection = slice(None, 2)
        suffix = ""
        factor = 1.0
    else:
        spin_map = {"up~up": 0, "down~down": 1, "up~down": 2}
        spin_selection = slice(spin_map[selection], spin_map[selection] + 1)
        suffix = f"_{selection}"
        factor = 2.0

    graph = effective_coulomb.plot(selection, omega=0)
    assert len(graph) == 2
    expected_labels = ["screened", "bare"]
    for series, label in zip(graph, expected_labels):
        potential = factor * effective_coulomb.ref.radial_data[label]
        Assert.allclose(series.y, np.sum(potential[spin_selection], axis=0).real)
        assert series.label == f"{label}{suffix}"


@pytest.mark.parametrize(
    "selection", ["up~up", "down~down", "up~down", "invalid_selection"]
)
def test_plot_invalid_selection(nonpolarized_crpar, selection):
    # must not use magnetism-specific selections for nonpolarized data
    effective_coulomb = nonpolarized_crpar
    with pytest.raises(exception.IncorrectUsage):
        effective_coulomb.plot(selection, omega=0)

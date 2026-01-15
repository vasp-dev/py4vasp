# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._calculation.effective_coulomb import EffectiveCoulomb
from py4vasp._util import check, convert


@dataclass
class Setup:
    has_frequencies: bool
    has_positions: bool
    is_nonpolarized: bool
    is_collinear: bool


@pytest.fixture(params=["crpa", "crpa_two_center", "crpar", "crpar_two_center"])
def effective_coulomb(raw_data, request):
    return create_coulomb_reference(raw_data, request.param)


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
    coulomb.ref.setup = determine_setup(param, raw_coulomb)
    coulomb.ref.num_wannier = raw_coulomb.number_wannier_states
    coulomb.ref.expected = setup_expected_dict(coulomb.ref.setup, raw_coulomb)
    return coulomb


def determine_setup(param, raw_coulomb):
    return Setup(
        has_frequencies=len(raw_coulomb.frequencies) > 1,
        has_positions=not check.is_none(raw_coulomb.positions),
        is_nonpolarized=(len(raw_coulomb.bare_potential_low_cutoff) == 1),
        is_collinear=(len(raw_coulomb.bare_potential_low_cutoff) == 3),
    )


def setup_expected_dict(setup, raw_coulomb):
    num_wannier = raw_coulomb.number_wannier_states
    axis_U = 2 if setup.has_frequencies else 1
    C = unpack(num_wannier, raw_coulomb.bare_potential_low_cutoff, axis=1)
    V = unpack(num_wannier, raw_coulomb.bare_potential_high_cutoff, axis=1)
    U = unpack(num_wannier, raw_coulomb.screened_potential, axis=axis_U)
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


def test_read(effective_coulomb, Assert):
    actual = effective_coulomb.read()
    assert actual.keys() == effective_coulomb.ref.expected.keys()
    for key in actual.keys():
        assert actual[key].shape == effective_coulomb.ref.expected[key].shape
        Assert.allclose(actual[key], effective_coulomb.ref.expected[key])


def test_plot(effective_coulomb, Assert):
    if effective_coulomb.ref.setup.has_frequencies:
        check_plot_has_correct_series(effective_coulomb, Assert)
    else:
        check_plot_raises_error(effective_coulomb)


def check_plot_has_correct_series(effective_coulomb, Assert):
    graph = effective_coulomb.plot()
    assert len(graph) == 2
    assert graph.xlabel == "Im(ω) (eV)"
    assert graph.ylabel == "Coulomb potential (eV)"
    frequencies = effective_coulomb.ref.expected.get("frequencies")
    if effective_coulomb.ref.setup.has_positions:
        screened_potential = effective_coulomb.ref.expected["screened"][0]
        bare_potential = effective_coulomb.ref.expected["bare_high_cutoff"][0]
    else:
        screened_potential = effective_coulomb.ref.expected["screened"]
        bare_potential = effective_coulomb.ref.expected["bare_high_cutoff"]
    if effective_coulomb.ref.setup.is_nonpolarized:
        weight = 1 / effective_coulomb.ref.num_wannier
    else:
        weight = 0.5 / effective_coulomb.ref.num_wannier
    expected_lines = (
        np.einsum(f"siiiiw->w", screened_potential[:2].real) * weight,
        np.einsum(f"siiiiw->w", bare_potential[:2].real) * weight,
    )
    expected_labels = ["screened", "bare"]
    for series, expected_line, label in zip(graph, expected_lines, expected_labels):
        Assert.allclose(series.x, frequencies.imag)
        Assert.allclose(series.y, expected_line)
        assert series.label == label


def check_plot_raises_error(effective_coulomb):
    with pytest.raises(exception.DataMismatch):
        effective_coulomb.plot()


@pytest.mark.parametrize("selection", ["total", "up~up", "down~down", "up~down"])
def test_plot_selected_spin(collinear_crpar, selection, Assert):
    effective_coulomb = collinear_crpar
    graph = effective_coulomb.plot(selection)
    assert len(graph) == 2
    screened_potential = effective_coulomb.ref.expected["screened"]
    bare_potential = effective_coulomb.ref.expected["bare_high_cutoff"]
    if selection == "total":
        weight = 0.5 / effective_coulomb.ref.num_wannier
        spin_selection = slice(None, 2)
        suffix = ""
    else:
        weight = 1 / effective_coulomb.ref.num_wannier
        spin_map = {"up~up": 0, "down~down": 1, "up~down": 2}
        spin_selection = slice(spin_map[selection], spin_map[selection] + 1)
        suffix = f"_{selection}"
    expected_lines = (
        np.einsum(f"siiiiw->w", screened_potential[spin_selection].real) * weight,
        np.einsum(f"siiiiw->w", bare_potential[spin_selection].real) * weight,
    )
    expected_labels = [f"screened{suffix}", f"bare{suffix}"]
    for series, expected_line, label in zip(graph, expected_lines, expected_labels):
        Assert.allclose(series.y, expected_line)
        assert series.label == label


@pytest.mark.parametrize(
    "selection", ["up~up", "down~down", "up~down", "invalid_selection"]
)
def test_plot_invalid_selection(nonpolarized_crpar, selection):
    # must not use magnetism-specific selections for nonpolarized data
    effective_coulomb = nonpolarized_crpar
    with pytest.raises(exception.IncorrectUsage):
        effective_coulomb.plot(selection)


def test_plot_with_analytic_continuation(nonpolarized_crpar, Assert):
    effective_coulomb = nonpolarized_crpar
    frequencies = effective_coulomb.ref.expected["frequencies"]
    weight = 1 / effective_coulomb.ref.num_wannier
    screened_potential = (
        np.einsum(f"siiiiw->w", effective_coulomb.ref.expected["screened"][0, :2])
        * weight
    )
    bare_potential = (
        np.einsum(
            f"siiiiw->w", effective_coulomb.ref.expected["bare_high_cutoff"][0, :2].real
        )
        * weight
    )

    omega = np.linspace(0, 10, 50)
    analytic_continuation = "py4vasp._third_party.numeric.analytic_continuation"
    mock_data = np.random.rand(len(omega), 1)
    with patch(analytic_continuation, return_value=mock_data) as mock_analytic:
        graph = effective_coulomb.plot(omega=omega)
        mock_analytic.assert_called_once()
        z_in, f_in, z_out = mock_analytic.call_args.args
        Assert.allclose(z_in, frequencies)
        Assert.allclose(f_in, screened_potential)
        Assert.allclose(z_out, omega)

    assert len(graph) == 2
    assert graph.xlabel == "ω (eV)"
    assert graph.ylabel == "Coulomb potential (eV)"
    expected_lines = [mock_data, bare_potential]
    expected_labels = ["screened", "bare"]
    for series, expected_line, label in zip(graph, expected_lines, expected_labels):
        Assert.allclose(series.x, omega)
        Assert.allclose(series.y, expected_line)
        assert series.label == label
    assert False

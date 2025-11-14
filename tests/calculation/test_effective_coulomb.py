# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types
from dataclasses import dataclass

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
    raw_coulomb = raw_data.effective_coulomb(request.param)
    coulomb = EffectiveCoulomb.from_data(raw_coulomb)
    coulomb.ref = types.SimpleNamespace()
    coulomb.ref.setup = determine_setup(request.param, raw_coulomb)
    coulomb.ref.num_wannier = raw_coulomb.number_wannier_states
    coulomb.ref.expected = setup_expected_dict(coulomb.ref.setup, raw_coulomb)
    return coulomb


def determine_setup(param, raw_coulomb):
    return Setup(
        has_frequencies=len(raw_coulomb.frequencies) > 1,
        has_positions=not check.is_none(raw_coulomb.positions),
        is_nonpolarized=(len(raw_coulomb.bare_potential_low_cutoff) == 1),
        is_collinear=(len(raw_coulomb.bare_potential_low_cutoff) == 2),
    )


def setup_expected_dict(setup, raw_coulomb):
    num_wannier = raw_coulomb.number_wannier_states
    axis_U = 3 if setup.has_frequencies else 2
    C = unpack(num_wannier, raw_coulomb.bare_potential_low_cutoff, axis=2)
    V = unpack(num_wannier, raw_coulomb.bare_potential_high_cutoff, axis=2)
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
    num_wannier = effective_coulomb.ref.num_wannier
    expected_lines = (
        np.einsum(f"ssiiiiw->w", screened_potential.real) / num_wannier,
        np.einsum(f"ssiiiiw->w", bare_potential.real) / num_wannier,
    )
    expected_labels = ["screened", "bare"]
    for series, expected_line, label in zip(graph, expected_lines, expected_labels):
        Assert.allclose(series.x, frequencies.imag)
        Assert.allclose(series.y, expected_line)
        assert series.label == label
        assert series.label == label


def check_plot_raises_error(effective_coulomb):
    with pytest.raises(exception.DataMismatch):
        effective_coulomb.plot()

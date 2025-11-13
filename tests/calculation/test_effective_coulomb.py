# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._calculation.effective_coulomb import EffectiveCoulomb
from py4vasp._util import check, convert


@pytest.fixture(params=["crpa", "crpa_two_center", "crpar", "crpar_two_center"])
def effective_coulomb(raw_data, request):
    raw_coulomb = raw_data.effective_coulomb(request.param)
    coulomb = EffectiveCoulomb.from_data(raw_coulomb)
    coulomb.ref = types.SimpleNamespace()
    coulomb.ref.num_wannier = raw_coulomb.number_wannier_states
    coulomb.ref.expected = setup_expected_dict(request.param, raw_coulomb)
    return coulomb


def setup_expected_dict(param, raw_coulomb):
    has_frequency = "crpar" in param
    two_center = "two_center" in param
    num_wannier = raw_coulomb.number_wannier_states
    C = unpack(num_wannier, raw_coulomb.bare_potential_low_cutoff, axis=2)
    V = unpack(num_wannier, raw_coulomb.bare_potential_high_cutoff, axis=2)
    U = unpack(
        num_wannier, raw_coulomb.screened_potential, axis=3 if has_frequency else 2
    )
    if two_center:
        V = np.moveaxis(V, -1, 0)
        U = np.moveaxis(U, -1, 0)
    if has_frequency:
        U = np.moveaxis(U, 1 if two_center else 0, -1)
        V = V[..., np.newaxis]
        C = C[..., np.newaxis]
    result = {
        "bare_high_cutoff": V,
        "bare_low_cutoff": C,
        "screened": U,
    }
    if has_frequency:
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
    frequencies = effective_coulomb.ref.expected.get("frequencies")
    if frequencies is None:
        with pytest.raises(exception.DataMismatch):
            effective_coulomb.plot()
        return
    graph = effective_coulomb.plot()
    if "positions" in effective_coulomb.ref.expected:
        screened_potential = effective_coulomb.ref.expected["screened"][0, 0, 0]
        bare_potential = effective_coulomb.ref.expected["bare_high_cutoff"][0, 0, 0]
    else:
        screened_potential = effective_coulomb.ref.expected["screened"][0, 0]
        bare_potential = effective_coulomb.ref.expected["bare_high_cutoff"][0, 0]
    num_wannier = effective_coulomb.ref.num_wannier
    expected_lines = (
        np.einsum(f"iiiiw->w", screened_potential.real) / num_wannier,
        np.einsum(f"iiiiw->w", bare_potential.real) / num_wannier,
    )
    assert len(graph) == 2
    assert graph.xlabel == "Im(ω) (eV)"
    assert graph.ylabel == "U (eV)"
    expected_labels = ["screened", "bare"]
    for series, expected_line, label in zip(graph, expected_lines, expected_labels):
        Assert.allclose(series.x, frequencies.imag)
        Assert.allclose(series.y, expected_line)
        assert series.label == label

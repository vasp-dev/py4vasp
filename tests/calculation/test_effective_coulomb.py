# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types
from math import sqrt

import numpy as np
import pytest

from py4vasp._calculation.effective_coulomb import EffectiveCoulomb
from py4vasp._util import check, convert


@pytest.fixture(params=["crpa", "crpa_two_center", "crpar", "crpar_two_center"])
def effective_coulomb(raw_data, request):
    raw_coulomb = raw_data.effective_coulomb(request.param)
    coulomb = EffectiveCoulomb.from_data(raw_coulomb)
    coulomb.ref = types.SimpleNamespace()
    coulomb.ref.expected_dict = setup_expected_dict(request.param, raw_coulomb)
    return coulomb


def setup_expected_dict(param, raw_coulomb):
    has_frequency = "crpar" in param
    two_center = "two_center" in param
    C = unpack(raw_coulomb.bare_potential_low_cutoff, axis=2)
    V = unpack(raw_coulomb.bare_potential_high_cutoff, axis=2)
    U = unpack(raw_coulomb.screened_potential, axis=3 if has_frequency else 2)
    if two_center:
        V = np.moveaxis(V, -1, 0)
        U = np.moveaxis(U, -1, 0)
    if has_frequency:
        U = np.moveaxis(U, 1 if two_center else 0, -1)
        V = V[..., np.newaxis]
        C = C[..., np.newaxis]
    result = {
        "bare high cutoff": V,
        "bare low cutoff": C,
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


def unpack(data, axis):
    data = convert.to_complex(data[:])
    num_wannier = int(sqrt(sqrt(data.shape[axis])))
    shape = (
        data.shape[:axis]
        + (num_wannier, num_wannier, num_wannier, num_wannier)
        + data.shape[axis + 1 :]
    )
    return data.reshape(shape)


def test_read(effective_coulomb, Assert):
    actual = effective_coulomb.read()
    print(actual.keys())
    print(effective_coulomb.ref.expected_dict.keys())
    assert actual.keys() == effective_coulomb.ref.expected_dict.keys()
    # for key in actual.keys():
    #     Assert(actual[key]).allclose(effective_coulomb.ref.expected_dict[key])

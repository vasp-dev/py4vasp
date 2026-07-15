# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import numpy as np
import pytest

from py4vasp._calculation.symmetry import Symmetry


@pytest.fixture(params=["CoO", "AlP"])
def symmetry(raw_data, request):
    raw_symmetry = raw_data.symmetry(request.param)
    return setup_reference(raw_symmetry, request.param)


def setup_reference(raw_symmetry, name):
    symmetry = Symmetry.from_data(raw_symmetry)
    symmetry.ref = types.SimpleNamespace()
    symmetry.ref.name = name
    symmetry.ref.raw = raw_symmetry
    return symmetry


def test_read(symmetry, Assert):
    raw = symmetry.ref.raw
    actual = symmetry.read()
    Assert.allclose(actual["rotations"], np.array(raw.rotations))
    Assert.allclose(actual["reciprocal_rotations"], np.array(raw.reciprocal_rotations))
    Assert.allclose(actual["translations"], np.array(raw.translations))
    # Fortran 1-based indices in the file are converted to 0-based for Python
    Assert.allclose(actual["inverse_operations"], np.array(raw.inverse_operations) - 1)
    Assert.allclose(actual["atom_permutations"], np.array(raw.atom_permutations) - 1)
    Assert.allclose(
        actual["primitive_lattice_vectors"], np.array(raw.primitive_lattice_vectors)
    )
    Assert.allclose(
        actual["primitive_translations"], np.array(raw.primitive_translations)
    )
    assert actual["number_of_operations"] == raw.number_of_operations
    assert actual["number_of_primitive_cells"] == raw.number_of_primitive_cells
    assert actual["isym"] == raw.isym
    if symmetry.ref.name == "CoO":
        Assert.allclose(actual["spin_flips"], np.array(raw.spin_flips))
    else:
        assert actual["spin_flips"] is None


def test_to_dict_is_alias_of_read(symmetry, Assert):
    from_read = symmetry.read()
    from_dict = symmetry.to_dict()
    assert from_read.keys() == from_dict.keys()
    for key in from_read:
        if from_read[key] is None:
            assert from_dict[key] is None
        else:
            Assert.allclose(from_dict[key], from_read[key])

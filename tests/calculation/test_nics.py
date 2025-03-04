# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import dataclasses
import types

import numpy as np
import pytest

from py4vasp._calculation.nics import Nics
from py4vasp._calculation.structure import Structure


@pytest.fixture
def chemical_shift(raw_data):
    raw_nics = raw_data.nics("Sr2TiO4")
    nics = Nics.from_data(raw_nics)
    nics.ref = types.SimpleNamespace()
    transposed_nics = np.array(raw_nics.nics).T
    nics.ref.structure = Structure.from_data(raw_nics.structure)
    nics.ref.output = {"nics": transposed_nics.reshape((10, 12, 14, 3, 3))}
    return nics


def test_read(chemical_shift, Assert):
    actual = chemical_shift.read()
    actual_structure = actual.pop("structure")
    Assert.same_structure(actual_structure, chemical_shift.ref.structure.read())
    assert actual.keys() == chemical_shift.ref.output.keys()
    Assert.allclose(actual["nics"], chemical_shift.ref.output["nics"])


def get_3d_tensor_element_from_grid(tensor, element: str):
    if element == None:
        return tensor
    elif element == "xx":
        return tensor[:, :, :, 0, 0]
    elif element == "xy":
        return tensor[:, :, :, 0, 1]
    elif element == "xz":
        return tensor[:, :, :, 0, 2]
    elif element == "yx":
        return tensor[:, :, :, 1, 0]
    elif element == "yy":
        return tensor[:, :, :, 1, 1]
    elif element == "yz":
        return tensor[:, :, :, 1, 2]
    elif element == "zx":
        return tensor[:, :, :, 2, 0]
    elif element == "zy":
        return tensor[:, :, :, 2, 1]
    elif element == "zz":
        return tensor[:, :, :, 2, 2]
    elif element == "xx+yy":
        return tensor[:, :, :, 0, 0] + tensor[:, :, :, 1, 1]
    elif element == "xx yy":
        return [tensor[:, :, :, 0, 0], tensor[:, :, :, 1, 1]]
    elif element == "isotropic":
        tensor_sum = (
            tensor[:, :, :, 0, 0] + tensor[:, :, :, 1, 1] + tensor[:, :, :, 2, 2]
        )
        return tensor_sum / 3.0
    else:
        raise ValueError(
            f"Element {element} is unknown by get_3d_tensor_element_from_grid."
        )


@pytest.mark.parametrize(
    "selection",
    [
        None,
        "xx",
        "xy",
        "xz",
        "yx",
        "yy",
        "yz",
        "zx",
        "zy",
        "zz",
        "xx+yy",
        "xx yy",
        "isotropic",
    ],
)
def test_to_numpy(selection, chemical_shift, Assert):
    tensor = chemical_shift.ref.output["nics"]
    element = get_3d_tensor_element_from_grid(tensor, selection)
    Assert.allclose(chemical_shift.to_numpy(selection), element)

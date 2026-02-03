# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp._calculation.piezoelectric_tensor import PiezoelectricTensor
from py4vasp._util.tensor import symmetry_reduce, tensor_constants


@pytest.fixture
def piezoelectric_tensor(raw_data):
    raw_tensor = raw_data.piezoelectric_tensor("default")
    tensor = PiezoelectricTensor.from_data(raw_tensor)
    tensor.ref = types.SimpleNamespace()
    tensor.ref.clamped_ion = raw_tensor.electron
    tensor.ref.relaxed_ion = raw_tensor.ion + raw_tensor.electron
    tensor.ref.piezo = raw_tensor.ion
    tensor.ref.is_2d = False
    return tensor


@pytest.fixture
def piezoelectric_tensor_as_slab(raw_data):
    raw_tensor = raw_data.piezoelectric_tensor("as-slab")
    tensor = PiezoelectricTensor.from_data(raw_tensor)
    tensor.ref = types.SimpleNamespace()
    tensor.ref.clamped_ion = raw_tensor.electron
    tensor.ref.relaxed_ion = raw_tensor.ion + raw_tensor.electron
    tensor.ref.piezo = raw_tensor.ion
    tensor.ref.is_2d = True
    return tensor


def test_read(piezoelectric_tensor, Assert):
    actual = piezoelectric_tensor.read()
    Assert.allclose(actual["clamped_ion"], piezoelectric_tensor.ref.clamped_ion)
    Assert.allclose(actual["relaxed_ion"], piezoelectric_tensor.ref.relaxed_ion)


def test_print(piezoelectric_tensor, format_):
    actual, _ = format_(piezoelectric_tensor)
    reference = f"""
Piezoelectric tensor (C/m²)
         XX          YY          ZZ          XY          YZ          ZX
---------------------------------------------------------------------------
                                clamped-ion
 x     0.00000     4.00000     8.00000     2.00000     6.00000     4.00000
 y     9.00000    13.00000    17.00000    11.00000    15.00000    13.00000
 z    18.00000    22.00000    26.00000    20.00000    24.00000    22.00000
                                relaxed-ion
 x    27.00000    35.00000    43.00000    31.00000    39.00000    35.00000
 y    45.00000    53.00000    61.00000    49.00000    57.00000    53.00000
 z    63.00000    71.00000    79.00000    67.00000    75.00000    71.00000
""".strip()
    assert actual == {"text/plain": reference}


def _check_to_database(piezoelectric_tensor):
    db_dict = piezoelectric_tensor._read_to_database()["piezoelectric_tensor:default"]
    for idx, suffix in enumerate(["x", "y", "z"]):
        assert db_dict[f"3d_tensor_reduced_{suffix}"] == list(
            symmetry_reduce(piezoelectric_tensor.ref.piezo[idx])
        )
        assert db_dict[f"3d_piezoelectric_stress_coefficient_{suffix}"] is not None
    for desc in ["mean_absolute", "rms", "frobenius_norm"]:
        assert db_dict[f"3d_{desc}"] is not None
    if piezoelectric_tensor.ref.is_2d:
        assert db_dict["2d_plane"] is not None
        assert db_dict["2d_tensor_reduced"] is not None
        assert isinstance(db_dict["2d_plane"], str)
    else:
        assert db_dict["2d_plane"] is None
        assert db_dict["2d_tensor_reduced"] is None


def test_to_database(piezoelectric_tensor, Assert):
    _check_to_database(piezoelectric_tensor)


def test_to_database_as_slab(piezoelectric_tensor_as_slab, Assert):
    _check_to_database(piezoelectric_tensor_as_slab)


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.piezoelectric_tensor("default")
    check_factory_methods(PiezoelectricTensor, data)

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp._calculation.piezoelectric_tensor import (
    PiezoelectricTensor,
    _extract_tensor,
)
from py4vasp._raw.data_db import PiezoelectricTensor_DB
from py4vasp._util.tensor import symmetry_reduce


@pytest.fixture
def piezoelectric_tensor(raw_data):
    raw_tensor = raw_data.piezoelectric_tensor("default")
    tensor = PiezoelectricTensor.from_data(raw_tensor)
    tensor.ref = types.SimpleNamespace()
    tensor.ref.clamped_ion = raw_tensor.electron
    tensor.ref.relaxed_ion = raw_tensor.ion + raw_tensor.electron
    tensor.ref.piezo = [
        raw_tensor.ion + raw_tensor.electron,
        raw_tensor.ion,
        raw_tensor.electron,
    ]
    tensor.ref.is_2d = False
    tensor.ref.overview_data = {
        "total_3d_piezoelectric_stress_coefficient_x": 27.0,
        "total_3d_piezoelectric_stress_coefficient_y": 53.0,
        "total_3d_piezoelectric_stress_coefficient_z": 79.0,
        "total_3d_mean_absolute": 49.0,
        "total_3d_rms": 51.6107224001628,
        "total_3d_frobenius_norm": 218.9657507465494,
    }
    return tensor


@pytest.fixture
def piezoelectric_tensor_as_slab(raw_data):
    raw_tensor = raw_data.piezoelectric_tensor("as-slab")
    tensor = PiezoelectricTensor.from_data(raw_tensor)
    tensor.ref = types.SimpleNamespace()
    tensor.ref.clamped_ion = raw_tensor.electron
    tensor.ref.relaxed_ion = raw_tensor.ion + raw_tensor.electron
    tensor.ref.piezo = [
        raw_tensor.ion + raw_tensor.electron,
        raw_tensor.ion,
        raw_tensor.electron,
    ]
    tensor.ref.is_2d = True
    tensor.ref.overview_data = {
        "total_3d_piezoelectric_stress_coefficient_x": 27.0,
        "total_3d_piezoelectric_stress_coefficient_y": 53.0,
        "total_3d_piezoelectric_stress_coefficient_z": 79.0,
        "total_3d_mean_absolute": 49.0,
        "total_3d_rms": 51.6107224001628,
        "total_3d_frobenius_norm": 218.9657507465494,
    }
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
    db_data: PiezoelectricTensor_DB = piezoelectric_tensor._read_to_database()[
        "piezoelectric_tensor:default"
    ]
    assert isinstance(db_data, PiezoelectricTensor_DB)
    for idx, prefix in enumerate(["total", "ionic", "electronic"]):
        sum_2d_tensor_not_none = 0
        for idy, suffix in enumerate(["x", "y", "z"]):
            assert getattr(db_data, f"{prefix}_3d_tensor_{suffix}") == list(
                _extract_tensor(piezoelectric_tensor.ref.piezo[idx])[
                    idy, (0, 1, 2, 5, 3, 4)
                ]
            )
            assert (
                getattr(
                    db_data, f"{prefix}_3d_piezoelectric_stress_coefficient_{suffix}"
                )
                is not None
            )
            if not piezoelectric_tensor.ref.is_2d:
                assert getattr(db_data, f"{prefix}_2d_tensor_{suffix}") is None
            elif getattr(db_data, f"{prefix}_2d_tensor_{suffix}") is not None:
                sum_2d_tensor_not_none += 1
        for desc in ["mean_absolute", "rms", "frobenius_norm"]:
            assert getattr(db_data, f"{prefix}_3d_{desc}") is not None
        if piezoelectric_tensor.ref.is_2d:
            assert sum_2d_tensor_not_none == 2
    # TODO add real reference data for computed values
    for key, value in piezoelectric_tensor.ref.overview_data.items():
        assert getattr(db_data, key) == value


def test_to_database(piezoelectric_tensor, Assert):
    _check_to_database(piezoelectric_tensor)


def test_to_database_as_slab(piezoelectric_tensor_as_slab, Assert):
    _check_to_database(piezoelectric_tensor_as_slab)


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.piezoelectric_tensor("default")
    check_factory_methods(PiezoelectricTensor, data)

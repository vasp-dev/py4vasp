# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
from typing import List, Optional

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._raw.data_wrapper import VaspData
from py4vasp._raw.models import (
    BandModel,
    IntVector,
    StressModel,
    StructureModel,
    Vector,
    Voigt,
    VoigtMatrix,
    _coerce_field,
    _DatabaseModel,
)

# --- scalar leaf types -----------------------------------------------------


def test_float_accepts_numpy_float_and_int():
    result = _coerce_field(np.float64(1.5), float, "x")
    assert result == 1.5 and type(result) is float
    result = _coerce_field(3, float, "x")
    assert result == 3.0 and type(result) is float


def test_float_rejects_string_and_bool():
    with pytest.raises(exception.DataMismatch):
        _coerce_field("nope", float, "x")
    with pytest.raises(exception.DataMismatch):
        _coerce_field(True, float, "x")


def test_int_accepts_numpy_int_rejects_float_and_bool():
    result = _coerce_field(np.int64(4), int, "x")
    assert result == 4 and type(result) is int
    with pytest.raises(exception.DataMismatch):
        _coerce_field(3.5, int, "x")
    with pytest.raises(exception.DataMismatch):
        _coerce_field(True, int, "x")


def test_bool_accepts_numpy_bool_rejects_int():
    result = _coerce_field(np.bool_(True), bool, "x")
    assert result is True and type(result) is bool
    with pytest.raises(exception.DataMismatch):
        _coerce_field(1, bool, "x")


def test_str_accepts_numpy_str():
    result = _coerce_field(np.str_("Fm-3m"), str, "x")
    assert result == "Fm-3m" and type(result) is str


def test_str_decodes_bytes():
    result = _coerce_field(b"P1", str, "x")
    assert result == "P1" and type(result) is str


def test_zero_dim_array_coerces_to_scalar():
    result = _coerce_field(np.array(2.0), float, "x")
    assert result == 2.0 and type(result) is float


# --- Optional --------------------------------------------------------------


def test_optional_allows_none():
    assert _coerce_field(None, Optional[float], "x") is None


def test_non_optional_rejects_none():
    with pytest.raises(exception.DataMismatch):
        _coerce_field(None, float, "x")


def test_vaspdata_none_treated_as_none():
    assert _coerce_field(VaspData(None), Optional[float], "x") is None


# --- fixed-size Tuple aliases ---------------------------------------------


def test_vector_coerces_to_tuple_of_floats():
    result = _coerce_field([1, 2, 3], Vector, "v")
    assert result == (1.0, 2.0, 3.0)
    assert type(result) is tuple
    assert all(type(x) is float for x in result)
    result = _coerce_field(np.array([1.0, 2.0, 3.0]), Vector, "v")
    assert result == (1.0, 2.0, 3.0) and type(result) is tuple


def test_vector_enforces_length():
    with pytest.raises(exception.DataMismatch):
        _coerce_field([1.0, 2.0], Vector, "v")
    with pytest.raises(exception.DataMismatch):
        _coerce_field([1.0, 2.0, 3.0, 4.0], Vector, "v")


def test_int_vector_rejects_float_elements():
    result = _coerce_field([1, 2, 3], IntVector, "grid")
    assert result == (1, 2, 3) and all(type(x) is int for x in result)
    with pytest.raises(exception.DataMismatch):
        _coerce_field([1.0, 2.0, 3.0], IntVector, "grid")


def test_voigt_enforces_length_six():
    good = _coerce_field(np.arange(6.0), Voigt, "t")
    assert good == (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    with pytest.raises(exception.DataMismatch):
        _coerce_field(np.arange(5.0), Voigt, "t")


def test_voigt_matrix_enforces_six_by_six():
    result = _coerce_field(np.zeros((6, 6)), VoigtMatrix, "m")
    assert type(result) is tuple and len(result) == 6
    assert all(type(row) is tuple and len(row) == 6 for row in result)
    with pytest.raises(exception.DataMismatch):
        _coerce_field(np.zeros((6, 5)), VoigtMatrix, "m")
    with pytest.raises(exception.DataMismatch):
        _coerce_field(np.zeros((5, 6)), VoigtMatrix, "m")


# --- variable-length List --------------------------------------------------


def test_list_is_variable_length_and_checks_elements():
    assert _coerce_field(["a", "b"], List[str], "labels") == ["a", "b"]
    assert _coerce_field([], List[str], "labels") == []
    result = _coerce_field(np.array([1, 2, 3]), List[int], "counts")
    assert result == [1, 2, 3] and all(type(x) is int for x in result)
    with pytest.raises(exception.DataMismatch):
        _coerce_field([1, 2], List[str], "labels")


def test_scalar_where_sequence_expected_raises():
    with pytest.raises(exception.DataMismatch):
        _coerce_field(1.0, Vector, "v")


def test_error_message_names_field():
    with pytest.raises(exception.DataMismatch, match="fermi_energy"):
        _coerce_field("bad", float, "fermi_energy")


# --- validation wired into model construction (__post_init__) --------------


def test_model_coerces_numpy_scalar_to_native_float():
    model = StressModel(initial_stress_mean=np.float64(1.5))
    assert model.initial_stress_mean == 1.5
    assert type(model.initial_stress_mean) is float


def test_model_coerces_numpy_array_to_tuple():
    model = StressModel(final_stress_tensor=np.arange(6.0))
    assert model.final_stress_tensor == (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    assert type(model.final_stress_tensor) is tuple


def test_model_coerces_list_of_numpy_to_tuple_of_float():
    model = StructureModel(
        lattice_vector_1=[np.float64(1.0), np.float64(2.0), np.float64(3.0)]
    )
    assert model.lattice_vector_1 == (1.0, 2.0, 3.0)
    assert all(type(x) is float for x in model.lattice_vector_1)


def test_model_defaults_stay_none():
    model = StressModel()
    assert model.initial_stress_mean is None
    assert model.final_stress_tensor is None


def test_model_rejects_wrong_length_vector():
    with pytest.raises(exception.DataMismatch, match="final_stress_tensor"):
        StressModel(final_stress_tensor=[1.0, 2.0])


def test_model_rejects_wrong_scalar_type():
    with pytest.raises(exception.DataMismatch, match="fermi_energy"):
        BandModel(fermi_energy="not a number")


def test_model_is_json_serializable_after_construction():
    import dataclasses
    import json

    model = StressModel(
        initial_stress_mean=np.float64(1.5), final_stress_tensor=np.arange(6.0)
    )
    json.dumps(dataclasses.asdict(model))  # must not raise


# --- _DatabaseModel structural guarantees ----------------------------------


def test_model_has_no_schema_version_field():
    """schema_version lives in the metadata only; a model must not carry it."""

    @dataclasses.dataclass
    class SampleModel(_DatabaseModel):
        value: int = 0

    instance = SampleModel(value=42)
    assert not hasattr(instance, "__schema_version__")
    assert not hasattr(instance, "schema_version")


def test_vaspdata_is_unwrapped_in_model():
    """VaspData assigned to a field is replaced by its underlying data."""

    @dataclasses.dataclass
    class SampleModel(_DatabaseModel):
        field1: Optional[int] = None
        field2: Optional[str] = None
        field3: Optional[bool] = None

    instance = SampleModel(field1=VaspData(None), field2=VaspData("test"), field3=True)
    assert instance.field1 is None
    assert instance.field2 == "test"
    assert instance.field3 is True

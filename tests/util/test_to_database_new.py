# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Tests for the simplified _to_database contract.

These tests define the new expected behaviour:
  - _DatabaseData has exactly two fields: metadata and properties.
  - CalculationMetaData has a `path` (directory) and `schema_version`,
    but no `tags` and no `hdf5_original_path`.
  - _to_database() takes no arguments.
  - properties keys follow the format <quantity> (default) or
    <quantity>_<selection> (non-default), with no leading underscore.
  - schema_version is stored only in metadata, not on individual _DB dataclasses.
"""

import dataclasses
import pathlib

import pytest

from py4vasp import demo
from py4vasp._raw.data import CalculationMetaData, _DatabaseData
from py4vasp._raw.data_db import _DBDataMixin

# ---------------------------------------------------------------------------
# Structural tests — these do not need a running calculation
# ---------------------------------------------------------------------------


def test_database_data_has_only_metadata_and_properties():
    """_DatabaseData must have exactly the two fields metadata and properties."""
    field_names = {f.name for f in dataclasses.fields(_DatabaseData)}
    assert field_names == {"metadata", "properties"}


def test_metadata_field_path_not_hdf5_path():
    """CalculationMetaData uses 'path' for the directory, not hdf5_original_path."""
    field_names = {f.name for f in dataclasses.fields(CalculationMetaData)}
    assert "path" in field_names
    assert "hdf5_original_path" not in field_names


def test_metadata_has_schema_version_field():
    """CalculationMetaData must expose schema_version."""
    field_names = {f.name for f in dataclasses.fields(CalculationMetaData)}
    assert "schema_version" in field_names


def test_metadata_has_no_tags_field():
    """tags was removed from CalculationMetaData; _to_database() takes no arguments."""
    field_names = {f.name for f in dataclasses.fields(CalculationMetaData)}
    assert "tags" not in field_names


def test_metadata_has_file_presence_flags():
    """CalculationMetaData must still expose has_incar, has_poscar, etc."""
    field_names = {f.name for f in dataclasses.fields(CalculationMetaData)}
    for flag in ("has_incar", "has_poscar", "has_kpoints", "has_potcar"):
        assert flag in field_names, f"Missing flag {flag!r} in CalculationMetaData"


def test_dbdatamixin_no_schema_version_field():
    """schema_version is now in metadata only; _DBDataMixin must not carry it."""

    @dataclasses.dataclass
    class SampleDB(_DBDataMixin):
        value: int = 0

    instance = SampleDB(value=42)
    # Neither the double-underscore form nor a plain attribute
    assert not hasattr(instance, "__schema_version__")
    assert not hasattr(instance, "schema_version")


# ---------------------------------------------------------------------------
# Integration tests — need a real (demo) calculation
# ---------------------------------------------------------------------------


@pytest.fixture
def demo_db(tmp_path):
    actual_path = tmp_path / "demo_calc"
    calc = demo.calculation(actual_path)
    return calc._to_database()


def test_to_database_returns_database_data(demo_db):
    assert isinstance(demo_db, _DatabaseData)


def test_to_database_takes_no_arguments(tmp_path):
    """_to_database() must not accept any arguments at all (no tags, no fermi_energy)."""
    import inspect

    from py4vasp._calculation import Calculation

    sig = inspect.signature(Calculation._to_database)
    params = [name for name, _ in sig.parameters.items() if name != "self"]
    # No parameters at all beyond 'self'
    assert params == []


def test_metadata_path_is_calculation_directory(tmp_path, demo_db):
    assert isinstance(demo_db.metadata.path, pathlib.Path)
    # path must be a directory, not a .h5 file
    assert demo_db.metadata.path.is_dir()
    assert not demo_db.metadata.path.name.endswith(".h5")


def test_metadata_schema_version_is_nonempty_string(demo_db):
    assert isinstance(demo_db.metadata.schema_version, str)
    assert demo_db.metadata.schema_version != ""


def test_properties_is_dict(demo_db):
    assert isinstance(demo_db.properties, dict)


def test_properties_has_entries(demo_db):
    assert len(demo_db.properties) > 0


def test_no_leading_underscore_in_properties_keys(demo_db):
    """Private quantities like _CONTCAR must be stored under 'CONTCAR', not '_CONTCAR'."""
    for key in demo_db.properties:
        assert not key.startswith("_"), f"Key {key!r} starts with underscore"


def test_run_info_in_properties(demo_db):
    """run_info is always available and must be present in properties."""
    assert "run_info" in demo_db.properties


def test_default_selection_key_has_no_suffix(demo_db):
    """Keys for the default selection must not have a '_default' suffix."""
    for key in demo_db.properties:
        assert not key.endswith("_default"), f"Key {key!r} still has '_default' suffix"


def test_non_default_selection_key_format(tmp_path):
    """Non-default selections are appended with an underscore: quantity_selection."""
    actual_path = tmp_path / "demo_calc_band"
    calc = demo.calculation(actual_path)
    db = calc._to_database()
    # band has kpoints_opt and kpoints_wan selections in the schema.
    # If the demo data doesn't have them, they simply won't appear — that is fine.
    # But if they DO appear, the key format must be 'band_kpoints_opt' not
    # 'band:kpoints_opt' or 'band.kpoints_opt'.
    for key in db.properties:
        assert ":" not in key, f"Key {key!r} contains a colon"
        assert "." not in key, f"Key {key!r} contains a dot"

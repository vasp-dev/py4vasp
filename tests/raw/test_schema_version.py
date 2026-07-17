# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import json
import re
from pathlib import Path

import pytest

from py4vasp import __version__
from py4vasp._raw import models
from py4vasp._util import database

SNAPSHOT_PATH = Path(__file__).parent / "schema_snapshot.json"


def _snapshot(version, models_):
    return {"schema_version": version, "models": models_}


MODELS_A = {"BandModel": [["fermi_energy", "Optional[float]"]]}
MODELS_B = {"BandModel": [["fermi_energy", "Optional[float]"], ["gap", "Optional[float]"]]}


def test_parse_schema_version():
    assert models.parse_schema_version("0.11+db.3") == ("0.11", 3)
    assert models.parse_schema_version("0.12+db.1") == ("0.12", 1)


def test_schema_version_format():
    assert re.fullmatch(r"\d+\.\d+\+db\.\d+", models.schema_version())


def test_schema_version_uses_py4vasp_series_and_counter():
    major, minor = __version__.split(".")[:2]
    expected = f"{major}.{minor}+db.{models.__DB_SCHEMA__}"
    assert models.schema_version() == expected


def test_db_schema_counter_is_a_positive_int():
    assert isinstance(models.__DB_SCHEMA__, int)
    assert models.__DB_SCHEMA__ >= 1


# ---------------------------------------------------------------------------
# check_schema_snapshot: enforces that any model change advances the version
# ---------------------------------------------------------------------------


def test_check_ok_when_nothing_changed():
    stored = _snapshot("0.11+db.1", MODELS_A)
    current = _snapshot("0.11+db.1", MODELS_A)
    assert database.check_schema_snapshot(stored, current) is None


def test_check_rejects_version_change_without_model_change():
    stored = _snapshot("0.11+db.1", MODELS_A)
    current = _snapshot("0.11+db.2", MODELS_A)
    problem = database.check_schema_snapshot(stored, current)
    assert problem is not None
    assert "unchanged" in problem.lower()


def test_check_ok_when_models_changed_and_counter_incremented():
    stored = _snapshot("0.11+db.1", MODELS_A)
    current = _snapshot("0.11+db.2", MODELS_B)
    assert database.check_schema_snapshot(stored, current) is None


def test_check_rejects_model_change_without_counter_bump():
    stored = _snapshot("0.11+db.1", MODELS_A)
    current = _snapshot("0.11+db.1", MODELS_B)
    problem = database.check_schema_snapshot(stored, current)
    assert problem is not None
    assert "__DB_SCHEMA__" in problem
    # the diff should mention the added field
    assert "gap" in problem


def test_check_ok_when_py4vasp_series_advances_and_counter_resets():
    stored = _snapshot("0.11+db.3", MODELS_A)
    current = _snapshot("0.12+db.1", MODELS_B)
    assert database.check_schema_snapshot(stored, current) is None


def test_check_rejects_new_series_without_counter_reset():
    stored = _snapshot("0.11+db.3", MODELS_A)
    current = _snapshot("0.12+db.4", MODELS_B)
    problem = database.check_schema_snapshot(stored, current)
    assert problem is not None
    assert "reset" in problem.lower()


# ---------------------------------------------------------------------------
# schema_fingerprint: the canonical, serializable description of every model
# ---------------------------------------------------------------------------


def test_fingerprint_top_level_shape():
    fingerprint = models.schema_fingerprint()
    assert fingerprint["schema_version"] == models.schema_version()
    assert isinstance(fingerprint["models"], dict)


def test_fingerprint_includes_all_models_but_not_the_base():
    model_names = models.schema_fingerprint()["models"]
    for name in ("BandModel", "StructureModel", "DispersionModel", "StoichiometryModel"):
        assert name in model_names
    for name in ("EnergyRelaxationModel", "EnergyMDModel", "EnergyAfqmcModel"):
        assert name in model_names
    assert "_DatabaseModel" not in model_names


def test_fingerprint_is_sorted_and_field_entries_are_name_type_pairs():
    model_dict = models.schema_fingerprint()["models"]
    assert list(model_dict) == sorted(model_dict)
    for fields in model_dict.values():
        assert [f[0] for f in fields] == sorted(f[0] for f in fields)
        for entry in fields:
            assert len(entry) == 2
            assert all(isinstance(part, str) for part in entry)


def test_fingerprint_is_json_serializable():
    dumped = json.dumps(models.schema_fingerprint())
    assert json.loads(dumped) == models.schema_fingerprint()


def test_fingerprint_types_are_canonical_and_interpreter_independent():
    """Types must render the same on every Python version, so the snapshot is stable."""
    model_dict = models.schema_fingerprint()["models"]
    band = dict(model_dict["BandModel"])
    assert band["fermi_energy"] == "Optional[float]"
    stoichiometry = dict(model_dict["StoichiometryModel"])
    assert stoichiometry["ion_types"] == "Optional[list[str]]"
    tensor = dict(model_dict["DielectricTensorModel"])
    assert tensor["total_3d_tensor"] == "Optional[list[list[float]]]"


# ---------------------------------------------------------------------------
# The drift test: fails whenever the models change without a version bump.
# Regenerate the committed snapshot with
#   pytest tests/raw/test_schema_version.py --update-schema-snapshot
# ---------------------------------------------------------------------------


def test_database_schema_matches_snapshot(request):
    current = models.schema_fingerprint()
    if request.config.getoption("--update-schema-snapshot"):
        _update_snapshot(current)
        return
    assert SNAPSHOT_PATH.exists(), (
        f"Missing {SNAPSHOT_PATH.name}; create it with "
        "`pytest tests/raw/test_schema_version.py --update-schema-snapshot`."
    )
    stored = json.loads(SNAPSHOT_PATH.read_text())
    problem = database.check_schema_snapshot(stored, current)
    assert problem is None, problem
    assert stored["models"] == current["models"], (
        "The database schema snapshot is stale. Regenerate it with "
        "`pytest tests/raw/test_schema_version.py --update-schema-snapshot`."
    )


def _update_snapshot(current):
    if SNAPSHOT_PATH.exists():
        stored = json.loads(SNAPSHOT_PATH.read_text())
        models_changed = stored["models"] != current["models"]
        version_unchanged = stored["schema_version"] == current["schema_version"]
        if models_changed and version_unchanged:
            pytest.fail(
                "Refusing to update the snapshot: the models changed but the schema "
                "version did not. Increment __DB_SCHEMA__ in models.py first.\n"
                + database.check_schema_snapshot(stored, current)
            )
    SNAPSHOT_PATH.write_text(json.dumps(current, indent=2) + "\n")

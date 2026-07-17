# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import json
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
    # the default (released) version carries no "+db" suffix -> counter 0
    assert models.parse_schema_version("0.11") == ("0.11", 0)
    assert models.parse_schema_version("0.12") == ("0.12", 0)
    # intermediate database migrations carry the counter
    assert models.parse_schema_version("0.11+db.3") == ("0.11", 3)


def test_schema_version_default_has_no_db_suffix(monkeypatch):
    monkeypatch.setattr(models, "__DB_SCHEMA__", 0)
    major, minor = __version__.split(".")[:2]
    assert models.schema_version() == f"{major}.{minor}"


def test_schema_version_shows_counter_only_when_nonzero(monkeypatch):
    monkeypatch.setattr(models, "__DB_SCHEMA__", 5)
    major, minor = __version__.split(".")[:2]
    assert models.schema_version() == f"{major}.{minor}+db.5"


def test_db_schema_counter_is_a_nonnegative_int():
    assert isinstance(models.__DB_SCHEMA__, int)
    assert models.__DB_SCHEMA__ >= 0


# ---------------------------------------------------------------------------
# check_schema_snapshot: enforces that any model change advances the version
# ---------------------------------------------------------------------------


def test_check_ok_when_nothing_changed():
    stored = _snapshot("0.11", MODELS_A)
    current = _snapshot("0.11", MODELS_A)
    assert database.check_schema_snapshot(stored, current) is None


def test_check_rejects_version_change_without_model_change():
    stored = _snapshot("0.11", MODELS_A)
    current = _snapshot("0.11+db.1", MODELS_A)
    problem = database.check_schema_snapshot(stored, current)
    assert problem is not None
    assert "unchanged" in problem.lower()


def test_check_ok_when_models_changed_and_counter_incremented():
    stored = _snapshot("0.11", MODELS_A)
    current = _snapshot("0.11+db.1", MODELS_B)
    assert database.check_schema_snapshot(stored, current) is None


def test_check_rejects_model_change_without_counter_bump():
    stored = _snapshot("0.11", MODELS_A)
    current = _snapshot("0.11", MODELS_B)
    problem = database.check_schema_snapshot(stored, current)
    assert problem is not None
    assert "__DB_SCHEMA__" in problem
    # the diff should mention the added field
    assert "gap" in problem


def test_check_ok_when_py4vasp_series_advances_with_clean_reset():
    # a new py4vasp release resets to the bare series; models may or may not change
    assert database.check_schema_snapshot(_snapshot("0.11", MODELS_A), _snapshot("0.12", MODELS_A)) is None
    assert database.check_schema_snapshot(_snapshot("0.11+db.3", MODELS_A), _snapshot("0.12", MODELS_B)) is None


def test_check_rejects_new_series_carrying_a_counter():
    # "0.11+db.17" was never a released model version, so "0.12+db.1" must not
    # inherit a counter across the release -- it has to reset to the bare "0.12".
    stored = _snapshot("0.11+db.17", MODELS_A)
    current = _snapshot("0.12+db.1", MODELS_A)
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
    # genuinely variable-length fields stay as list
    assert stoichiometry["ion_types"] == "Optional[list[str]]"
    # fixed-size fields use the tuple aliases, so the length is part of the type
    vec3 = "tuple[float, float, float]"
    voigt = "tuple[float, float, float, float, float, float]"
    voigt_matrix = f"tuple[{', '.join([voigt] * 6)}]"
    assert dict(model_dict["StructureModel"])["lattice_vector_1"] == f"Optional[{vec3}]"
    assert dict(model_dict["CurrentDensityModel"])["grid_shape"] == "Optional[tuple[int, int, int]]"
    assert dict(model_dict["DielectricTensorModel"])["total_3d_tensor"] == f"Optional[{voigt}]"
    assert dict(model_dict["StressModel"])["final_stress_tensor"] == f"Optional[{voigt}]"
    assert dict(model_dict["ElasticModulusModel"])["total_3d_tensor"] == f"Optional[{voigt_matrix}]"


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
        problem = database.check_schema_snapshot(stored, current)
        if problem is not None:
            pytest.fail(
                "Refusing to update the snapshot; the version transition is illegal.\n"
                + problem
            )
    SNAPSHOT_PATH.write_text(json.dumps(current, indent=2) + "\n")

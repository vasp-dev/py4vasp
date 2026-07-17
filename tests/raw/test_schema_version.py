# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import re

import pytest

from py4vasp import __version__
from py4vasp._raw import models
from py4vasp._util import database


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

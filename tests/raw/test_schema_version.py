# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import re

from py4vasp import __version__
from py4vasp._raw import models


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

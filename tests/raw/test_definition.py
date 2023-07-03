# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from unittest.mock import patch

from py4vasp._raw.definition import get_schema, schema


def test_all_quantities_have_default():
    for source in schema.sources.values():
        assert "default" in source


def test_schema_is_valid():
    schema.verify()


def test_get_schema(complex_schema):
    mock_schema, _ = complex_schema
    with patch("py4vasp._raw.definition.schema", mock_schema):
        assert get_schema() == str(mock_schema)

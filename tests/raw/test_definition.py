# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from unittest.mock import patch

from py4vasp import raw
from py4vasp._raw.definition import schema


def test_all_quantities_have_default():
    for quantity, source in schema.sources.items():
        if quantity == "current_density":
            # currently no default current density is implemented
            assert "default" not in source
        else:
            assert "default" in source


def test_schema_is_valid():
    schema.verify()


def test_get_schema(complex_schema):
    mock_schema, _ = complex_schema
    with patch("py4vasp._raw.definition.schema", mock_schema):
        assert raw.get_schema() == str(mock_schema)


def test_get_selections(complex_schema):
    mock_schema, _ = complex_schema
    with patch("py4vasp._raw.definition.schema", mock_schema):
        for quantity in mock_schema.sources.keys():
            assert raw.selections(quantity) == mock_schema.selections(quantity)

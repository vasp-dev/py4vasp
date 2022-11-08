# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._raw.definition import schema


def test_all_quantities_have_default():
    for source in schema.sources.values():
        assert "default" in source


def test_schema_is_valid():
    schema.verify()

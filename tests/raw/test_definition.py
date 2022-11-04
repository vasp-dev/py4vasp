from py4vasp.raw._definition import schema


def test_all_quantities_have_default():
    for source in schema.sources.values():
        assert "default" in source

def test_schema_is_valid():
    schema.verify()

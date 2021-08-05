from py4vasp._util.convert import text_to_string


def test_text_to_string():
    assert text_to_string(b"foo") == "foo"
    assert text_to_string("bar") == "bar"

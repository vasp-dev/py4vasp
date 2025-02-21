# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._util import suggest


def test_similar_selection():
    assert suggest.did_you_mean("bar", ["foo", "baz"]) == 'Did you mean "baz"? '


def test_no_similar_selection():
    assert suggest.did_you_mean("foo", ["bar", "baz"]) == ""


def test_key_is_converted_to_string():
    assert suggest.did_you_mean(120, ["99", "121", "700"]) == 'Did you mean "121"? '

def test_possibilities_are_converted_to_string():
    assert suggest.did_you_mean("320", range(291, 330, 10)) == 'Did you mean "321"? '
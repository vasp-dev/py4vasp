# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._util import suggest


def test_similar_selection():
    assert suggest.did_you_mean("bar", ["foo", "baz"]) == 'Did you mean "baz"? '


def test_no_similar_selection():
    assert suggest.did_you_mean("foo", ["bar", "baz"]) == ""

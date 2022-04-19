# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import py4vasp._util.sanity_check as check
import py4vasp.exceptions as exception
import pytest


def test_error_if_not_string():
    with pytest.raises(exception.IncorrectUsage):
        check.raise_error_if_not_string(1, "message")

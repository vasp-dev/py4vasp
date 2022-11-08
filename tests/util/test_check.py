# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest

from py4vasp._util import check
import py4vasp.exceptions as exception


def test_error_if_not_string():
    check.raise_error_if_not_string("string does not raise exception", "message")
    with pytest.raises(exception.IncorrectUsage):
        check.raise_error_if_not_string(1, "message")


def test_error_if_not_number():
    check.raise_error_if_not_number(1, "number does not raise exception")
    with pytest.raises(exception.IncorrectUsage):
        check.raise_error_if_not_number("should be number", "message")


def test_error_if_not_callable():
    def func(x, y=1):
        pass

    # valid calls do not raise error
    check.raise_error_if_not_callable(func, 0)
    check.raise_error_if_not_callable(func, 1, 2)
    check.raise_error_if_not_callable(func, x=3, y=4)

    # invalid calls raise error
    with pytest.raises(exception.IncorrectUsage):
        check.raise_error_if_not_callable(func, y=5)
    with pytest.raises(exception.IncorrectUsage):
        check.raise_error_if_not_callable(func, 6, 7, 8)
    with pytest.raises(exception.IncorrectUsage):
        check.raise_error_if_not_callable(func, 9, z=10)

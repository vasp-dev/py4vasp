# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp import exception, raw
from py4vasp._util import check


@pytest.mark.parametrize("is_none", (None, raw.VaspData(None)))
def test_check_is_none(is_none):
    assert check.is_none(is_none)


@pytest.mark.parametrize("is_not_none", (np.zeros(3), [1, 2], (2, 3), "text", 2))
def test_check_is_not_none(is_not_none):
    assert not check.is_none(is_not_none)


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

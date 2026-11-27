# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest

from py4vasp import exception

_ERRORS = [
    error
    for error in vars(exception).values()
    if isinstance(error, type) and issubclass(error, exception.Py4VaspError)
]


def test_incorrect_usage_names_no_field_by_default():
    assert exception.IncorrectUsage("message").field is None


def test_incorrect_usage_names_the_field_the_user_got_wrong():
    error = exception.IncorrectUsage("The cutoff must be positive.", field="cutoff")
    assert error.field == "cutoff"
    assert str(error) == "The cutoff must be positive."


@pytest.mark.parametrize("error", _ERRORS)
def test_every_error_of_py4vasp_can_name_a_field(error):
    # a tool reads the field of whatever py4vasp raises, not only of IncorrectUsage
    assert error("message").field is None
    assert error("message", field="selection").field == "selection"

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp import exception


def test_incorrect_usage_names_no_field_by_default():
    assert exception.IncorrectUsage("message").field is None


def test_incorrect_usage_names_the_field_the_user_got_wrong():
    error = exception.IncorrectUsage("The cutoff must be positive.", field="cutoff")
    assert error.field == "cutoff"
    assert str(error) == "The cutoff must be positive."

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numbers
import py4vasp.exceptions as exception


def raise_error_if_not_string(test_if_string, error_message):
    if test_if_string.__class__ != str:
        raise exception.IncorrectUsage(error_message)


def raise_error_if_not_number(test_if_number, error_message):
    if not isinstance(test_if_number, numbers.Number):
        raise exception.IncorrectUsage(error_message)

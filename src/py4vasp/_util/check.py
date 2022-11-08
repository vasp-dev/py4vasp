# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import inspect
import numbers

from py4vasp import exception


def raise_error_if_not_string(test_if_string, error_message):
    if test_if_string.__class__ != str:
        raise exception.IncorrectUsage(error_message)


def raise_error_if_not_number(test_if_number, error_message):
    if not isinstance(test_if_number, numbers.Number):
        raise exception.IncorrectUsage(error_message)


def raise_error_if_not_callable(function, *args, **kwargs):
    signature = inspect.signature(function)
    try:
        signature.bind(*args, **kwargs)
    except TypeError as error:
        message = f"You tried to call {function.__name__}, but the arguments are incorrect! Please double check your input."
        raise exception.IncorrectUsage(message) from error

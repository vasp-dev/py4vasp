from contextlib import contextmanager, nullcontext
from IPython.core.formatters import DisplayFormatter
from IPython.lib.pretty import pretty
from typing import NamedTuple, Iterable
from pathlib import Path
from py4vasp.raw import RawVersion
import py4vasp.exceptions as exception
import py4vasp.raw as raw
import numpy as np
import functools
import numbers


def raise_error_if_not_string(test_if_string, error_message):
    if test_if_string.__class__ != str:
        raise exception.IncorrectUsage(error_message)


def raise_error_if_not_number(test_if_number, error_message):
    if not isinstance(test_if_number, numbers.Number):
        raise exception.IncorrectUsage(error_message)


def add_doc(doc):
    def add_doc_to_func(func):
        func.__doc__ = doc
        return func

    return add_doc_to_func


def decode_if_possible(string):
    try:
        return string.decode()
    except (UnicodeDecodeError, AttributeError):
        return string


default_selection = "*"


class Selection(NamedTuple):
    "Helper class specifying which indices to extract their label."
    indices: Iterable[int]
    "Indices from which the specified quantity is read."
    label: str = ""
    "Label identifying the quantity."


format_ = DisplayFormatter().format
_minimal_vasp_version = RawVersion(6, 2)


def require(version, err_msg=None):
    def decorator_require(func):
        @functools.wraps(func)
        def func_with_requirement_test(raw_data, *args, **kwargs):
            my_version = raw_data.version
            if my_version >= version:
                return func(raw_data, *args, **kwargs)
            else:
                error_message = err_msg
                if error_message is None:
                    error_message = (
                        f"You called {func.__qualname__} which is not compatible with "
                        f"the {my_version.major}.{my_version.minor}.{my_version.patch}"
                        " version of Vasp you are using. Please use at least version "
                        f"{version.major}.{version.minor}.{version.patch} for this "
                        "feature."
                    )
                raise exception.OutdatedVaspVersion(error_message)

        return func_with_requirement_test

    return decorator_require


class Reader:
    "Helper class to deal with error handling of the array reading."

    def __init__(self, array):
        self._array = array
        self.shape = np.shape(array)

    def error_message(self, key, err):
        "We can overload this message in a subclass to make it more specific"
        return (
            "Error reading from the array, please check that the shape of the "
            "array is consistent with the access key."
        )

    def __getitem__(self, key):
        try:
            return self._array[key]
        except (ValueError, IndexError, TypeError) as err:
            raise exception.IncorrectUsage(self.error_message(key, err)) from err

    def __len__(self):
        return len(self._array)

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import re
import textwrap

import numpy as np


def text_to_string(text):
    "Text can be either bytes or string"
    try:
        return text.decode()
    except (UnicodeDecodeError, AttributeError):
        return text


def to_complex(array):
    assert array.dtype == np.float64
    assert array.shape[-1] == 2
    return array.view(np.complex128).reshape(array.shape[:-1])


# NOTE: to_snakecase is the function underscore from the inflection package
#       (Copyright (C) 2012-2020 Janne Vanhala)
def to_snakecase(word: str) -> str:
    """
    Make an underscored, lowercase form from the expression in the string.
    Example::
        >>> underscore("DeviceType")
        'device_type'
    As a rule of thumb you can think of :func:`underscore` as the inverse of
    :func:`camelize`, though there are cases where that does not hold::
        >>> camelize(underscore("IOError"))
        'IoError'
    """
    word = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", word)
    word = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", word)
    word = word.replace("-", "_")
    return word.lower()


def to_rgb(hex):
    "Convert a HEX color code to fractional RGB."
    hex = hex.lstrip("#")
    return np.array([int(part, 16) for part in textwrap.wrap(hex, 2)]) / 255

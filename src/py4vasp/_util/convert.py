# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import re
import textwrap

import numpy as np

from py4vasp import exception
from py4vasp._raw.data_wrapper import VaspData


def text_to_string(text):
    "Text can be either bytes or string"
    try:
        return _attempt_str_conversion(text.decode())
    except (UnicodeDecodeError, AttributeError):
        return _attempt_str_conversion(text)


def _attempt_str_conversion(string) -> str:
    try:
        return str(string)
    except Exception as exc:
        raise exception.NoData(
            "The data could not be converted to string (likely missing)."
        ) from exc


def to_complex(array):
    assert array.dtype == np.float64
    assert array.shape[-1] == 2
    return array.view(np.complex128).reshape(array.shape[:-1])


def quantity_name(quantity):
    if quantity in ["CONTCAR"]:
        return quantity
    else:
        return _to_snakecase(quantity)


# NOTE: to_snakecase is the function underscore from the inflection package
#       (Copyright (C) 2012-2020 Janne Vanhala)
def _to_snakecase(word: str) -> str:
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


# NOTE: to_camelcase is based on the function camelize from the inflection package
#       (Copyright (C) 2012-2020 Janne Vanhala)
def to_camelcase(string: str, uppercase_first_letter: bool = True) -> str:
    """Convert strings to CamelCase.

    Examples::

        >>> camelize("device_type")
        'DeviceType'
        >>> camelize("device_type", False)
        'deviceType'

    :func:`camelize` can be thought of as a inverse of :func:`underscore`,
    although there are some cases where that does not hold::

        >>> camelize(underscore("IOError"))
        'IoError'

    :param uppercase_first_letter: if set to `True` :func:`camelize` converts
        strings to UpperCamelCase. If set to `False` :func:`camelize` produces
        lowerCamelCase. Defaults to `True`.
    """
    if uppercase_first_letter:
        return re.sub(r"(?:_|^)(.)", lambda m: m.group(1).upper(), string)
    else:
        return string[0].lower() + to_camelcase(string)[1:]


def to_rgb(hex):
    "Convert a HEX color code to fractional RGB."
    hex = hex.lstrip("#")
    return np.array([int(part, 16) for part in textwrap.wrap(hex, 2)]) / 255


def to_lab(hex):
    "Convert a HEX color code to CIELAB color space."
    rgb = to_rgb(hex)
    rgb_to_xyz = np.array(
        (
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        )
    )
    rgb = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    x, y, z = rgb_to_xyz @ rgb
    xn = 0.950489
    zn = 1.088840
    t0 = (6 / 29) ** 3
    f = lambda t: t ** (1 / 3) if t > t0 else 2 / 29 * (t / t0 + 2)
    l = 116 * f(y) - 16
    a = 500 * (f(x / xn) - f(y))
    b = 200 * (f(y) - f(z / zn))
    return np.array((l, a, b))

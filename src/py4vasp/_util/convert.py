# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import fractions
import re
import unicodedata

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


# VASP labels the k points of a band structure with LaTeX, e.g. "$\Gamma$", and py4vasp
# writes the same markup for the points a KPOINTS file leaves unnamed. Plotly hands
# "$...$" to MathJax, but JupyterLab exposes MathJax in a shape plotly.js cannot use, so a
# single such label makes the whole figure fail to render, and the browser behind
# `to_image` has no MathJax at all and prints the markup verbatim. Unicode says the same
# thing in every backend.
_MATH_MODE = re.compile(r"\$(.+?)\$")
_FRACTION = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")
_OVERLINE = re.compile(r"\\(?:overline|bar)\{([^{}]*)\}")
_COMMAND = re.compile(r"\\([A-Za-z]+)")
_COMBINING_OVERLINE = "\u0305"


def math_to_unicode(text: str) -> str:
    r"""Replace inline LaTeX math by the equivalent Unicode characters.

    Commands without a Unicode equivalent are passed through unchanged, so an unusual
    label degrades to slightly odd text instead of a figure that does not render.

    Examples
    --------
    >>> math_to_unicode(r"$\Gamma$")
    'Γ'
    >>> math_to_unicode(r"M|$\Gamma$")
    'M|Γ'
    >>> math_to_unicode(r"$[\frac{1}{2} 0 0]$")
    '[1/2 0 0]'
    """
    return _MATH_MODE.sub(lambda match: _convert_math(match.group(1)), text)


def _convert_math(formula: str) -> str:
    formula = _FRACTION.sub(r"\1/\2", formula)
    formula = _OVERLINE.sub(rf"\1{_COMBINING_OVERLINE}", formula)
    return _COMMAND.sub(_replace_command, formula)


def _replace_command(match: re.Match) -> str:
    # every Greek letter is named after the LaTeX command producing it
    case = "CAPITAL" if match.group(1)[0].isupper() else "SMALL"
    try:
        return unicodedata.lookup(f"GREEK {case} LETTER {match.group(1).upper()}")
    except KeyError:
        return match.group(0)


class Fraction:
    "A wrapper around Fraction that returns the original number if the error is too large."

    def __init__(self, number):
        fraction = fractions.Fraction(number).limit_denominator(20)
        if abs(float(fraction) - number) > 1e-5:
            self.value = number
        else:
            self.value = fraction

    def __str__(self):
        if self.is_fraction():
            return str(self.value)
        else:
            return f"{self.value:.3f}"

    def is_fraction(self):
        return isinstance(self.value, fractions.Fraction)

    def latex(self):
        if not self.is_fraction():
            return f"{self.value:.3f}"
        if self.value.denominator == 1:
            return str(self.value.numerator)
        return f"\\frac{{{self.value.numerator}}}{{{self.value.denominator}}}"

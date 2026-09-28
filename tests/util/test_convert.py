# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import fractions

import numpy as np
import pytest

from py4vasp._util import convert
from py4vasp._util.convert import (
    Fraction,
    math_to_unicode,
    text_to_string,
    to_camelcase,
    to_complex,
)


def test_text_to_string():
    assert text_to_string(b"foo") == "foo"
    assert text_to_string("bar") == "bar"


def test_scalar_to_complex(Assert):
    scalar = np.array((0.0, 1.0))
    converted = to_complex(scalar)
    assert converted.shape == ()
    Assert.allclose(converted.real, scalar[0])
    Assert.allclose(converted.imag, scalar[1])


def test_vector_to_complex(Assert):
    vector = np.linspace(0, 9, 10).reshape(5, 2)
    converted = to_complex(vector)
    assert converted.shape == (5,)
    Assert.allclose(converted.real, vector[:, 0])
    Assert.allclose(converted.imag, vector[:, 1])


def test_matrix_to_complex(Assert):
    matrix = np.linspace(0, 29, 30).reshape(3, 5, 2)
    converted = to_complex(matrix)
    assert converted.shape == (3, 5)
    Assert.allclose(converted.real, matrix[:, :, 0])
    Assert.allclose(converted.imag, matrix[:, :, 1])


def test_camelcase():
    assert to_camelcase("foo") == "Foo"
    assert to_camelcase("foo_bar") == "FooBar"
    assert to_camelcase("foo_bar_baz") == "FooBarBaz"
    assert to_camelcase("_foo") == "Foo"
    assert to_camelcase("_foo_bar") == "FooBar"
    assert to_camelcase("foo_bar", uppercase_first_letter=False) == "fooBar"


@pytest.mark.parametrize(
    "number, expected, string, latex",
    [
        (0, fractions.Fraction(0), "0", "0"),
        (0.5, fractions.Fraction(1, 2), "1/2", "\\frac{1}{2}"),
        (0.3333333333333333, fractions.Fraction(1, 3), "1/3", "\\frac{1}{3}"),
        (0.25, fractions.Fraction(1, 4), "1/4", "\\frac{1}{4}"),
        (0.2, fractions.Fraction(1, 5), "1/5", "\\frac{1}{5}"),
        (0.125, fractions.Fraction(1, 8), "1/8", "\\frac{1}{8}"),
        (np.sqrt(0.5), np.sqrt(0.5), f"0.707", f"0.707"),
    ],
)
def test_Fraction(number, expected, string, latex):
    fraction = Fraction(number)
    assert fraction.is_fraction() == isinstance(expected, fractions.Fraction)
    assert fraction.value == expected
    assert str(fraction) == string
    assert fraction.latex() == latex


@pytest.mark.parametrize(
    "label, expected",
    [
        (r"$\Gamma$", "Γ"),
        (r"$\Sigma$", "Σ"),
        (r"$\alpha$", "α"),
        ("X", "X"),  # a label without math is passed through
        (r"M|$\Gamma$", "M|Γ"),  # the corner of a discontinuous path
        (r"$[\frac{1}{2} 0 0]$", "[1/2 0 0]"),  # generated for unlabeled k points
        (r"$[0 0 0]$", "[0 0 0]"),
        (r"$\overline{\Gamma}$", "Γ\u0305"),  # the surface Brillouin zone
        (r"$\bar{K}$", "K\u0305"),
        (r"$\sqrt{2}$", r"\sqrt{2}"),  # unknown commands lose only the math delimiters
    ],
)
def test_math_to_unicode(label, expected):
    assert math_to_unicode(label) == expected


def test_energy_conversion_factors():
    # VASP's own values, so that a table py4vasp prints still matches the OUTCAR it
    # came from. backlog/public-unit-constants.md tracks the CODATA discrepancy.
    assert convert.EV_TO_THZ == 241.798934781
    assert convert.EV_TO_CM1 == 8065.610420
    assert convert.EV_TO_MEV == 1000


def test_energy_conversion_factors_are_consistent():
    # a frequency of 1 THz corresponds to 1e12 / c wavenumbers, which ties the two
    # factors together and catches a mistyped digit in either of them. VASP's values
    # predate the current CODATA ones, so they agree only to 8e-6 -- see
    # backlog/public-unit-constants.md for why that spread is tolerated here.
    speed_of_light_in_cm_per_s = 2.99792458e10
    wavenumbers_per_THz = 1e12 / speed_of_light_in_cm_per_s
    ratio = convert.EV_TO_CM1 / convert.EV_TO_THZ
    assert np.isclose(ratio, wavenumbers_per_THz, rtol=1e-5)

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp._config import VASP_COLORS
from py4vasp._util.convert import (
    text_to_string,
    to_camelcase,
    to_complex,
    to_lab,
    to_rgb,
)


@pytest.fixture
def reference_colors():
    return ("#4C265F", "#2FB5AB", "#2C68FC", "#A82C35", "#808080", "#212529")


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


def test_hex_to_rgb(reference_colors, Assert):
    converted = [
        [76, 38, 95],
        [47, 181, 171],
        [44, 104, 252],
        [168, 44, 53],
        [128, 128, 128],
        [33, 37, 41],
    ]
    expected = np.array(converted) / 255
    actual = np.array([to_rgb(color) for color in reference_colors])
    Assert.allclose(expected, actual)


def test_hex_to_lab(reference_colors, Assert):
    expected = [
        (22.824, 28.808, -26.898),
        (66.968, -37.078, -5.109),
        (48.836, 34.588, -78.789),
        (38.526, 50.465, 25.169),
        (53.585, -0.002, -0.000),
        (14.437, -0.722, -3.262),
    ]
    actual = np.array([np.round(to_lab(color), 3) for color in reference_colors])
    Assert.allclose(expected, actual)


def test_camelcase():
    assert to_camelcase("foo") == "Foo"
    assert to_camelcase("foo_bar") == "FooBar"
    assert to_camelcase("foo_bar_baz") == "FooBarBaz"
    assert to_camelcase("_foo") == "Foo"
    assert to_camelcase("_foo_bar") == "FooBar"
    assert to_camelcase("foo_bar", uppercase_first_letter=False) == "fooBar"

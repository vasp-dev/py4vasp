# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._util.color import Color

REFERENCE_COLORS = ("#4c265f", "#2fb5ab", "#2c68fc", "#a82c35", "#808080", "#212529")
REFERENCE_RGB = np.array(
    [
        [76, 38, 95],
        [47, 181, 171],
        [44, 104, 252],
        [168, 44, 53],
        [128, 128, 128],
        [33, 37, 41],
    ]
) / 255
REFERENCE_LAB = [
    (22.824, 28.808, -26.898),
    (66.968, -37.078, -5.109),
    (48.836, 34.588, -78.789),
    (38.526, 50.465, 25.169),
    (53.585, -0.002, -0.000),
    (14.437, -0.722, -3.262),
]


def test_from_hex(Assert):
    for hex_code, expected in zip(REFERENCE_COLORS, REFERENCE_RGB):
        Assert.allclose(Color(hex_code).rgb, expected)


def test_hex_is_case_insensitive(Assert):
    Assert.allclose(Color("#2FB5AB").rgb, Color("#2fb5ab").rgb)


def test_hex_without_hash(Assert):
    Assert.allclose(Color("2fb5ab").rgb, Color("#2fb5ab").rgb)


def test_from_rgb():
    color = Color([1.0, 0.0, 0.0])
    assert color.rgb == (1.0, 0.0, 0.0)
    assert color.hex == "#ff0000"


def test_hex_roundtrip():
    for hex_code in REFERENCE_COLORS:
        assert Color(hex_code).hex == hex_code


def test_hex_clips_out_of_range():
    assert Color([2.0, -1.0, 0.5]).hex == "#ff0080"


def test_to_lab(Assert):
    for hex_code, expected in zip(REFERENCE_COLORS, REFERENCE_LAB):
        Assert.allclose(np.round(Color(hex_code).to_lab(), 3), np.array(expected))


def test_label():
    assert Color("#2fb5ab").label == ""
    assert Color("#2fb5ab", label="teal").label == "teal"


def test_str():
    text = str(Color("#2fb5ab", label="teal"))
    assert "teal" in text
    assert "#2fb5ab" in text
    text_without_label = str(Color("#2fb5ab"))
    assert text_without_label.startswith("#2fb5ab")


def test_repr_html_contains_information():
    html = Color("#2fb5ab", label="teal")._repr_html_()
    assert "teal" in html
    assert "#2fb5ab" in html
    assert "sRGB" in html
    assert "background:#2fb5ab" in html


def test_repr_html_text_color_is_readable():
    # dark background -> white text, light background -> black text
    assert "color:#ffffff" in Color("#000000")._repr_html_()
    assert "color:#000000" in Color("#ffffff")._repr_html_()


def test_invalid_hex_raises_error():
    with pytest.raises(exception.IncorrectUsage):
        Color("#12")
    with pytest.raises(exception.IncorrectUsage):
        Color("#gggggg")


def test_invalid_rgb_length_raises_error():
    with pytest.raises(exception.IncorrectUsage):
        Color([1.0, 0.0])


def test_color_is_frozen():
    color = Color("#2fb5ab")
    with pytest.raises(dataclasses.FrozenInstanceError):
        color.label = "new"


def test_color_equality_and_hash():
    assert Color("#2fb5ab", label="teal") == Color("#2fb5ab", label="teal")
    assert Color("#2fb5ab") != Color("#2fb5ab", label="teal")
    assert len({Color("#2fb5ab"), Color("#2fb5ab")}) == 1

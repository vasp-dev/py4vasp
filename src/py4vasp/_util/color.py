# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""A small color helper used across py4vasp.

The :class:`Color` dataclass stores a color as fractional sRGB and can be created either
from an HTML/HEX string or from an sRGB sequence. It knows how to convert to a HEX code
and to the CIELAB color space, and it renders as a labeled swatch in Jupyter.
"""

import dataclasses

import numpy as np

from py4vasp import exception

# sRGB (D65) to XYZ matrix used for the CIELAB conversion.
_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
# reference white point (D65) for the CIELAB conversion
_WHITE_X = 0.950489
_WHITE_Z = 1.088840
# WCAG crossover luminance: above it black text has more contrast, below it white text.
_CONTRAST_THRESHOLD = 0.179


@dataclasses.dataclass(frozen=True, init=False, repr=False)
class Color:
    """A color stored as fractional sRGB with an optional label.

    Construct it either from a sequence of three fractional sRGB values in [0, 1] or,
    using :meth:`from_hex`, from an HTML/HEX color code.

    In a Jupyter notebook a color renders as a labeled swatch; the text on the swatch is
    automatically black or white depending on the background so that it stays readable.

    Parameters
    ----------
    rgb : sequence of float
        The three fractional sRGB values in [0, 1].
    label : str
        An optional label describing the color.

    Examples
    --------
    Create a color from fractional sRGB values or from an HTML/HEX code

    >>> from py4vasp._util.color import Color
    >>> Color([0.18, 0.71, 0.67])
    Color((0.18, 0.71, 0.67))
    >>> Color.from_hex("#2fb5ab").hex
    '#2fb5ab'

    Attach or replace a label; the original color is left unchanged because a color is
    immutable

    >>> teal = Color.from_hex("#2fb5ab", label="teal")
    >>> teal.label()
    'teal'
    >>> teal.label("ocean").label()
    'ocean'
    """

    rgb: tuple
    "The color as a tuple of three fractional sRGB values in [0, 1]."
    _label: str = ""

    def __init__(self, rgb, label=""):
        object.__setattr__(self, "rgb", _to_rgb(rgb))
        object.__setattr__(self, "_label", label)

    @classmethod
    def from_hex(cls, hex_code, label="") -> "Color":
        "Create a color from an HTML/HEX color code such as ``'#2fb5ab'``."
        return cls(_hex_to_rgb(hex_code), label)

    def label(self, new_label=None):
        "Return the label, or if *new_label* is given a copy of the color relabeled."
        if new_label is None:
            return self._label
        return Color(self.rgb, new_label)

    @property
    def hex(self) -> str:
        "The color as an HTML/HEX code, e.g. ``'#2fb5ab'``."
        channels = (int(round(255 * min(max(c, 0), 1))) for c in self.rgb)
        return "#{:02x}{:02x}{:02x}".format(*channels)

    def to_lab(self) -> np.ndarray:
        "Convert the color to the perceptually uniform CIELAB color space."
        linear = _linearize(np.array(self.rgb))
        x, y, z = _RGB_TO_XYZ @ linear
        threshold = (6 / 29) ** 3
        f = lambda t: t ** (1 / 3) if t > threshold else 2 / 29 * (t / threshold + 2)
        lightness = 116 * f(y) - 16
        a = 500 * (f(x / _WHITE_X) - f(y))
        b = 200 * (f(y) - f(z / _WHITE_Z))
        return np.array((lightness, a, b))

    def __repr__(self) -> str:
        if self._label:
            return f"Color({self.rgb!r}, label={self._label!r})"
        return f"Color({self.rgb!r})"

    def __str__(self) -> str:
        rgb = ", ".join(f"{c:.3f}" for c in self.rgb)
        prefix = f"{self._label}: " if self._label else ""
        return f"{prefix}{self.hex} (sRGB: {rgb})"

    def _repr_html_(self) -> str:
        lines = []
        if self._label:
            lines.append(f"<strong>{self._label}</strong>")
        lines.append(self.hex)
        lines.append("sRGB " + ", ".join(f"{c:.2f}" for c in self.rgb))
        content = "<br>".join(lines)
        return (
            f'<div style="display:inline-flex;align-items:center;'
            f"justify-content:center;width:200px;height:100px;margin:4px;"
            f"background:{self.hex};color:{self._readable_text_color()};"
            f"border:1px solid rgba(128,128,128,0.4);border-radius:6px;"
            f'font-family:sans-serif;font-size:13px;text-align:center;line-height:1.5">'
            f"<div>{content}</div></div>"
        )

    def _readable_text_color(self) -> str:
        # Pick black or white text depending on the background luminance so that the
        # text always stays readable (WCAG relative luminance and crossover threshold).
        red, green, blue = _linearize(np.array(self.rgb))
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        return "#000000" if luminance > _CONTRAST_THRESHOLD else "#ffffff"


def _to_rgb(rgb):
    rgb = tuple(float(channel) for channel in rgb)
    if len(rgb) != 3:
        message = f"A color needs exactly three sRGB values but got {len(rgb)}."
        raise exception.IncorrectUsage(message)
    return rgb


def _hex_to_rgb(hex_code):
    digits = hex_code.lstrip("#")
    if len(digits) != 6:
        message = f'"{hex_code}" is not a valid HTML color code (expected e.g. "#2fb5ab").'
        raise exception.IncorrectUsage(message)
    try:
        channels = [int(digits[i : i + 2], 16) for i in (0, 2, 4)]
    except ValueError:
        message = f'"{hex_code}" is not a valid HTML color code (expected e.g. "#2fb5ab").'
        raise exception.IncorrectUsage(message) from None
    return tuple(channel / 255 for channel in channels)


def _linearize(rgb):
    "Undo the sRGB gamma correction to obtain linear intensities."
    return np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

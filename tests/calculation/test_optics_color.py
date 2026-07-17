# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp import exception
from py4vasp._calculation import _optics_color as color

WAVELENGTHS = np.linspace(380, 780, 401)


def test_list_color_matching_functions():
    assert color.list_color_matching_functions() == ["1931_2", "1964_10"]


def test_list_illuminants():
    illuminants = color.list_illuminants()
    assert "D65" in illuminants
    assert "E" in illuminants
    assert set(illuminants) == {
        "A",
        "C",
        "D50",
        "D55",
        "D65",
        "D75",
        "E",
        "FL1",
        "FL2",
        "FL3",
        "FL4",
        "FL5",
        "FL6",
    }


def test_xyz_to_srgb_black():
    assert np.array_equal(color.xyz_to_srgb(0, 0, 0), [0, 0, 0])


def test_xyz_to_srgb_primary_green():
    # X=0, Y=1, Z=0 lies outside the sRGB gamut and clips to pure green
    np.testing.assert_allclose(color.xyz_to_srgb(0, 1, 0), [0, 1, 0])


def test_black_spectrum_is_black():
    pytest.importorskip("scipy")
    rgb = color.spectrum_to_rgb(WAVELENGTHS, np.zeros_like(WAVELENGTHS))
    assert np.array_equal(rgb, [0, 0, 0])


def test_spectrum_in_unit_range_gives_valid_rgb():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(0)
    spectrum = rng.uniform(0, 1, WAVELENGTHS.shape)
    rgb = color.spectrum_to_rgb(WAVELENGTHS, spectrum)
    assert rgb.shape == (3,)
    assert np.all(rgb >= 0) and np.all(rgb <= 1)


@pytest.mark.parametrize("peak, channel", [(615, 0), (530, 1), (465, 2)])
def test_monochromatic_spectrum_has_expected_hue(peak, channel):
    pytest.importorskip("scipy")
    # A spectrum peaked in the red/green/blue band lights up the R/G/B channel most.
    spectrum = np.exp(-((WAVELENGTHS - peak) ** 2) / (2 * 10**2))
    rgb = color.spectrum_to_rgb(WAVELENGTHS, spectrum, illuminant="E")
    assert np.argmax(rgb) == channel


def test_unknown_illuminant_raises_error():
    pytest.importorskip("scipy")
    # the illuminant is validated only after the spectrum is interpolated with scipy,
    # so this path requires the full (not core) installation
    with pytest.raises(exception.IncorrectUsage):
        color.spectrum_to_rgb(WAVELENGTHS, np.ones_like(WAVELENGTHS), illuminant="bad")


def test_unknown_cmf_raises_error():
    with pytest.raises(exception.IncorrectUsage):
        color.spectrum_to_rgb(WAVELENGTHS, np.ones_like(WAVELENGTHS), cmf="bad")

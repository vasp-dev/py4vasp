# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp import exception, interpolate
from py4vasp._third_party import numeric


def test_analytic_continuation_for_lorentzian(Assert):
    pytest.importorskip("scipy")
    z_in = 1j * np.array([0.1, 1.0, 10.0])
    f_in = lorentzian(z_in)
    z_out = np.linspace(0.0, 2.5, 6)
    f_out = numeric.analytic_continuation(z_in, f_in, z_out)
    f_expected = lorentzian(z_out)
    Assert.allclose(f_out, f_expected)


def lorentzian(z):
    z0 = 1.0
    gamma = 0.5
    return 1 / (z - z0 + 1j * gamma)


def test_analytic_continuation_for_higher_dimensions(Assert):
    pytest.importorskip("scipy")
    z_in = np.random.rand(3)
    f_in = np.random.rand(5, 4, 3)
    z_out = z_in
    f_out = numeric.analytic_continuation(z_in, f_in, z_out)
    Assert.allclose(f_out, f_in, tolerance=10)


def test_pass_parameters_to_analytic_continuation(Assert):
    pytest.importorskip("scipy")
    config = interpolate.AAAConfig(
        rtol=1e-5, max_terms=50, clean_up=False, clean_up_tol=1e-6
    )
    with patch("scipy.interpolate.AAA") as AAAMock:
        z_in = np.random.rand(3)
        f_in = np.random.rand(3)
        z_out = np.random.rand(3)
        f_expected = AAAMock.return_value.return_value = np.random.rand(3)
        f_out = numeric.analytic_continuation(z_in, f_in, z_out, config=config)
        AAAMock.assert_called_once()
        assert AAAMock.call_args.kwargs == {
            "rtol": config.rtol,
            "max_terms": config.max_terms,
            "clean_up": config.clean_up,
            "clean_up_tol": config.clean_up_tol,
        }
        Assert.allclose(f_out, f_expected)


def test_interpolate_with_function(Assert):
    pytest.importorskip("scipy")
    x_in = np.array([0.1, 0.5, 1.0, 2.0])
    amplitude = 3.0
    stddev = 0.5
    y_in = gaussian(x_in, amplitude=amplitude, stddev=stddev)
    x_out = np.linspace(0.0, 3.0, 12)
    y_out = numeric.interpolate_with_function(gaussian, x_in, y_in, x_out)
    y_expected = gaussian(x_out, amplitude=amplitude, stddev=stddev)
    Assert.allclose(y_out, y_expected, tolerance=1e6)


def gaussian(x, amplitude=1.0, mean=0.0, stddev=1.0):
    x = np.tile(x, np.shape(amplitude) + (1,))
    amplitude = amplitude[..., np.newaxis] if np.ndim(amplitude) else amplitude
    mean = mean[..., np.newaxis] if np.ndim(mean) else mean
    stddev = stddev[..., np.newaxis] if np.ndim(stddev) else stddev
    coeff = amplitude / (stddev * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mean) / stddev) ** 2
    return coeff * np.exp(exponent)


def test_interpolate_with_function_higher_dimensions(Assert):
    pytest.importorskip("scipy")
    x_in = np.random.rand(5)
    amplitude = np.random.rand(3, 2)
    mean = np.random.rand(3, 2)
    y_in = gaussian(x_in, amplitude=amplitude, mean=mean)
    x_out = x_in
    y_out = numeric.interpolate_with_function(gaussian, x_in, y_in, x_out)
    Assert.allclose(y_out, y_in)
    Assert.allclose(y_out, y_in)


# a mesh wide and fine enough that the trapezoidal rule resolves either line shape; the
# Lorentzian needs the width because its tails decay only algebraically
OFFSETS = np.linspace(-500, 500, 2_000_001)
FWHM = 1.5


@pytest.fixture(params=["Gaussian", "Lorentzian"])
def line_shape_class(request):
    # resolved when the test runs rather than when it is collected, so a missing class
    # fails this test instead of the whole module
    return getattr(numeric, request.param)


@pytest.fixture
def line_shape(line_shape_class):
    return line_shape_class(fwhm=FWHM)


# How much of a unit-area line actually lies on OFFSETS, from the antiderivative of each
# shape. The Gaussian is one to machine precision; the Lorentzian is measurably less,
# because its tails decay only as 1/x^2 and no finite mesh holds all of its weight.
_MASS_ON_MESH = {
    "Gaussian": 1.0,
    "Lorentzian": 2 / np.pi * np.arctan(OFFSETS[-1] / (0.5 * FWHM)),
}


def test_line_shape_is_normalized_to_unit_area(line_shape, Assert):
    mass = np.trapezoid(line_shape.profile(OFFSETS), OFFSETS)
    Assert.allclose(mass, _MASS_ON_MESH[type(line_shape).__name__])


def test_fwhm_is_the_width_at_half_maximum(line_shape, Assert):
    # the assertion that makes the name of the parameter true
    Assert.allclose(line_shape.profile(0.5 * FWHM), 0.5 * line_shape.profile(0.0))


def test_line_shape_keeps_the_shape_of_the_offsets(line_shape):
    assert line_shape.profile(np.zeros((4, 3))).shape == (4, 3)


def test_gaussian_accepts_sigma_or_fwhm(Assert):
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    Assert.allclose(
        numeric.Gaussian(sigma=sigma).profile(OFFSETS),
        numeric.Gaussian(fwhm=FWHM).profile(OFFSETS),
    )
    Assert.allclose(numeric.Gaussian(fwhm=FWHM).sigma, sigma)
    Assert.allclose(numeric.Gaussian(sigma=sigma).fwhm, FWHM)


def test_lorentzian_accepts_gamma_or_fwhm(Assert):
    # gamma is the half width at half maximum, the parameter of 1 / (x - x0 + i gamma)
    Assert.allclose(
        numeric.Lorentzian(gamma=0.5 * FWHM).profile(OFFSETS),
        numeric.Lorentzian(fwhm=FWHM).profile(OFFSETS),
    )
    Assert.allclose(numeric.Lorentzian(fwhm=FWHM).gamma, 0.5 * FWHM)
    Assert.allclose(numeric.Lorentzian(gamma=0.5 * FWHM).fwhm, FWHM)


def test_line_shape_takes_exactly_one_width(line_shape_class):
    alias = "sigma" if line_shape_class is numeric.Gaussian else "gamma"
    with pytest.raises(exception.IncorrectUsage):
        line_shape_class()
    with pytest.raises(exception.IncorrectUsage):
        line_shape_class(**{"fwhm": FWHM, alias: 0.5 * FWHM})


@pytest.mark.parametrize("width", [0.0, -1.0, np.array([1.0, -1.0])])
def test_line_shape_rejects_a_width_that_is_not_positive(line_shape_class, width):
    with pytest.raises(exception.IncorrectUsage):
        line_shape_class(fwhm=width)


def test_line_shape_must_be_given_by_keyword(line_shape_class):
    # a bare number cannot say whether it is a FWHM or a standard deviation
    with pytest.raises(TypeError):
        line_shape_class(FWHM)


MESH = np.linspace(-10, 10, 2001)


def test_broaden_conserves_the_total_weight(Assert):
    positions = np.array([-2.0, 0.5, 3.0])
    weights = np.array([1.0, 2.5, 0.5])
    spectrum = numeric.broaden(
        MESH, positions, weights, shape=numeric.Gaussian(fwhm=FWHM)
    )
    Assert.allclose(np.trapezoid(spectrum, MESH), np.sum(weights))


def test_broaden_defaults_to_one_per_position(Assert):
    positions = np.array([-2.0, 0.5, 3.0])
    shape = numeric.Gaussian(fwhm=FWHM)
    Assert.allclose(
        numeric.broaden(MESH, positions, shape=shape),
        numeric.broaden(MESH, positions, np.ones(3), shape=shape),
    )


def test_broaden_broadcasts_scalar_weights(Assert):
    positions = np.array([-2.0, 0.5, 3.0])
    shape = numeric.Gaussian(fwhm=FWHM)
    Assert.allclose(
        numeric.broaden(MESH, positions, 0.25, shape=shape),
        0.25 * numeric.broaden(MESH, positions, shape=shape),
    )


def test_broaden_reduces_only_the_last_axis(Assert):
    positions = np.random.rand(3, 2, 5)
    shape = numeric.Lorentzian(fwhm=FWHM)
    spectrum = numeric.broaden(MESH, positions, shape=shape)
    assert spectrum.shape == (3, 2, len(MESH))
    for i in range(3):
        for j in range(2):
            Assert.allclose(
                spectrum[i, j], numeric.broaden(MESH, positions[i, j], shape=shape)
            )


def test_broaden_is_the_sum_of_its_peaks(Assert):
    positions = np.array([-2.0, 0.5])
    shape = numeric.Lorentzian(fwhm=FWHM)
    separate = [numeric.broaden(MESH, position, shape=shape) for position in positions]
    Assert.allclose(numeric.broaden(MESH, positions, shape=shape), sum(separate))


def test_broaden_accepts_one_width_per_position(Assert):
    # a spectral function broadens every peak with its own |Im Sigma|, so the width has
    # to broadcast against the positions rather than be a single number
    positions = np.array([-2.0, 0.5])
    widths = np.array([0.5, 2.0])
    spectrum = numeric.broaden(MESH, positions, shape=numeric.Lorentzian(fwhm=widths))
    separate = [
        numeric.broaden(MESH, position, shape=numeric.Lorentzian(fwhm=width))
        for position, width in zip(positions, widths)
    ]
    Assert.allclose(spectrum, sum(separate))


def test_broaden_matches_an_explicit_gaussian_sum(Assert):
    # the closed form written out by hand, which is what the demo data was built with
    # before this helper existed; it pins the numbers independently of the shapes
    levels = np.array([-1.5, 0.0, 2.25])
    weights = np.array([0.5, 1.0, 1.5])
    sigma = 0.15
    distance = (MESH[:, np.newaxis] - levels) / sigma
    expected = np.exp(-0.5 * distance**2) @ weights / (sigma * np.sqrt(2 * np.pi))
    Assert.allclose(
        numeric.broaden(MESH, levels, weights, shape=numeric.Gaussian(sigma=sigma)),
        expected,
    )


def test_broaden_peaks_at_the_position(Assert):
    # migrated from the demo helper this replaces
    position = 1.5
    spectrum = numeric.broaden(MESH, [position], shape=numeric.Gaussian(fwhm=FWHM))
    spacing = MESH[1] - MESH[0]
    assert spectrum.shape == MESH.shape
    assert np.all(spectrum >= 0)
    assert abs(MESH[np.argmax(spectrum)] - position) < spacing


def test_broaden_width_controls_the_peak_height():
    # migrated from the demo helper this replaces; unit area means a narrower line has
    # to be taller
    sharp = numeric.broaden(MESH, [0.0], shape=numeric.Gaussian(fwhm=0.05))
    broad = numeric.broaden(MESH, [0.0], shape=numeric.Gaussian(fwhm=0.5))
    assert sharp.max() > broad.max()

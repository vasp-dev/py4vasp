# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from unittest.mock import patch

import numpy as np

from py4vasp import interpolate
from py4vasp._third_party import numeric


def test_analytic_continuation_for_lorentzian(not_core, Assert):
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


def test_analytic_continuation_for_higher_dimensions(not_core, Assert):
    z_in = np.random.rand(3)
    f_in = np.random.rand(5, 4, 3)
    z_out = z_in
    f_out = numeric.analytic_continuation(z_in, f_in, z_out)
    Assert.allclose(f_out, f_in, tolerance=10)


def test_pass_parameters_to_analytic_continuation(not_core, Assert):
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


def test_interpolate_with_function(not_core, Assert):
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


def test_interpolate_with_function_higher_dimensions(not_core, Assert):
    x_in = np.random.rand(5)
    amplitude = np.random.rand(3, 2)
    mean = np.random.rand(3, 2)
    y_in = gaussian(x_in, amplitude=amplitude, mean=mean)
    x_out = x_in
    y_out = numeric.interpolate_with_function(gaussian, x_in, y_in, x_out)
    Assert.allclose(y_out, y_in)
    Assert.allclose(y_out, y_in)

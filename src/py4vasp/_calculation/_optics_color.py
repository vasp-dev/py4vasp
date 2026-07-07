# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Convert an optical spectrum into a perceived sRGB color.

This implements the color-matching pipeline: a reflectivity or transmission spectrum is
weighted by a standard illuminant and integrated against the CIE color matching
functions to obtain XYZ tristimulus values, which are then converted to sRGB. The color
matching functions use the analytic Gaussian approximation of Wyman, Sloan, and Shirley
(2013); the illuminants are the standard CIE spectral power distributions.
"""

import numpy as np

from py4vasp import exception
from py4vasp._util import import_
from py4vasp._util.suggest import did_you_mean

# scipy is only required for the full (not core) installation, so import it lazily; the
# color-matching routines below raise exception.ModuleNotInstalled if it is missing.
interpolate = import_.optional("scipy.interpolate")

# Wavelength grid on which the color matching and illuminant spectra are evaluated.
_WAVELENGTHS = np.linspace(380, 780, 401)  # nm, 1 nm resolution

# XYZ (D65) to linear sRGB transformation matrix.
_XYZ_TO_RGB = np.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ]
)


def list_color_matching_functions():
    """Return the names of the available color matching functions."""
    return list(_COLOR_MATCHING_FUNCTIONS)


def list_illuminants():
    """Return the names of the available standard illuminants."""
    return list(_ILLUMINANTS)


def color_matching_function(name="1931_2"):
    """Return ``(wavelengths, x_bar, y_bar, z_bar)`` for the requested CIE observer."""
    builder = _select(name, _COLOR_MATCHING_FUNCTIONS, "color matching function")
    x_bar, y_bar, z_bar = builder(_WAVELENGTHS)
    return _WAVELENGTHS, x_bar, y_bar, z_bar


def illuminant_spectrum(name="D65", wavelengths=None):
    """Return the relative spectral power distribution of a standard illuminant.

    The tabulated data is interpolated onto *wavelengths* (defaults to the internal grid).
    """
    if wavelengths is None:
        wavelengths = _WAVELENGTHS
    wl_data, power_data = _select(name, _ILLUMINANTS, "illuminant")()
    power = interpolate.interp1d(
        wl_data, power_data, kind="cubic", bounds_error=False, fill_value="extrapolate"
    )(wavelengths)
    return np.maximum(power, 0)


def xyz_to_srgb(X, Y, Z):
    """Convert CIE XYZ tristimulus values to gamma-corrected sRGB in [0, 1]."""
    linear_rgb = _XYZ_TO_RGB @ np.array([X, Y, Z])
    srgb = np.where(
        linear_rgb <= 0.0031308,
        12.92 * linear_rgb,
        1.055 * np.power(np.clip(linear_rgb, 0, None), 1 / 2.4) - 0.055,
    )
    return np.clip(srgb, 0, 1)


def spectrum_to_rgb(wavelengths, spectrum, *, illuminant="D65", cmf="1931_2"):
    """Compute the sRGB color of a material from its optical *spectrum*.

    Parameters
    ----------
    wavelengths : np.ndarray
        Wavelengths (nm) at which *spectrum* is sampled.
    spectrum : np.ndarray
        Reflectivity or transmission spectrum in [0, 1].
    illuminant : str
        Name of the standard illuminant used to light the material.
    cmf : str
        Name of the CIE color matching function (standard observer).

    Returns
    -------
    np.ndarray
        The sRGB color as a length-3 array in [0, 1].
    """
    cmf_wl, x_bar, y_bar, z_bar = color_matching_function(cmf)
    spectrum_on_grid = interpolate.interp1d(
        wavelengths, spectrum, kind="linear", bounds_error=False, fill_value=0.0
    )(cmf_wl)
    illuminant_on_grid = illuminant_spectrum(illuminant, cmf_wl)
    weighted = spectrum_on_grid * illuminant_on_grid
    step = cmf_wl[1] - cmf_wl[0]
    XYZ = np.array([np.sum(weighted * bar) * step for bar in (x_bar, y_bar, z_bar)])
    total = np.sum(XYZ)
    if total > 0:
        XYZ = XYZ / total
    return xyz_to_srgb(*XYZ)


def _select(name, options, kind):
    try:
        return options[name]
    except KeyError:
        available = '", "'.join(options)
        message = (
            f'Could not find the {kind} "{name}". {did_you_mean(name, options)}'
            f'py4vasp supports the following {kind}s: "{available}".'
        )
        raise exception.IncorrectUsage(message) from None


def _gaussian(wavelengths, mu, tau_left, tau_right):
    """Piecewise Gaussian used to approximate the color matching functions."""
    tau = np.where(wavelengths < mu, tau_left, tau_right)
    return np.exp(-(tau**2) * (wavelengths - mu) ** 2 / 2)


def _cmf_1931_2(wavelengths):
    x_bar = (
        1.056 * _gaussian(wavelengths, 599.8, 0.0264, 0.0323)
        + 0.362 * _gaussian(wavelengths, 442.0, 0.0624, 0.0374)
        - 0.065 * _gaussian(wavelengths, 501.1, 0.0490, 0.0382)
    )
    y_bar = 0.821 * _gaussian(wavelengths, 568.8, 0.0213, 0.0247) + 0.286 * _gaussian(
        wavelengths, 530.9, 0.0613, 0.0322
    )
    z_bar = 1.217 * _gaussian(wavelengths, 437.0, 0.0845, 0.0278) + 0.681 * _gaussian(
        wavelengths, 459.0, 0.0385, 0.0725
    )
    return x_bar, y_bar, z_bar


def _cmf_1964_10(wavelengths):
    x_bar = (
        1.074 * _gaussian(wavelengths, 603.0, 0.0258, 0.0331)
        + 0.389 * _gaussian(wavelengths, 448.0, 0.0651, 0.0402)
        - 0.079 * _gaussian(wavelengths, 507.0, 0.0496, 0.0381)
    )
    y_bar = 0.844 * _gaussian(wavelengths, 575.0, 0.0209, 0.0251) + 0.301 * _gaussian(
        wavelengths, 542.0, 0.0612, 0.0313
    )
    z_bar = 1.158 * _gaussian(wavelengths, 441.0, 0.0861, 0.0274) + 0.708 * _gaussian(
        wavelengths, 465.0, 0.0380, 0.0718
    )
    return x_bar, y_bar, z_bar


_COLOR_MATCHING_FUNCTIONS = {
    "1931_2": _cmf_1931_2,
    "1964_10": _cmf_1964_10,
}

# Coarse wavelength grid shared by the tabulated illuminants (nm).
_ILLUMINANT_GRID = np.arange(380, 781, 20)


def _tabulated_illuminant(power):
    return lambda: (_ILLUMINANT_GRID, np.array(power))


def _illuminant_A():
    # Incandescent tungsten lamp (2856 K), approximated by Planck's blackbody law.
    wavelengths = np.linspace(380, 780, 100)
    temperature = 2856
    planck = 6.626e-34
    speed_of_light = 3e8
    boltzmann = 1.381e-23
    meters = wavelengths * 1e-9
    power = (1.0 / meters**5) / (
        np.exp(planck * speed_of_light / (meters * boltzmann * temperature)) - 1
    )
    return wavelengths, power / np.max(power)


def _illuminant_E():
    # Equal energy illuminant (flat spectrum).
    wavelengths = np.linspace(380, 780, 100)
    return wavelengths, np.ones_like(wavelengths)


_ILLUMINANTS = {
    "A": _illuminant_A,
    "C": _tabulated_illuminant(
        [0.04, 0.08, 0.15, 0.28, 0.45, 0.65, 0.82, 0.95, 1.05, 1.10, 1.12,
         1.10, 1.08, 1.05, 1.00, 0.93, 0.85, 0.76, 0.65, 0.53, 0.40]
    ),
    "D50": _tabulated_illuminant(
        [0.02, 0.04, 0.08, 0.16, 0.28, 0.43, 0.62, 0.80, 0.94, 1.05, 1.12,
         1.15, 1.13, 1.10, 1.05, 0.97, 0.88, 0.77, 0.65, 0.50, 0.35]
    ),
    "D55": _tabulated_illuminant(
        [0.03, 0.05, 0.10, 0.19, 0.32, 0.48, 0.68, 0.85, 0.98, 1.07, 1.13,
         1.15, 1.12, 1.08, 1.03, 0.95, 0.86, 0.75, 0.63, 0.48, 0.33]
    ),
    "D65": _tabulated_illuminant(
        [0.04, 0.06, 0.12, 0.22, 0.35, 0.46, 0.57, 0.68, 0.77, 0.86, 0.94,
         1.00, 1.04, 1.08, 1.10, 1.10, 1.09, 1.08, 1.06, 1.04, 1.02]
    ),
    "D75": _tabulated_illuminant(
        [0.05, 0.08, 0.15, 0.26, 0.40, 0.52, 0.63, 0.73, 0.82, 0.90, 0.96,
         1.01, 1.05, 1.08, 1.10, 1.09, 1.08, 1.06, 1.03, 1.00, 0.98]
    ),
    "E": _illuminant_E,
    "FL1": _tabulated_illuminant(
        [0.02, 0.08, 0.18, 0.32, 0.50, 0.55, 0.60, 0.68, 0.75, 0.88, 1.00,
         1.05, 1.08, 1.10, 1.09, 1.05, 1.00, 0.95, 0.88, 0.80, 0.70]
    ),
    "FL2": _tabulated_illuminant(
        [0.01, 0.05, 0.12, 0.24, 0.42, 0.52, 0.62, 0.72, 0.82, 0.92, 1.00,
         1.03, 1.04, 1.02, 0.98, 0.92, 0.85, 0.76, 0.65, 0.52, 0.40]
    ),
    "FL3": _tabulated_illuminant(
        [0.00, 0.03, 0.08, 0.16, 0.32, 0.48, 0.65, 0.80, 0.92, 1.00, 1.05,
         1.06, 1.04, 1.00, 0.94, 0.86, 0.77, 0.66, 0.54, 0.41, 0.28]
    ),
    "FL4": _tabulated_illuminant(
        [0.00, 0.02, 0.05, 0.10, 0.20, 0.35, 0.52, 0.70, 0.85, 0.96, 1.02,
         1.05, 1.05, 1.02, 0.98, 0.92, 0.85, 0.76, 0.65, 0.52, 0.38]
    ),
    "FL5": _tabulated_illuminant(
        [0.03, 0.07, 0.16, 0.30, 0.48, 0.54, 0.60, 0.67, 0.74, 0.86, 0.98,
         1.03, 1.06, 1.08, 1.07, 1.03, 0.98, 0.92, 0.85, 0.77, 0.68]
    ),
    "FL6": _tabulated_illuminant(
        [0.00, 0.01, 0.04, 0.09, 0.18, 0.33, 0.50, 0.68, 0.83, 0.94, 1.00,
         1.03, 1.03, 1.00, 0.96, 0.90, 0.83, 0.74, 0.63, 0.50, 0.37]
    ),
}

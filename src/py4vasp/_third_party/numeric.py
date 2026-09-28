# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from dataclasses import dataclass
from typing import Optional

import numpy as np

from py4vasp import exception
from py4vasp._util import import_

interpolate = import_.optional("scipy.interpolate")
optimize = import_.optional("scipy.optimize")


@dataclass(kw_only=True)
class AAAConfig:
    rtol: Optional[float] = None
    max_terms: int = 100
    clean_up: bool = True
    clean_up_tol: float = 1e-13


def analytic_continuation(z_in, f_in, z_out, *, config: AAAConfig = AAAConfig()):
    shape = f_in.shape
    data_sets = f_in.reshape((-1, shape[-1]))
    f_out = [
        _analytic_continuation_single(z_in, data_set, z_out, config)
        for data_set in data_sets
    ]
    return np.reshape(f_out, shape[:-1] + (len(z_out),))


def _analytic_continuation_single(z_in, f_in, z_out, config):
    aaa = interpolate.AAA(
        z_in,
        f_in,
        rtol=config.rtol,
        max_terms=config.max_terms,
        clean_up=config.clean_up,
        clean_up_tol=config.clean_up_tol,
    )
    return aaa(z_out)


def interpolate_with_function(function, x_in, y_in, x_out):
    shape = y_in.shape
    data_sets = y_in.reshape((-1, shape[-1]))
    y_out = np.array(
        [
            _interpolate_with_function_single(function, x_in, data_set, x_out)
            for data_set in data_sets
        ]
    )
    return y_out.reshape(shape[:-1] + (len(x_out),))


def _interpolate_with_function_single(function, x_in, y_in, x_out):
    parameters, _ = optimize.curve_fit(function, x_in, y_in)
    return function(x_out, *parameters)


# A line shape can be written down with more than one width parameter and the literature
# uses all of them, so every shape here converts to and from the FWHM. Doing it once
# means a standard deviation cannot reach a call site that expects a full width.
_SIGMA_PER_FWHM = 1 / (2 * np.sqrt(2 * np.log(2)))
_GAMMA_PER_FWHM = 0.5


class _LineShape:
    """Width bookkeeping shared by the normalized line shapes.

    A subclass is a dataclass with two optional fields -- ``fwhm`` and the parameter its
    analytic form is written with -- and connects them with the class attributes
    ``_alias`` and ``_per_fwhm``. Exactly one of the two must be given; the other is
    filled in here, so both are available afterwards whichever one the caller used.
    """

    def __post_init__(self):
        self._raise_error_unless_exactly_one_width_is_given()
        if self.fwhm is None:
            self.fwhm = getattr(self, self._alias) / self._per_fwhm
        else:
            setattr(self, self._alias, self._per_fwhm * self.fwhm)
        self._raise_error_if_width_is_not_positive()

    def _raise_error_unless_exactly_one_width_is_given(self):
        given = [
            name for name in ("fwhm", self._alias) if getattr(self, name) is not None
        ]
        if len(given) == 1:
            return
        problem = "you gave both" if given else "you gave neither"
        raise exception.IncorrectUsage(
            f"Please specify the width of the {type(self).__name__} either as 'fwhm', "
            f"the full width at half maximum, or as '{self._alias}', but {problem}. "
            "The two describe the same line, so giving both cannot be resolved."
        )

    def _raise_error_if_width_is_not_positive(self):
        if np.all(np.asarray(self.fwhm) > 0):
            return
        raise exception.IncorrectUsage(
            f"The width of the {type(self).__name__} must be positive everywhere, but "
            f"'fwhm' is {self.fwhm}. A width of zero describes a delta peak, which no "
            "mesh can represent."
        )


@dataclass(kw_only=True)
class Gaussian(_LineShape):
    """A Gaussian line shape normalized to unit area.

    Broadening with this shape conserves the total weight, so a density of states still
    integrates to the number of states no matter which width is chosen.

    Parameters
    ----------
    fwhm
        Full width at half maximum, the width a spectroscopist quotes. Give this or
        *sigma*, not both. An array is allowed as long as it broadcasts against the
        positions it broadens, which lets every peak have its own width.
    sigma
        Standard deviation of the Gaussian, the width of its analytic form. Give this or
        *fwhm*, not both.
    """

    fwhm: Optional[float] = None
    sigma: Optional[float] = None
    _alias = "sigma"
    _per_fwhm = _SIGMA_PER_FWHM

    def profile(self, offsets):
        """Evaluate the line shape at the given distances from its center.

        Parameters
        ----------
        offsets
            Distance from the center of the peak, in the unit of the width.

        Returns
        -------
        -
            The line shape with the shape of *offsets*, normalized such that it
            integrates to one.
        """
        return np.exp(-0.5 * (offsets / self.sigma) ** 2) / (
            self.sigma * np.sqrt(2 * np.pi)
        )


@dataclass(kw_only=True)
class Lorentzian(_LineShape):
    """A Lorentzian line shape normalized to unit area.

    Broadening with this shape conserves the total weight. Its tails decay only
    algebraically, so a mesh has to reach further than for a Gaussian before the weight
    outside it becomes negligible.

    Parameters
    ----------
    fwhm
        Full width at half maximum, the width a spectroscopist quotes. Give this or
        *gamma*, not both. An array is allowed as long as it broadcasts against the
        positions it broadens, which lets every peak have its own width.
    gamma
        Half width at half maximum, the parameter of the analytic form
        1 / (x - x₀ + iγ). It is half of *fwhm*. Give this or *fwhm*, not both.
    """

    fwhm: Optional[float] = None
    gamma: Optional[float] = None
    _alias = "gamma"
    _per_fwhm = _GAMMA_PER_FWHM

    def profile(self, offsets):
        """Evaluate the line shape at the given distances from its center.

        Parameters
        ----------
        offsets
            Distance from the center of the peak, in the unit of the width.

        Returns
        -------
        -
            The line shape with the shape of *offsets*, normalized such that it
            integrates to one.
        """
        return self.gamma / np.pi / (offsets**2 + self.gamma**2)


def broaden(mesh, positions, weights=None, *, shape):
    """Spread discrete peaks into a smooth spectrum on the given mesh.

    Every position contributes one line of the given shape, scaled by its weight. The
    shapes are normalized to unit area, so the spectrum integrates to the total weight
    as long as the peaks are inside the mesh -- changing the width redistributes the
    spectrum but does not change what it adds up to.

    Nothing here is specific to an energy axis; the mesh, the positions and the width
    only have to share a unit. Neighbour distances in Å broaden the same way.

    Parameters
    ----------
    mesh
        Positions at which the spectrum is evaluated, e.g. an energy axis.
    positions
        Center of every peak, such as eigenvalues or mode frequencies. Only the last
        axis is broadened over; any leading axes are kept, so the bands of a
        calculation can be broadened into one spectrum each in a single call.
    weights
        Contribution of every peak, broadcast against *positions*. Defaults to one per
        peak.
    shape
        The line shape to give every peak, e.g. :class:`Gaussian` or
        :class:`Lorentzian`. Its width may be an array that broadcasts against
        *positions*, which gives every peak its own width.

    Returns
    -------
    -
        The spectrum, with the shape of *positions* except that its last axis is
        replaced by the mesh.

    Notes
    -----
    The line shapes are not truncated, so the intermediate array holds one value per
    mesh point and peak. Broadening very many peaks onto a very fine mesh is therefore
    limited by memory rather than by time.
    """
    mesh = np.atleast_1d(mesh)
    positions = np.atleast_1d(np.asarray(positions, dtype=np.float64))
    weights = np.broadcast_to(1.0 if weights is None else weights, positions.shape)
    # the mesh becomes an axis of its own in front of the peaks, so that the width of
    # the shape broadcasts against the peaks the way the weights do
    offsets = mesh[:, np.newaxis] - positions[..., np.newaxis, :]
    return np.einsum("...mp,...p->...m", shape.profile(offsets), weights)

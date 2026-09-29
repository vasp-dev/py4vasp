# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

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
# float() rather than the numpy scalar, so that a width given as a plain number stays
# one: Gaussian(fwhm=1.0).sigma should print like Lorentzian(fwhm=1.0).gamma does
_SIGMA_PER_FWHM = float(1 / (2 * np.sqrt(2 * np.log(2))))
_GAMMA_PER_FWHM = 0.5


def _numeric_array(values):
    """Return the values as an array of floats, or None if they are not numbers."""
    try:
        array = np.asarray(values)
    except ValueError:
        # a ragged nested sequence, which numpy refuses to turn into an array
        return None
    if array.dtype.kind not in "iuf":
        return None
    return array.astype(np.float64)


def _validated_width(width, shape_name, parameter):
    array = _numeric_array(width)
    if array is None:
        raise exception.IncorrectUsage(
            f"The width of the {shape_name} has to be a number or an array of numbers, "
            f"but '{parameter}' is {width!r}."
        )
    if array.size == 0:
        raise exception.IncorrectUsage(
            f"The width of the {shape_name} is an empty array, so there is no line to "
            f"put on the peaks. Check where '{parameter}' comes from."
        )
    if not np.all(np.isfinite(array)):
        raise exception.IncorrectUsage(
            f"The width of the {shape_name} has to be finite, but '{parameter}' is "
            f"{width}. An infinitely wide line carries no weight anywhere, and a width "
            "that is not a number usually comes from an earlier division by zero."
        )
    if not np.all(array > 0):
        raise exception.IncorrectUsage(
            f"The width of the {shape_name} has to be positive everywhere, but "
            f"'{parameter}' is {width}. A width of zero describes a delta peak, which "
            "no mesh can represent, and a negative width describes no line at all."
        )
    # a plain number stays a plain number, so that the converted width prints like one
    return array if array.ndim else float(array)


class _LineShape:
    """Width bookkeeping shared by the normalized line shapes.

    A subclass is a frozen dataclass with two optional fields -- ``fwhm`` and the
    parameter its analytic form is written with -- and connects them with the class
    attributes ``_alias`` and ``_per_fwhm``. Exactly one of the two must be given; the
    other is filled in here, so both are available afterwards whichever one was used.
    """

    def __post_init__(self):
        self._raise_error_unless_exactly_one_width_is_given()
        given = "fwhm" if self.fwhm is not None else self._alias
        width = _validated_width(getattr(self, given), type(self).__name__, given)
        # the instance is frozen, so the two widths can only be resolved this way; that
        # is the point, because a later assignment would desynchronize them
        if given == "fwhm":
            object.__setattr__(self, "fwhm", width)
            object.__setattr__(self, self._alias, self._per_fwhm * width)
        else:
            object.__setattr__(self, self._alias, width)
            object.__setattr__(self, "fwhm", width / self._per_fwhm)

    def _raise_error_unless_exactly_one_width_is_given(self):
        given = [
            name for name in ("fwhm", self._alias) if getattr(self, name) is not None
        ]
        if len(given) == 1:
            return
        name = type(self).__name__
        if given:
            raise exception.IncorrectUsage(
                f"Please give the width of the {name} either as 'fwhm', the full width "
                f"at half maximum, or as '{self._alias}', but not as both. The two "
                "describe the same line, so giving both cannot be resolved."
            )
        raise exception.IncorrectUsage(
            f"Please give the width of the {name} either as 'fwhm', the full width at "
            f"half maximum, or as '{self._alias}'. Without one of them, {name}() does "
            "not describe a line yet."
        )

    def _with_mesh_axis(self):
        """The same shape with its width carrying one more trailing axis.

        :func:`broaden` lays the peaks out along the second to last axis and the mesh
        along the last one. A width given per peak has the shape of the positions, so it
        needs the mesh axis before it lines up with the offsets it is evaluated at.
        """
        width = np.asarray(getattr(self, self._alias))[..., np.newaxis]
        return type(self)(**{self._alias: width})


@dataclass(kw_only=True, frozen=True)
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

    Examples
    --------
    Whichever width you give, the other one is available afterwards

    >>> from py4vasp.broadening import Gaussian
    >>> round(Gaussian(fwhm=1.0).sigma, 6)
    0.424661
    >>> round(Gaussian(sigma=1.0).fwhm, 6)
    2.35482
    """

    fwhm: Optional[ArrayLike] = None
    sigma: Optional[ArrayLike] = None
    _alias = "sigma"
    _per_fwhm = _SIGMA_PER_FWHM

    def profile(self, offsets):
        """Evaluate the line shape at the given distances from its center.

        Parameters
        ----------
        offsets
            Distance from the center of the peak, in the same unit as the width.

        Returns
        -------
        -
            The line shape with the shape of *offsets*, normalized such that it
            integrates to one.

        Examples
        --------
        The value at half the full width is half the value at the center

        >>> from py4vasp.broadening import Gaussian
        >>> shape = Gaussian(fwhm=2.0)
        >>> float(shape.profile(1.0) / shape.profile(0.0))
        0.5
        """
        return np.exp(-0.5 * (offsets / self.sigma) ** 2) / (
            self.sigma * np.sqrt(2 * np.pi)
        )


@dataclass(kw_only=True, frozen=True)
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

    Examples
    --------
    Whichever width you give, the other one is available afterwards

    >>> from py4vasp.broadening import Lorentzian
    >>> Lorentzian(fwhm=1.0).gamma
    0.5
    >>> Lorentzian(gamma=2.0).fwhm
    4.0
    """

    fwhm: Optional[ArrayLike] = None
    gamma: Optional[ArrayLike] = None
    _alias = "gamma"
    _per_fwhm = _GAMMA_PER_FWHM

    def profile(self, offsets):
        """Evaluate the line shape at the given distances from its center.

        Parameters
        ----------
        offsets
            Distance from the center of the peak, in the same unit as the width.

        Returns
        -------
        -
            The line shape with the shape of *offsets*, normalized such that it
            integrates to one.

        Examples
        --------
        The value at half the full width is half the value at the center

        >>> from py4vasp.broadening import Lorentzian
        >>> shape = Lorentzian(fwhm=2.0)
        >>> float(shape.profile(1.0) / shape.profile(0.0))
        0.5
        """
        return self.gamma / np.pi / (offsets**2 + self.gamma**2)


def _raise_error_if_not_a_line_shape(shape):
    if isinstance(shape, _LineShape):
        return
    raise exception.IncorrectUsage(
        f"The shape {shape!r} is not a line shape, so the peaks cannot be given one. "
        "Pass an instance such as py4vasp.broadening.Gaussian(fwhm=0.1) or "
        "py4vasp.broadening.Lorentzian(fwhm=0.1)."
    )


def _validated_mesh(mesh):
    array = _numeric_array(mesh)
    if array is None:
        raise exception.IncorrectUsage(
            f"The mesh has to be an array of numbers, but it is {mesh!r}."
        )
    array = np.atleast_1d(array)
    if array.ndim > 1:
        raise exception.IncorrectUsage(
            "The mesh has to be one dimensional, because it becomes the last axis of "
            f"the spectrum, but its shape is {array.shape}."
        )
    return array


def _validated_positions(positions):
    array = _numeric_array(positions)
    if array is not None:
        return np.atleast_1d(array)
    # the complex check comes second because it needs an array to look at, and a ragged
    # sequence cannot be made into one
    if _is_complex(positions):
        raise exception.IncorrectUsage(
            "The positions of the peaks are complex. py4vasp reports an unstable phonon "
            "mode as an imaginary frequency, for instance, so decide what such a peak "
            "means on a real axis -- its signed real part, say -- rather than letting "
            "the imaginary part be dropped here."
        )
    raise exception.IncorrectUsage(
        "The positions of the peaks have to be an array of numbers, but they are "
        f"{positions!r}."
    )


def _is_complex(values):
    try:
        return np.iscomplexobj(values)
    except ValueError:
        return False


def _validated_weights(weights, positions):
    array = _numeric_array(1.0 if weights is None else weights)
    if array is None:
        raise exception.IncorrectUsage(
            f"The weights of the peaks have to be an array of numbers, but they are "
            f"{weights!r}."
        )
    try:
        return np.broadcast_to(array, positions.shape)
    except ValueError as error:
        raise exception.IncorrectUsage(
            "There is one weight per peak, so the weights have to broadcast against "
            f"the positions of shape {positions.shape}, but their shape is "
            f"{array.shape}."
        ) from error


# Two levels closer than this count as one. py4vasp measures every frequency as an
# energy in eV, and 0.1 meV is about 0.8 cm^-1: below what a vibrational experiment
# resolves and above the noise a diagonalization leaves behind.
DEFAULT_DEGENERACY_TOLERANCE = 1e-4


def degenerate_groups(values, tolerance=DEFAULT_DEGENERACY_TOLERANCE):
    """Group the values that lie within *tolerance* of one another.

    Symmetry forces some eigenvalues of a physical problem to coincide exactly, so a
    calculation reports them as a handful of values repeated rather than as distinct
    ones. Recovering which ones belong together is what this does.

    Parameters
    ----------
    values : ArrayLike
        The values to group. Complex values are sorted by their real part first, so
        that a purely imaginary one does not join a real one of the same magnitude.
    tolerance : float
        Values closer than this form one group. Members are chained, so a group may
        end up wider than the tolerance if its members overlap in sequence.

    Returns
    -------
    list
        One list of indices per group, the groups ordered by value and the indices
        within a group ascending, so that each one indexes *values* directly.
    """
    values = np.asarray(values)
    order = np.lexsort((values.imag, values.real))
    groups = []
    for index in order:
        if groups and abs(values[index] - values[groups[-1][-1]]) <= tolerance:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    return [sorted(group) for group in groups]


def _warn_if_the_mesh_does_not_resolve(mesh, shape):
    if len(mesh) < 2:
        return
    spacing = np.max(np.abs(np.diff(mesh)))
    width = np.min(shape.fwhm)
    # below about 1.5 points per width the discrete sum loses several percent of the
    # weight, and it degrades fast from there; two points leaves a margin
    if width >= 2 * spacing:
        return
    message = f"""The mesh does not resolve the line shape.
    Its spacing is {spacing:.3g} but the narrowest width is {width:.3g}, so the peaks
    fall between the mesh points and the spectrum is wrong by however much of each line
    the mesh happened to catch. Use a finer mesh, or check that the width is quoted in
    the same unit as the mesh."""
    warnings.warn(message, UserWarning)


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
        Positions at which the spectrum is evaluated, e.g. an energy axis. It has to be
        one dimensional and should resolve the width, see the notes below.
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
        *positions*, which gives every peak, or every band, its own width.

    Returns
    -------
    -
        The spectrum, with the shape of *positions* except that its last axis is
        replaced by the mesh.

    Notes
    -----
    The mesh has to resolve the width: a line narrower than the spacing between mesh
    points falls between them, and the spectrum is then wrong by whatever fraction of
    each line the mesh happened to catch. Nothing about the result gives that away, so
    broadening warns when fewer than two mesh points fit inside the narrowest width.
    The usual cause is quoting the width in a different unit than the mesh.

    The line shapes are not truncated, so the intermediate array holds one value per
    mesh point and peak. Broadening very many peaks onto a very fine mesh is therefore
    limited by memory rather than by time.

    Examples
    --------
    >>> import numpy as np
    >>> from py4vasp.broadening import Gaussian, broaden
    >>> energies = np.linspace(-5, 5, 1001)
    >>> peaks, weights = [-1.5, 0.5], [2.0, 1.0]
    >>> spectrum = broaden(energies, peaks, weights, shape=Gaussian(fwhm=0.4))
    >>> round(float(np.trapezoid(spectrum, energies)), 10)
    3.0

    A wider line redistributes the spectrum but does not change what it adds up to

    >>> wider = broaden(energies, peaks, weights, shape=Gaussian(fwhm=1.2))
    >>> round(float(np.trapezoid(wider, energies)), 10)
    3.0
    >>> bool(wider.max() < spectrum.max())
    True

    A width the mesh cannot resolve still returns a spectrum, but warns, because the
    weight it reports is not the weight you gave it

    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     unresolved = broaden(energies, peaks, weights, shape=Gaussian(fwhm=0.001))
    >>> [str(w.message).splitlines()[0] for w in caught if w.category is UserWarning]
    ['The mesh does not resolve the line shape.']
    """
    _raise_error_if_not_a_line_shape(shape)
    mesh = _validated_mesh(mesh)
    _warn_if_the_mesh_does_not_resolve(mesh, shape)
    positions = _validated_positions(positions)
    weights = _validated_weights(weights, positions)
    # the peaks go on the second to last axis and the mesh on the last one. Putting the
    # mesh in front instead would right-align the width against the mesh rather than
    # against the peaks, so a width meant per band would land on the wrong axis.
    offsets = mesh - positions[..., np.newaxis]
    profile = shape._with_mesh_axis().profile(offsets)
    return np.einsum("...pm,...p->...m", profile, weights)

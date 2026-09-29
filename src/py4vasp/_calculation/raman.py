# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import warnings

import numpy as np

from py4vasp import broadening, exception, raw
from py4vasp._calculation.dispatch import (
    DataSource,
    merge_default,
    merge_graphs,
    merge_strings,
    merge_to_database,
    quantity,
)
from py4vasp._raw.models import RamanModel
from py4vasp._third_party import graph, numeric
from py4vasp._util import convert, select

# Modes below this energy translate or rotate the system instead of vibrating it. The
# same threshold and the same keyword name the phonon modes use, so that the two
# vibrational quantities drop the same modes.
_MINIMUM_FREQUENCY = 1e-3  # eV
_DEFAULT_OBSERVABLE = "powder"
# Width of a Raman line in eV. Vibrational spectroscopy quotes linewidths in cm^-1 and
# a few wavenumbers is the usual choice; 1 meV is 8.07 cm^-1.
_DEFAULT_FWHM = 1e-3  # eV
# The CODATA value in eV/K. py4vasp keeps its conversion factors in _util/convert.py,
# but those are the ones VASP itself uses for its printout; a constant a user does
# physics with wants the measured value instead. See backlog/public-unit-constants.md.
_BOLTZMANN = 8.617333262e-5  # eV/K
_MESH_MARGIN = 5  # line widths of empty axis on either side of the outermost line
_MESH_POINTS_PER_WIDTH = 8  # enough that the narrowest line is drawn as a curve
_MAXIMUM_MESH_POINTS = 200000
_DIRECTIONS = {"x": 0, "y": 1, "z": 2}
# The observables are nonlinear functions of the Raman tensor -- a squared modulus, or a
# ratio of two of them -- so they cannot be expressed as the weighted sums over an axis
# that index.Selector reduces. The selection string is still parsed with select.Tree, so
# that listing several observables works the way it does for every other quantity, but
# the labels map to functions of the two rotational invariants instead.
_OBSERVABLES = {
    # 45a^2 + 7g^2 is what a powder scatters, since every orientation contributes
    "powder": lambda isotropic, anisotropy: 45 * isotropic + 7 * anisotropy,
    "isotropic": lambda isotropic, anisotropy: isotropic,
    "anisotropy": lambda isotropic, anisotropy: anisotropy,
    # what passes a polarizer parallel to the one of the laser, and perpendicular to it
    "parallel": lambda isotropic, anisotropy: 45 * isotropic + 4 * anisotropy,
    "perpendicular": lambda isotropic, anisotropy: 3 * anisotropy,
    "depolarization": lambda isotropic, anisotropy: _ratio(
        3 * anisotropy, 45 * isotropic + 4 * anisotropy
    ),
}


# Observables that are a ratio of two activities rather than an activity. The factors
# that turn an activity into an intensity are common to numerator and denominator, so
# they cancel; applying them would report a ratio far outside the range it can take.
_RATIO_OBSERVABLES = frozenset({"depolarization"})


def _ratio(numerator, denominator):
    # a mode that does not scatter at all has no depolarization ratio, so reporting a
    # number for it would claim knowledge the data does not contain
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )


def _validated_number(value, name, hint=""):
    """Return the value as a float, or report that the user passed something else."""
    try:
        # a string is rejected even when it parses, because a selection is a string in
        # this API and silently accepting "2.33" for a number invites the confusion
        number = np.nan if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        number = np.nan
    if np.isfinite(number):
        return number
    message = f"{name} must be a single finite number, but {value!r} is not. {hint}"
    raise exception.IncorrectUsage(message.strip())


def _validated_modes(modes):
    """Return the mode numbers as integers, or report what is wrong with them."""
    if isinstance(modes, str) or not np.iterable(modes):
        message = (
            f"modes must be a sequence of mode numbers, but {modes!r} is not one. "
            f"Pass a list even for a single mode, e.g. modes=[{modes!r}]."
        )
        raise exception.IncorrectUsage(message)
    validated = []
    for mode in modes:
        number = _validated_number(mode, "Every entry of modes")
        if number != int(number):
            message = (
                f"The mode number {mode!r} is not a whole number. The modes are "
                "numbered the way print labels them, so they are integers."
            )
            raise exception.IncorrectUsage(message)
        validated.append(int(number))
    return validated


def _scaled(label, value, scale):
    """Apply the intensity factors, unless the observable is a ratio they cancel from."""
    if label in ("frequencies", "laser") or label in _RATIO_OBSERVABLES:
        return value
    return value * scale


def _intensity_scale(frequencies, laser, temperature):
    """Turn an activity into an intensity, broadcasting over the laser energy.

    Passing an array of laser energies gives one column per energy, which is what an
    excitation profile needs.
    """
    frequencies = np.asarray(frequencies)
    if np.ndim(laser) > 0:
        frequencies = frequencies[:, np.newaxis]
    scattered = _scattered_photon(frequencies, laser)
    return _bose_factor(frequencies, temperature) * scattered**4 / frequencies


def _bose_factor(frequencies, temperature):
    """The Stokes factor n + 1, i.e. how much a warm crystal scatters over a cold one.

    A vibration that is already excited stimulates the scattering, so a warm crystal
    gives a stronger Stokes line. The limit as the temperature goes to zero is one.
    """
    if temperature <= 0:
        return np.ones_like(frequencies)
    exponent = frequencies / (_BOLTZMANN * temperature)
    return 1 / (1 - np.exp(-exponent))


def _scattered_photon(frequencies, laser):
    """Energy the scattered photon keeps, which is zero if the laser cannot excite it."""
    return np.clip(laser - frequencies, 0.0, None)


def _mesh(frequencies, shape):
    """Energy axis wide enough for every line and fine enough to resolve the narrowest.

    The spacing follows the narrowest line, because that is the one that has to come
    out as a curve rather than a spike. The margin follows the widest, because that is
    the one reaching furthest beyond the mode it belongs to; the two differ as soon as
    the caller gives every mode its own width.

    Note that a Lorentzian has tails that reach beyond any window, so the area under
    the spectrum is slightly smaller than the total activity however wide the axis is.
    """
    _warn_if_the_line_is_wider_than_the_spectrum(frequencies, shape)
    narrowest = np.min(shape.fwhm)
    margin = _MESH_MARGIN * np.max(shape.fwhm)
    first = max(np.min(frequencies) - margin, 0.0)
    last = np.max(frequencies) + margin
    points = int(np.ceil((last - first) / narrowest * _MESH_POINTS_PER_WIDTH)) + 1
    _raise_error_if_the_mesh_would_be_too_large(points, narrowest, last - first)
    return np.linspace(first, last, points)


def _warn_if_the_line_is_wider_than_the_spectrum(frequencies, shape):
    width = np.max(shape.fwhm)
    highest = np.max(frequencies)
    if width <= highest:
        return
    message = f"""The line shape is wider than the whole spectrum.
    Its width is {width:.3g} eV but the highest mode is only at {highest:.3g} eV, so
    every line is smeared into a single flat curve. Note that py4vasp quotes the width
    as an energy in eV where a Raman experiment quotes it in cm^-1: if you meant
    {width:.3g} cm^-1, pass {width / convert.EV_TO_CM1:.3g} instead."""
    warnings.warn(message, UserWarning)


def _raise_error_if_the_mesh_would_be_too_large(points, width, span):
    if points <= _MAXIMUM_MESH_POINTS:
        return
    resolvable = span / _MAXIMUM_MESH_POINTS * _MESH_POINTS_PER_WIDTH
    message = (
        f"Drawing a line of width {width:.3g} eV over a range of {span:.3g} eV needs "
        f"{points} points, more than the {_MAXIMUM_MESH_POINTS} py4vasp will build. "
        f"The narrowest line this range resolves is about {resolvable:.3g} eV, which "
        f"is {resolvable * convert.EV_TO_CM1:.3g} cm^-1. Widen the line shape, or "
        "broaden the modes yourself with py4vasp.broadening.broaden on a mesh you "
        "choose."
    )
    raise exception.IncorrectUsage(message)


def _raise_error_if_not_a_line_shape(shape):
    if isinstance(shape, (broadening.Gaussian, broadening.Lorentzian)):
        return
    message = (
        f"The shape {shape} is not a line shape. Pass one of the line shapes of "
        "py4vasp.broadening, e.g. shape=Lorentzian(fwhm=0.001) for a width of 1 meV. "
        "The width is an energy in eV like every other one in py4vasp, so a width you "
        "know in cm^-1 has to be divided by 8065.61."
    )
    raise exception.IncorrectUsage(message)


def _mode_to_string(index, degeneracy, frequency, activity, depolarization):
    # both units of the frequency, because cm^-1 is what a Raman experiment is quoted
    # in and meV is what the rest of py4vasp and the plotted axis use
    wavenumber = frequency * convert.EV_TO_CM1
    energy = frequency * convert.EV_TO_MEV
    return (
        f"{index:4d}{degeneracy:5d}{wavenumber:15.2f}{energy:14.2f}"
        f"{activity:14.4f}{depolarization:17.4f}"
    )


def _degeneracies(frequencies):
    """How many modes share the frequency of each mode, shape ``(mode,)``."""
    degeneracies = np.ones(len(frequencies), dtype=np.int64)
    for group in numeric.degenerate_groups(frequencies):
        degeneracies[group] = len(group)
    return degeneracies


def _invariants(tensors):
    """The two rotational invariants of every tensor, shape ``(mode,)`` each.

    Every orientational average of a Raman tensor is built from these two numbers: the
    mean polarizability, which survives averaging over all orientations, and the
    anisotropy, which measures how far the tensor is from a multiple of the identity.
    """
    mean_polarizability = np.trace(tensors, axis1=1, axis2=2) / 3
    difference = lambda i, j: np.abs(tensors[:, i, i] - tensors[:, j, j]) ** 2
    off_diagonal = lambda i, j: np.abs(tensors[:, i, j]) ** 2
    anisotropy = 0.5 * (difference(0, 1) + difference(1, 2) + difference(2, 0)) + 3 * (
        off_diagonal(0, 1) + off_diagonal(1, 2) + off_diagonal(0, 2)
    )
    return np.abs(mean_polarizability) ** 2, anisotropy


class RamanHandler:
    """Handler for the raman quantity. Works with exactly one raw.Raman object."""

    def __init__(self, raw_raman: raw.Raman):
        self._raw_raman = raw_raman

    @classmethod
    def from_data(cls, raw_raman: raw.Raman) -> "RamanHandler":
        return cls(raw_raman)

    def to_dict(self, minimum_frequency: float = _MINIMUM_FREQUENCY) -> dict:
        """Read the Raman tensor and the axes it is defined on into a dictionary."""
        minimum_frequency = self._validated_minimum_frequency(minimum_frequency)
        frequencies = self._frequencies()
        vibrating = frequencies > minimum_frequency
        self._raise_error_if_no_mode_vibrates(frequencies, vibrating, minimum_frequency)
        return {
            "frequencies": frequencies[vibrating],
            "energies": np.array(self._raw_raman.energies[:]),
            "raman_tensor": self._raman_tensor()[vibrating],
        }

    def _frequencies(self):
        # VASP stores cm^-1 in this group, where the dynamical matrix uses eV. py4vasp
        # returns an energy in eV from every quantity, so the conversion belongs here
        # and not in the plot, which converts again to the meV the axis is labeled in.
        return np.array(self._raw_raman.frequencies[:]) / convert.EV_TO_CM1

    def _raman_tensor(self):
        return convert.to_complex(np.array(self._raw_raman.raman_tensor[:]))

    def activity(
        self,
        selection: str | None = None,
        *,
        laser: float = 0.0,
        minimum_frequency: float = _MINIMUM_FREQUENCY,
    ) -> dict:
        """Compute how strongly every mode scatters light."""
        data = self.to_dict(minimum_frequency)
        index = self._index_of_laser(data["energies"], laser)
        tensors = data["raman_tensor"][..., index]
        self._raise_error_if_not_symmetric(tensors)
        invariants = _invariants(tensors)
        result = {
            "frequencies": data["frequencies"],
            "laser": data["energies"][index],
        }
        tree = select.Tree.from_selection(selection or _DEFAULT_OBSERVABLE)
        for choice in tree.selections():
            label = "_".join(choice)
            result[label] = self._observable(label, tensors, invariants)
        return result

    def _observable(self, label, tensors, invariants):
        if label in _OBSERVABLES:
            return _OBSERVABLES[label](*invariants)
        if len(label) == 2 and all(character in _DIRECTIONS for character in label):
            row, column = (_DIRECTIONS[character] for character in label)
            return np.abs(tensors[:, row, column]) ** 2
        self._raise_error_unknown_observable(label)

    def _index_of_laser(self, energies, laser):
        self._raise_error_if_laser_outside_grid(energies, laser)
        return int(np.argmin(np.abs(energies - float(laser))))

    def _raise_error_if_laser_outside_grid(self, energies, laser):
        hint = "Divide 1239.84 eV nm by a wavelength in nm to get the photon energy."
        laser = _validated_number(laser, "laser", hint)
        if np.min(energies) <= laser <= np.max(energies):
            return
        message = (
            f"The laser energy {laser} is not a single energy within the range "
            f"[{np.min(energies):.4g}, {np.max(energies):.4g}] eV that VASP evaluated "
            "the Raman tensor on. Note that py4vasp expects the photon energy in eV; "
            "if you know your laser by its wavelength, divide 1239.84 eV nm by it."
        )
        raise exception.IncorrectUsage(message)

    def _raise_error_if_not_symmetric(self, tensors):
        deviation = np.max(np.abs(tensors - np.swapaxes(tensors, 1, 2)))
        if deviation <= 1e-10 * max(np.max(np.abs(tensors)), 1.0):
            return
        message = (
            "The Raman tensor is not symmetric in its two directions, which every "
            "orientational average assumes. Exchanging the polarization of the "
            f"incoming and the scattered light changes it by {deviation:.4g}. Please "
            "report this, because VASP is expected to write a symmetric tensor."
        )
        raise exception.DataMismatch(message)

    def _raise_error_unknown_observable(self, label):
        directions = ", ".join(
            f"{row}{column}" for row in _DIRECTIONS for column in _DIRECTIONS
        )
        message = (
            f"'{label}' is not an observable of the Raman tensor. Choose one of "
            f"{', '.join(_OBSERVABLES)} to average over the orientations of a "
            f"crystallite, or one of {directions} for a single element of the tensor "
            "of an oriented crystal."
        )
        raise exception.IncorrectUsage(message)

    def intensity(
        self,
        selection: str | None = None,
        *,
        laser: float,
        temperature: float = 0.0,
        minimum_frequency: float = _MINIMUM_FREQUENCY,
    ) -> dict:
        """Scale the activity to what a spectrometer measures."""
        temperature = self._validated_temperature(temperature)
        data = self.activity(
            selection, laser=laser, minimum_frequency=minimum_frequency
        )
        frequencies = data["frequencies"]
        scale = _intensity_scale(frequencies, data["laser"], temperature)
        return {label: _scaled(label, value, scale) for label, value in data.items()}

    def excitation_profile(
        self,
        selection: str | None = None,
        *,
        modes=None,
        temperature: float | None = None,
        minimum_frequency: float = _MINIMUM_FREQUENCY,
    ) -> graph.Graph:
        """Draw the selected observable against the energy of the laser."""
        data = self.to_dict(minimum_frequency)
        tensors = data["raman_tensor"]
        self._raise_error_if_not_symmetric(tensors)
        frequencies = data["frequencies"]
        energies = data["energies"]
        indices = self._selected_modes(modes, len(frequencies))
        # the invariants carry the photon energy along as a trailing axis, so the whole
        # profile comes out of one evaluation rather than one per point of the grid
        invariants = _invariants(tensors)
        if temperature is None:
            scale, quantity = 1.0, "activity"
        else:
            temperature = self._validated_temperature(temperature)
            scale = _intensity_scale(frequencies, energies, temperature)
            quantity = "intensity"
        series = []
        tree = select.Tree.from_selection(selection or _DEFAULT_OBSERVABLE)
        for choice in tree.selections():
            label = "_".join(choice)
            values = _scaled(label, self._observable(label, tensors, invariants), scale)
            series += [
                graph.Series(
                    energies,
                    values[index],
                    f"{label} {frequencies[index] * convert.EV_TO_CM1:.0f} cm-1",
                )
                for index in indices
            ]
        return graph.Graph(
            series=series,
            xlabel="Laser energy (eV)",
            ylabel=f"Raman {quantity}",
        )

    def _selected_modes(self, modes, number_modes):
        if modes is None:
            return range(number_modes)
        modes = _validated_modes(modes)
        self._raise_error_if_mode_does_not_exist(modes, number_modes)
        return [mode - 1 for mode in modes]

    def _raise_error_if_mode_does_not_exist(self, modes, number_modes):
        invalid = [mode for mode in modes if not 1 <= mode <= number_modes]
        if not invalid:
            return
        message = (
            f"The modes {invalid} do not exist. The modes are numbered the way print "
            f"labels them, so they run from 1 to {number_modes}. Note that the modes "
            "below minimum_frequency are left out before they are numbered."
        )
        raise exception.IncorrectUsage(message)

    def _validated_temperature(self, temperature):
        temperature = _validated_number(temperature, "temperature", "It is in Kelvin.")
        self._raise_error_if_temperature_is_negative(temperature)
        return temperature

    def _raise_error_if_temperature_is_negative(self, temperature):
        if temperature >= 0:
            return
        message = (
            f"The temperature {temperature} is negative. Pass a temperature in Kelvin, "
            "or 0 for the limit in which the crystal is not vibrating on its own."
        )
        raise exception.IncorrectUsage(message)

    def to_graph(
        self,
        selection: str | None = None,
        *,
        laser: float = 0.0,
        temperature: float | None = None,
        shape=None,
        minimum_frequency: float = _MINIMUM_FREQUENCY,
    ) -> graph.Graph:
        """Broaden the lines of the selected observable into a spectrum."""
        shape = broadening.Lorentzian(fwhm=_DEFAULT_FWHM) if shape is None else shape
        _raise_error_if_not_a_line_shape(shape)
        if temperature is None:
            data = self.activity(
                selection, laser=laser, minimum_frequency=minimum_frequency
            )
            quantity = "activity"
        else:
            self._raise_error_if_the_laser_cannot_excite_anything(laser)
            data = self.intensity(
                selection,
                laser=laser,
                temperature=temperature,
                minimum_frequency=minimum_frequency,
            )
            quantity = "intensity"
        frequencies = data["frequencies"]
        self._raise_error_if_a_ratio_is_broadened(data)
        mesh = _mesh(frequencies, shape)
        # the axis is drawn in meV, so the spectrum is divided by the same factor to
        # keep the area under a line equal to the activity of that line
        series = [
            graph.Series(
                mesh * convert.EV_TO_MEV,
                broadening.broaden(mesh, frequencies, data[label], shape=shape)
                / convert.EV_TO_MEV,
                label,
            )
            for label in data
            if label not in ("frequencies", "laser")
        ]
        return graph.Graph(
            series=series, xlabel="ω (meV)", ylabel=f"Raman {quantity} (1/meV)"
        )

    def _raise_error_if_a_ratio_is_broadened(self, data):
        ratios = sorted(set(data) & _RATIO_OBSERVABLES)
        if not ratios:
            return
        message = (
            f"{', '.join(ratios)} cannot be broadened into a spectrum. It is a ratio "
            "of two activities, so it has no area to spread over the axis, and a mode "
            "that does not scatter has no ratio at all. Read it per mode with "
            "activity() instead, or plot 'parallel, perpendicular' to see the two "
            "polarizations that the ratio is built from."
        )
        raise exception.IncorrectUsage(message)

    def _raise_error_if_the_laser_cannot_excite_anything(self, laser):
        hint = "Divide 1239.84 eV nm by a wavelength in nm to get the photon energy."
        if _validated_number(laser, "laser", hint) > 0:
            return
        message = (
            "Plotting an intensity needs the energy of the laser, because the "
            "scattered photon has an energy only once you say what went in. Pass "
            "laser= in eV, for example laser=2.33 for the 532 nm line. Leave the "
            "temperature out to plot the activity, which needs no laser."
        )
        raise exception.IncorrectUsage(message)

    def to_database(self) -> RamanModel:
        data = self.activity()
        strongest = int(np.argmax(data["powder"]))
        return RamanModel(
            number_modes=len(data["frequencies"]),
            frequency_max=float(np.max(data["frequencies"])),
            photon_energy_max=float(np.max(self._raw_raman.energies[:])),
            strongest_frequency=float(data["frequencies"][strongest]),
            strongest_activity=float(data["powder"][strongest]),
        )

    def __str__(self) -> str:
        data = self.activity("powder, depolarization")
        header = f"Raman activity at a laser energy of {data['laser']:.2f} eV"
        columns = (
            "mode  deg   omega (cm-1)   omega (meV)      activity   depolarization"
        )
        rows = zip(
            _degeneracies(data["frequencies"]),
            data["frequencies"],
            data["powder"],
            data["depolarization"],
        )
        table = "\n".join(
            _mode_to_string(index, *row) for index, row in enumerate(rows, start=1)
        )
        return f"{header}\n{'-' * len(header)}\n{columns}\n{table}"

    def _validated_minimum_frequency(self, minimum_frequency):
        hint = "It is an energy in eV, so divide a value you know in cm^-1 by 8065.61."
        minimum_frequency = _validated_number(
            minimum_frequency, "minimum_frequency", hint
        )
        self._raise_error_if_frequency_is_negative(minimum_frequency)
        return minimum_frequency

    def _raise_error_if_no_mode_vibrates(self, frequencies, vibrating, minimum):
        if np.any(vibrating):
            return
        highest = np.max(frequencies) if len(frequencies) else 0.0
        message = (
            f"No mode has a frequency above minimum_frequency={minimum}. The highest "
            f"is {highest:.4g} eV, which is {highest * convert.EV_TO_CM1:.4g} cm^-1. "
            "Note that minimum_frequency is an energy in eV rather than a wavenumber; "
            "divide a value you know in cm^-1 by 8065.61."
        )
        raise exception.IncorrectUsage(message)

    def _raise_error_if_frequency_is_negative(self, minimum_frequency):
        if minimum_frequency >= 0:
            return
        message = (
            f"The minimum frequency {minimum_frequency} is negative, but it is "
            "compared to the magnitude of the frequency of a mode, which is never "
            "negative. Use minimum_frequency=0 to keep every mode."
        )
        raise exception.IncorrectUsage(message)


@quantity("raman")
class Raman(graph.Mixin):
    """The Raman tensor describes how a vibration changes the susceptibility.

    Light scattering off a crystal exchanges energy with its vibrations, so the
    scattered light carries lines shifted by the phonon frequencies. How strong each
    line is follows from the derivative of the susceptibility with respect to that
    mode -- the Raman tensor. VASP evaluates it for every mode and, because the
    susceptibility depends on the color of the light, on a mesh of photon energies.
    That second axis is what makes a Raman measurement resonant: tuning the laser onto
    an electronic transition can amplify particular lines by orders of magnitude.

    py4vasp reports the frequencies as energies in eV like every other quantity,
    although VASP stores them as wavenumbers in cm^-1 in this group.

    Notes
    -----
    VASP stores the magnitude of every frequency here, so a mode that is actually
    unstable is indistinguishable from a stable one of the same magnitude. The
    ``minimum_frequency`` argument drops the modes near zero, but it cannot recognize
    an unstable mode of a structure that is not relaxed. Check
    :attr:`~py4vasp.calculation.phonon.mode`, which reports those as imaginary, if you
    are unsure whether your structure is at a minimum.

    See Also
    --------
    py4vasp._calculation.phonon_mode.PhononMode :
        The same modes with their eigenvalues and their displacement patterns, where an
        unstable mode is reported as an imaginary frequency.

    Examples
    --------
    First, we create some example data so that you can follow along. Please define a
    variable `path` with the path to a directory that does not exist yet. Alternatively,
    use your own data if you have run VASP.

    >>> from py4vasp import demo
    >>> calculation = demo.calculation(path)

    Reading the quantity gives the tensor and the two axes it is defined on

    >>> raman = calculation.raman.read()
    >>> sorted(raman)
    ['energies', 'frequencies', 'raman_tensor']

    The tensor has two directions, because it relates the polarization of the incoming
    light to that of the scattered light

    >>> raman["raman_tensor"].shape
    (18, 3, 3, 301)

    Printing the quantity lists every mode with the strength of its line and how much
    it depolarizes the scattered light. Sr2TiO4 has an inversion centre, so half of its
    modes are infrared active instead and do not scatter at all -- a mode that does not
    scatter has no depolarization ratio either

    >>> print(calculation.raman)
    Raman activity at a laser energy of 0.00 eV
    -------------------------------------------
    mode  deg   omega (cm-1)   omega (meV)      activity   depolarization
       1    1         106.74         13.23        0.3862           0.0139
       2    1         120.08         14.89        0.0000              nan
       3...
    """

    def __init__(self, source, quantity_name: str = "raman"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_raman: raw.Raman) -> "Raman":
        """Create a Raman dispatcher from raw data (convenience for testing)."""
        return cls(source=DataSource(raw_raman))

    def _handler_factory(self, raw_data):
        return RamanHandler.from_data(raw_data)

    def read(
        self,
        selection: str | None = None,
        *,
        minimum_frequency: float = _MINIMUM_FREQUENCY,
    ) -> dict:
        """Read the Raman tensor and the axes it is defined on into a dictionary.

        Parameters
        ----------
        selection : str | None
            Select which source of the quantity is read.
        minimum_frequency : float
            Modes with a frequency below this energy in eV are omitted. They translate
            or rotate the system instead of vibrating it, so they do not scatter light.
            Pass 0 to keep every mode that vibrates at all; a mode of exactly zero
            frequency is dropped whatever you pass, because it is not a vibration.

        Returns
        -------
        dict
            The frequency of every mode in eV, the photon energies in eV at which the
            Raman tensor was evaluated, and the complex tensor itself with one mode
            axis, two direction axes and one photon-energy axis.

        Examples
        --------
        First, we create some example data so that you can follow along. Please define a
        variable `path` with the path to a directory that does not exist yet.
        Alternatively, use your own data if you have run VASP.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        The frequencies are energies in eV like everywhere else in py4vasp. A vibration
        of an oxide is a few tens of meV, so the numbers are small

        >>> raman = calculation.raman.read()
        >>> bool(0.01 < raman["frequencies"].max() < 0.1)
        True

        Every mode that vibrates is included. The three that merely translate the
        crystal are dropped, because they cannot change its polarizability

        >>> len(raman["frequencies"])
        18

        The tensor was evaluated at each of these photon energies, which is what lets
        you ask for the spectrum a particular laser produces

        >>> raman["energies"].shape
        (301,)
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            RamanHandler.to_dict,
            # keyword argument, because the dispatcher only passes the selection on when
            # the user made one and would otherwise shift the positional arguments
            minimum_frequency=minimum_frequency,
        )

    def to_dict(
        self,
        selection: str | None = None,
        *,
        minimum_frequency: float = _MINIMUM_FREQUENCY,
    ) -> dict:
        """Convenient alias for :py:meth:`read`. Please read the documentation there."""
        return self.read(selection, minimum_frequency=minimum_frequency)

    def activity(
        self,
        selection: str | None = None,
        *,
        laser: float = 0.0,
        minimum_frequency: float = _MINIMUM_FREQUENCY,
    ) -> dict:
        """Compute how strongly every mode scatters light.

        The Raman tensor relates the polarization of the incoming light to that of the
        scattered light, so what you measure depends on how the crystal is oriented and
        on which polarizations you let through. A powder averages over all orientations;
        an oriented single crystal picks out one element of the tensor.

        Parameters
        ----------
        selection : str | None
            Which observable to compute, ``"powder"`` by default. Use ``"isotropic"``
            or ``"anisotropy"`` for the two rotational invariants the averages are
            built from, ``"parallel"`` and ``"perpendicular"`` for the light that
            passes a polarizer aligned with the laser or crossed with it, and
            ``"depolarization"`` for their ratio. Pass two directions such as
            ``"xy"`` to select a single element of the tensor instead. Select several
            at once by separating them with a comma.
        laser : float
            Photon energy of the laser in eV. VASP evaluates the Raman tensor on a mesh
            of photon energies and py4vasp picks the closest one, which it reports back
            under ``"laser"``. The default of 0 is the limit far below any electronic
            transition, which is the ordinary non-resonant Raman experiment.
        minimum_frequency : float
            Modes with a frequency below this energy in eV are omitted.

        Returns
        -------
        dict
            The frequency of every mode in eV, the photon energy that was actually
            used, and one entry per selected observable. A mode that does not scatter
            has no depolarization ratio, which is reported as not-a-number.

        Examples
        --------
        First, we create some example data so that you can follow along. Please define a
        variable `path` with the path to a directory that does not exist yet.
        Alternatively, use your own data if you have run VASP.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        By default you get the activity a powder sample would show

        >>> activity = calculation.raman.activity()
        >>> sorted(activity)
        ['frequencies', 'laser', 'powder']

        The two polarizations add up to it, because a powder scatters both

        >>> import numpy as np
        >>> both = calculation.raman.activity("parallel, perpendicular")
        >>> total = both["parallel"] + both["perpendicular"]
        >>> bool(np.allclose(total, activity["powder"]))
        True

        Tuning the laser onto an electronic transition changes the intensities, which
        is the resonance a Raman experiment looks for

        >>> resonant = calculation.raman.activity(laser=3.0)
        >>> bool(np.any(resonant["powder"] > activity["powder"]))
        True
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            RamanHandler.activity,
            laser=laser,
            minimum_frequency=minimum_frequency,
        )

    def intensity(
        self,
        selection: str | None = None,
        *,
        laser: float,
        temperature: float = 0.0,
        minimum_frequency: float = _MINIMUM_FREQUENCY,
    ) -> dict:
        """Scale the activity to what a spectrometer measures.

        The activity says how strongly a vibration modulates the susceptibility, which
        is not yet what a detector counts. Three factors separate them: a warm crystal
        is already vibrating and scatters more, a photon that has given up energy to a
        vibration carries less, and the scattered power goes with the fourth power of
        the frequency of the light that comes out.

        Parameters
        ----------
        selection : str | None
            Which observable to scale, ``"powder"`` by default. See
            :py:meth:`activity` for the alternatives.
        laser : float
            Photon energy of the laser in eV. There is no default, because an
            intensity is not defined without one: the scattered photon has an energy
            only once you say what went in. Divide 1239.84 eV nm by your wavelength in
            nm to get it -- 532 nm is 2.33 eV.
        temperature : float
            Temperature of the sample in Kelvin, 0 by default, which is the limit in
            which the crystal is not vibrating on its own.
        minimum_frequency : float
            Modes with a frequency below this energy in eV are omitted.

        Returns
        -------
        dict
            The frequency of every mode in eV, the photon energy that was used, and
            one entry per selected observable. The intensities are on an arbitrary
            scale, so compare them with one another rather than with an absolute
            number. A mode that costs more energy than one laser photon carries cannot
            be excited and comes back as exactly zero.

        Examples
        --------
        First, we create some example data so that you can follow along. Please define a
        variable `path` with the path to a directory that does not exist yet.
        Alternatively, use your own data if you have run VASP.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        A green laser of 532 nm is 2.33 eV

        >>> intensity = calculation.raman.intensity(laser=2.33)
        >>> sorted(intensity)
        ['frequencies', 'laser', 'powder']

        Warming the sample makes every line stronger, because the crystal is already
        vibrating and stimulates the scattering

        >>> import numpy as np
        >>> warm = calculation.raman.intensity(laser=2.33, temperature=300)
        >>> bool(np.all(warm["powder"] >= intensity["powder"]))
        True

        The effect is largest for the modes of lowest energy, which are the easiest to
        excite thermally

        >>> ratio = np.divide(warm["powder"], intensity["powder"],
        ...     out=np.ones_like(warm["powder"]), where=intensity["powder"] > 0)
        >>> bool(ratio[0] > ratio[-1])
        True
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            RamanHandler.intensity,
            laser=laser,
            temperature=temperature,
            minimum_frequency=minimum_frequency,
        )

    def excitation_profile(
        self,
        selection: str | None = None,
        *,
        modes=None,
        temperature: float | None = None,
        minimum_frequency: float = _MINIMUM_FREQUENCY,
    ) -> graph.Graph:
        """Draw the selected observable against the energy of the laser.

        A Raman line does not have one strength: it grows as the laser approaches an
        electronic transition of the material, sometimes by orders of magnitude, and
        which line grows tells you which transition it is. VASP evaluates the Raman
        tensor on a whole mesh of photon energies, so this profile costs no extra
        calculation -- it is the other axis of the data you already have.

        Parameters
        ----------
        selection : str | None
            Which observable to draw, ``"powder"`` by default. See
            :py:meth:`activity` for the alternatives.
        modes : Sequence[int] | None
            Which modes to draw, numbered the way :py:meth:`print` labels them, so
            counting from one. Every mode by default, which is usually more curves
            than a figure can carry; pick the few strong lines instead.
        temperature : float | None
            Leave this out to draw the bare activity. Give a temperature in Kelvin to
            draw the intensity a spectrometer measures instead. No laser energy is
            needed here, because the laser energy is the axis.
        minimum_frequency : float
            Modes with a frequency below this energy in eV are omitted, and the
            remaining ones are numbered afterwards.

        Returns
        -------
        Graph
            One curve per mode and observable against the photon energy in eV.

        Examples
        --------
        First, we create some example data so that you can follow along. Please define a
        variable `path` with the path to a directory that does not exist yet.
        Alternatively, use your own data if you have run VASP.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        Pick the modes you are interested in by the number the table gives them

        >>> graph = calculation.raman.excitation_profile(modes=[1, 3])
        >>> [series.label for series in graph.series]
        ['powder 107 cm-1', 'powder 137 cm-1']
        >>> graph.xlabel
        'Laser energy (eV)'

        The curve peaks where the laser meets an electronic transition, which is what
        a resonance Raman experiment scans for

        >>> import numpy as np
        >>> series = graph.series[0]
        >>> resonance = series.x[np.argmax(series.y)]
        >>> bool(resonance > 0)
        True
        """
        return merge_graphs(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            RamanHandler.excitation_profile,
            modes=modes,
            temperature=temperature,
            minimum_frequency=minimum_frequency,
        )

    def to_graph(
        self,
        selection: str | None = None,
        *,
        laser: float = 0.0,
        temperature: float | None = None,
        shape=None,
        minimum_frequency: float = _MINIMUM_FREQUENCY,
    ) -> graph.Graph:
        """Broaden the lines of the selected observable into a spectrum.

        A calculation reports a Raman spectrum as a list of lines, but a measurement
        shows peaks of finite width. Giving every line a shape and adding them up is
        what makes the two comparable.

        Parameters
        ----------
        selection : str | None
            Which observable to plot, ``"powder"`` by default. See
            :py:meth:`activity` for the alternatives; selecting several draws one
            spectrum each.
        laser : float
            Photon energy of the laser in eV, 0 by default, which is the ordinary
            non-resonant experiment.
        temperature : float | None
            Leave this out to plot the bare activity. Give a temperature in Kelvin to
            plot the intensity a spectrometer measures instead, which also needs a
            laser energy; pass 0 for that intensity without the thermal enhancement.
            The axis label says which of the two is drawn.
        shape : Gaussian | Lorentzian
            The line shape every mode is broadened with, by default a Lorentzian of
            1 meV. Use :class:`py4vasp.broadening.Gaussian` or
            :class:`py4vasp.broadening.Lorentzian` and note that the width is an
            energy in eV, so a width known in cm^-1 has to be divided by 8065.61.
        minimum_frequency : float
            Modes with a frequency below this energy in eV are omitted.

        Returns
        -------
        Graph
            The spectrum drawn against the energy of the vibration in meV. The line
            shapes carry unit area, so the area under a peak is the activity of the
            mode that produced it -- exactly for a Gaussian, and up to a percent or so
            for the default Lorentzian, whose tails reach beyond any axis.

        Examples
        --------
        First, we create some example data so that you can follow along. Please define a
        variable `path` with the path to a directory that does not exist yet.
        Alternatively, use your own data if you have run VASP.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        Plotting the quantity broadens every line into a peak

        >>> graph = calculation.raman.plot()
        >>> graph.xlabel, graph.ylabel
        ('ω (meV)', 'Raman activity (1/meV)')

        Choose a wider line shape when you want to compare with an experiment that does
        not resolve neighboring modes

        >>> from py4vasp.broadening import Gaussian
        >>> graph = calculation.raman.to_graph(shape=Gaussian(fwhm=0.004))

        The area under the spectrum is the total activity, because every line shape
        carries unit area. A Gaussian shows it exactly; a Lorentzian loses a little,
        because its tails reach beyond whatever axis you draw

        >>> import numpy as np
        >>> series = graph.series[0]
        >>> total = calculation.raman.activity()["powder"].sum()
        >>> bool(np.isclose(np.trapezoid(series.y, series.x), total))
        True

        Ask for the two polarizations to see which modes are totally symmetric

        >>> graph = calculation.raman.to_graph("parallel, perpendicular")
        >>> [series.label for series in graph.series]
        ['parallel', 'perpendicular']

        Give a temperature to draw what a spectrometer measures instead of the bare
        activity. The axis label says which of the two you are looking at

        >>> graph = calculation.raman.to_graph(laser=2.33, temperature=300)
        >>> graph.ylabel
        'Raman intensity (1/meV)'
        """
        return merge_graphs(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            RamanHandler.to_graph,
            laser=laser,
            temperature=temperature,
            shape=shape,
            minimum_frequency=minimum_frequency,
        )

    def print(self, selection: str | None = None) -> None:
        """Print a string representation of this quantity.

        Parameters
        ----------
        selection : str | None
            Select which source of the quantity is printed. If you select multiple
            sources, py4vasp prints one block per source.
        """
        print(self.__str__(selection))

    def selections(self) -> dict:
        """Returns possible alternatives for this particular quantity VASP can produce.

        Returns
        -------
        dict
            The key indicates this quantity and the value lists the possible choices
            for the selection argument of its other methods.
        """
        from py4vasp._raw import definition as raw_module

        directions = [f"{row}{column}" for row in _DIRECTIONS for column in _DIRECTIONS]
        return {
            self._quantity_name: list(raw_module.selections(self._quantity_name)),
            "observables": list(_OBSERVABLES),
            "directions": directions,
        }

    def __str__(self, selection: str | None = None) -> str:
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            RamanHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def _to_database(self) -> dict:
        """Return {quantity[_selection]: handler_result} for database storage."""
        return merge_to_database(
            self._source,
            self._quantity_name,
            RamanHandler.from_data,
            RamanHandler.to_database,
        )

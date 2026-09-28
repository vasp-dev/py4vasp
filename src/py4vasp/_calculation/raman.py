# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import exception, raw
from py4vasp._calculation.dispatch import (
    DataSource,
    merge_default,
    merge_strings,
    quantity,
)
from py4vasp._util import convert

# Modes below this energy translate or rotate the system instead of vibrating it. The
# same threshold and the same keyword name the phonon modes use, so that the two
# vibrational quantities drop the same modes.
_MINIMUM_FREQUENCY = 1e-3  # eV


class RamanHandler:
    """Handler for the raman quantity. Works with exactly one raw.Raman object."""

    def __init__(self, raw_raman: raw.Raman):
        self._raw_raman = raw_raman

    @classmethod
    def from_data(cls, raw_raman: raw.Raman) -> "RamanHandler":
        return cls(raw_raman)

    def to_dict(self, minimum_frequency: float = _MINIMUM_FREQUENCY) -> dict:
        """Read the Raman tensor and the axes it is defined on into a dictionary."""
        self._raise_error_if_frequency_is_negative(minimum_frequency)
        frequencies = self._frequencies()
        vibrating = frequencies > minimum_frequency
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

    def __str__(self) -> str:
        data = self.to_dict()
        frequencies = data["frequencies"] * convert.EV_TO_MEV
        energies = data["energies"]
        return f"""Raman tensor:
    modes: {len(frequencies)} between {np.min(frequencies):.2f} and {np.max(frequencies):.2f} meV
    photon energies: [{np.min(energies):.2f}, {np.max(energies):.2f}] eV, {len(energies)} points"""

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
class Raman:
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

    Printing the quantity summarizes the two axes

    >>> print(calculation.raman)
    Raman tensor:
        modes: 18 between 13.23 and 80.23 meV
        photon energies: [0.00, 12.00] eV, 301 points
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

        return {self._quantity_name: list(raw_module.selections(self._quantity_name))}

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

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

from py4vasp._util import masses


class Mass:
    """The masses py4vasp uses for the atoms when it undoes the mass weighting.

    The normal modes of :py:class:`~py4vasp._calculation.force_constant.ForceConstant`
    and :py:class:`~py4vasp._calculation.phonon_mode.PhononMode` depend on the mass
    of every atom. Their methods take an argument `masses` that defaults to the
    standard atomic weight of the element, i.e. the mass IUPAC reports for the isotope
    mixture found on earth. VASP uses the POMASS of the POTCAR instead, which agrees
    with these values to a few parts in ten thousand unless you overwrite it, for
    example to study an isotope.

    To replace the mass of one element, pass a dictionary from element to mass, e.g.
    ``masses={"O": 17.999}``. To replace the mass of a single atom, start from
    :py:attr:`STANDARD_ATOMIC_WEIGHTS` and pass one mass per atom.

    Examples
    --------
    First, we create some example data so that you can follow along. Please define a
    variable `path` with the path to a directory that does not exist yet.
    Alternatively, use your own data if you have run VASP.

    >>> from py4vasp import demo
    >>> calculation = demo.calculation(path)

    Look up the standard atomic weight of an element in atomic mass units

    >>> calculation.mass.STANDARD_ATOMIC_WEIGHTS["O"]
    15.999

    Build the mass of every atom of the structure, e.g. to make only the fourth atom
    a heavier oxygen isotope

    >>> elements = calculation.structure.read()["elements"]
    >>> masses = [calculation.mass.STANDARD_ATOMIC_WEIGHTS[e] for e in elements]
    >>> masses[3] = 17.999
    >>> masses
    [87.62, 87.62, 47.867, 17.999, 15.999, 15.999, 15.999]
    """

    STANDARD_ATOMIC_WEIGHTS = types.MappingProxyType(masses.TABLE)
    """Standard atomic weight of every element in atomic mass units, keyed by the
    chemical symbol. For the elements without a stable isotope it is the mass number
    of the longest-lived one. The mapping is read-only; copy it with ``dict(...)`` to
    modify it."""

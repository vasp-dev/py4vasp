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

    To replace the mass of some atoms, pass a dictionary whose keys each name atoms
    like a single selection of the DOS: an element, e.g. ``masses={"O": 17.999}`` for
    every oxygen, the index of one atom counted from 1, e.g. ``masses={"4": 17.999}``,
    or a range such as ``"4:5"`` that includes both ends. A single atom takes
    precedence over a range and a range over an element, whatever the order of the
    dictionary. Every atom you do not name keeps the standard atomic weight of its
    element.

    Examples
    --------
    First, we create some example data so that you can follow along. Alternatively, use
    your own data if you have run VASP.

    >>> from py4vasp import demo
    >>> calculation = demo.calculation()

    Look up the standard atomic weight of an element in atomic mass units

    >>> calculation.mass.STANDARD_ATOMIC_WEIGHTS["O"]
    15.999

    Make only the fourth atom, an oxygen, a heavier isotope; this lowers the
    frequencies of the modes in which it moves and leaves none higher

    >>> import numpy as np
    >>> default = calculation.force_constant.frequencies()
    >>> heavy = calculation.force_constant.frequencies(masses={"4": 17.999})
    >>> calculation.structure.read()["elements"][3]
    'O'
    >>> bool(np.all(heavy[3:].real <= default[3:].real))
    True
    """

    STANDARD_ATOMIC_WEIGHTS = types.MappingProxyType(masses.TABLE)
    """Standard atomic weight of every element in atomic mass units, keyed by the
    chemical symbol. For the elements without a stable isotope it is the mass number
    of the longest-lived one. The mapping is read-only; copy it with ``dict(...)`` to
    modify it."""

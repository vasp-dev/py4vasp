# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Turn discrete peaks into a smooth spectrum.

VASP reports many quantities as a list of positions and intensities: phonon modes and
their infrared or Raman activity, optical transitions and their oscillator strength, the
eigenvalues of a band structure, the neighbours at a given distance. Comparing any of
them with an experiment means giving every peak a width and adding the peaks up.

The line shapes here are normalized to **unit area**. The spectrum therefore integrates
to the total weight, and choosing a different width redistributes it without changing
what it adds up to -- a density of states still integrates to the number of states. If
you want a curve whose highest peak is one, divide by its maximum yourself, so that the
normalization stays visible at the call site.

Nothing here is specific to an energy axis. The mesh, the positions and the width only
have to share a unit, so neighbour distances in Å broaden exactly like frequencies.

Examples
--------
Broaden three Raman-active modes, quoted in wavenumbers, with a Lorentzian of
8 cm⁻¹ full width at half maximum

>>> import numpy as np
>>> from py4vasp.broadening import Lorentzian, broaden
>>> wavenumbers = np.linspace(0, 2000, 4001)
>>> activities = [1.0, 3.5, 8.0]
>>> spectrum = broaden(wavenumbers, [606, 1178, 1600], activities,
...                    shape=Lorentzian(fwhm=8))
>>> spectrum.shape
(4001,)

Use a :class:`Gaussian` instead when the broadening is instrumental rather than a
lifetime. Its weight is conserved on a mesh this wide, whereas a Lorentzian always
leaves a little of its algebraic tails outside

>>> from py4vasp.broadening import Gaussian
>>> spectrum = broaden(wavenumbers, [606, 1178, 1600], activities,
...                    shape=Gaussian(fwhm=8))
>>> round(float(np.trapezoid(spectrum, wavenumbers)), 10)
12.5

Leading axes are kept, so a set of bands becomes one spectrum each in a single call

>>> eigenvalues = np.array([[-1.0, -0.8], [0.2, 0.5], [1.4, 1.9]])
>>> energies = np.linspace(-3, 3, 601)
>>> broaden(energies, eigenvalues, shape=Gaussian(fwhm=0.3)).shape
(3, 601)
"""

from py4vasp._third_party.numeric import Gaussian, Lorentzian, broaden

__all__ = ["broaden", "Gaussian", "Lorentzian"]

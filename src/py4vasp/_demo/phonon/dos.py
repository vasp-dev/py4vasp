# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def Sr2TiO4():
    energies = np.linspace(0, 5, _demo.NUMBER_POINTS)
    dos = energies**2
    lower_ratio = np.arange(_demo.NUMBER_MODES, dtype=np.float64)
    lower_ratio = lower_ratio.reshape(_demo.AXES, _demo.NUMBER_ATOMS)
    lower_ratio /= np.sum(lower_ratio)
    upper_ratio = np.array(list(reversed(lower_ratio)))
    ratio = np.linspace(lower_ratio, upper_ratio, _demo.NUMBER_POINTS).T
    projections = np.multiply(ratio, dos)
    return raw.PhononDos(energies, dos, projections, _demo.stoichiometry.Sr2TiO4())

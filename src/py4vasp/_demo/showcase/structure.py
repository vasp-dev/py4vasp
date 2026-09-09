# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw
from py4vasp._demo import showcase
from py4vasp._demo.showcase import cell

# Distortion of every atom at the first step in fractional coordinates, which the
# relaxation removes. Each column sums to zero, so the crystal as a whole does not drift
# and the forces of every step can balance.
_SR2TIO4_DISTORTION = (
    (0.010, 0.010, 0.000),  # Sr
    (-0.010, -0.010, 0.000),  # Sr
    (0.000, 0.000, 0.000),  # Ti
    (0.006, 0.006, 0.004),  # O apical
    (-0.006, -0.006, -0.004),  # O apical
    (0.005, -0.005, 0.003),  # O equatorial
    (-0.005, 0.005, -0.003),  # O equatorial
)


# Positions Sr2TiO4 relaxes onto: Sr and the apical O on the free Wyckoff position 4e of
# I4/mmm, Ti on 2a and the equatorial O on 4c. Each 4e pair sums to exactly one, which
# py4vasp._demo.structure.Sr2TiO4 misses by 1e-5 for the apical oxygens, one order of
# magnitude above the tolerance the symmetry analysis uses -- so that structure resolves
# as Cm rather than I4/mmm. The showcase carries the symmetric value instead.
_SR2TIO4_POSITIONS = (
    (0.64529, 0.64529, 0.0),  # Sr
    (0.35471, 0.35471, 0.0),  # Sr
    (0.00000, 0.00000, 0.0),  # Ti
    (0.84178, 0.84178, 0.0),  # O apical
    (0.15822, 0.15822, 0.0),  # O apical
    (0.50000, 0.00000, 0.5),  # O equatorial
    (0.00000, 0.50000, 0.5),  # O equatorial
)


def ideal_positions() -> np.ndarray:
    """Positions Sr2TiO4 relaxes onto, the Wyckoff positions of its space group."""
    return np.array(_SR2TIO4_POSITIONS)


def distortion() -> np.ndarray:
    """Deviation from the ideal positions at every step, shape ``(step, atom, axis)``."""
    return showcase.converge(np.array(_SR2TIO4_DISTORTION), np.zeros((7, 3)))


def Sr2TiO4() -> raw.Structure:
    """Ionic relaxation of Sr2TiO4 onto the ideal positions of its space group.

    The distortion decays to exactly zero, so the final structure is symmetric to the
    precision the symmetry analysis works with rather than merely close to it.
    """
    return raw.Structure(
        stoichiometry=_demo.stoichiometry.Sr2TiO4(has_ion_types=True),
        cell=cell.Sr2TiO4(),
        positions=_demo.wrap_data(ideal_positions() + distortion()),
    )

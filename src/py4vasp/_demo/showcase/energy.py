# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw
from py4vasp._demo import showcase

MINIMUM = -42.5  # eV, the free energy the relaxation converges onto
EXCESS = 0.35  # eV the first step is above the minimum
# offset of the other two energies from the free energy; a finite smearing puts the
# energy without entropy above it and extrapolates energy(sigma->0) in between
_OFFSETS = (0.0, 0.018, 0.009)
_LABELS = (
    "free energy    TOTEN   ",
    "energy without entropy ",
    "energy(sigma->0)       ",
)


def relax() -> raw.Energy:
    """Energies along the showcase relaxation, converging onto the minimum.

    The excess energy decays with the square of the remaining distortion, as it does in
    a harmonic well, so the energy converges twice as fast as the forces do.
    """
    excess = EXCESS * showcase.decay() ** 2
    values = MINIMUM + np.add.outer(excess, np.array(_OFFSETS))
    return raw.Energy(
        labels=_demo.wrap_data(np.array(_LABELS, dtype="S")),
        values=_demo.wrap_data(values),
    )

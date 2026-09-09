# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw
from py4vasp._demo import showcase
from py4vasp._demo.showcase import structure

# Stress of the first step in kBar. The cell starts compressed, so the diagonal is
# negative, and the two in-plane components are equal because the crystal is tetragonal.
_INITIAL_STRESS = (
    (-2.40, 0.00, 0.15),
    (0.00, -2.40, 0.15),
    (0.15, 0.15, -1.10),
)


def Sr2TiO4() -> raw.Stress:
    """Stress along the showcase relaxation, released as the cell expands onto its size."""
    stresses = showcase.converge(np.array(_INITIAL_STRESS), np.zeros((3, 3)))
    return raw.Stress(structure=structure.Sr2TiO4(), stress=_demo.wrap_data(stresses))

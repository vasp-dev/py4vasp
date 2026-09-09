# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw
from py4vasp._demo.showcase import force, structure

DAMPING = 0.04  # A per (eV/A), the step a damped relaxation takes per unit of force


def Sr2TiO4() -> raw.Velocity:
    """Velocities along the showcase relaxation.

    A damped relaxation moves every atom along the force acting on it, so the velocities
    point towards the ideal positions and die away together with the forces.
    """
    velocities = -DAMPING * force.cartesian_distortion() * force.FORCE_CONSTANT
    return raw.Velocity(
        structure=structure.Sr2TiO4(), velocities=_demo.wrap_data(velocities)
    )

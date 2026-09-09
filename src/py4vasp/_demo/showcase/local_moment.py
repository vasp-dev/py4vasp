# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw
from py4vasp._demo import showcase
from py4vasp._demo.showcase import structure

# Charge of every atom in the s, p and d shell. Iron carries its charge in the d shell,
# oxygen keeps its 2p shell nearly full.
_IRON_CHARGE = (0.42, 0.51, 6.24)
_OXYGEN_CHARGE = (1.78, 4.42, 0.09)
# Magnetic moment of the s, p and d shell. The tetrahedral iron orders antiparallel to
# the octahedral iron, and the two sublattices do not cancel: what is left is the
# 4 Bohr magnetons per formula unit that magnetite is measured to carry, so eight for the
# two formula units of the primitive cell.
_TETRAHEDRAL_MOMENT = (-0.04, -0.11, -4.85)
_OCTAHEDRAL_MOMENT = (0.03, 0.09, 4.28)
_OXYGEN_MOMENT = (0.002, 0.046, 0.002)


def Fe3O4() -> raw.LocalMoment:
    """Charges and magnetic moments of magnetite over the showcase relaxation.

    The moments grow onto their converged values as the relaxation settles, the way the
    magnetization of a spin-polarized calculation does.
    """
    converged = np.array(
        2 * [_TETRAHEDRAL_MOMENT] + 4 * [_OCTAHEDRAL_MOMENT] + 8 * [_OXYGEN_MOMENT]
    )
    charges = np.array(6 * [_IRON_CHARGE] + 8 * [_OXYGEN_CHARGE])
    # the moments start out too small, as they do before the magnetization is converged
    moments = showcase.converge(0.75 * converged, converged)
    steps = showcase.NUMBER_STEPS
    spin_moments = np.stack((np.broadcast_to(charges, (steps, 14, 3)), moments), axis=1)
    return raw.LocalMoment(
        structure=structure.Fe3O4(),
        spin_moments=_demo.wrap_data(spin_moments),
    )

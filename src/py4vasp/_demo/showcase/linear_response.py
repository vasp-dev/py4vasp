# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import raw
from py4vasp._demo.showcase import structure

# Born effective charges of the showcase Sr2TiO4 in the order of its POSCAR, close to the
# values of the related SrTiO3. Every site has a symmetry that makes its tensor diagonal.
# The oxygens in the TiO2 plane carry the anomalously large charge along their Ti-O bond,
# which is the y axis for the sixth atom and the x axis for the seventh. The apical
# oxygens are chosen so that the charges sum to zero, as a crystal shifted as a whole
# must not feel a force in a field.
_BORN_CHARGES = (
    (2.47, 2.47, 2.69),  # Sr
    (2.47, 2.47, 2.69),  # Sr
    (6.94, 6.94, 5.65),  # Ti
    (-2.21, -2.21, -3.815),  # O apical
    (-2.21, -2.21, -3.815),  # O apical
    (-1.98, -5.48, -1.70),  # O bonded along y
    (-5.48, -1.98, -1.70),  # O bonded along x
)

# The tetragonal axis is z, so the in-plane components agree. Without local field effects
# the electrons screen slightly less, and the polar modes add a large ionic contribution
# that is weaker along the layering direction.
_CLAMPED_ION = (4.62, 4.62, 4.35)
_ION = (32.5, 32.5, 13.8)
_INDEPENDENT_PARTICLE = (5.05, 5.05, 4.80)


def born_effective_charge_Sr2TiO4() -> raw.BornEffectiveCharge:
    """Born effective charges of the showcase Sr2TiO4 obeying the acoustic sum rule."""
    return raw.BornEffectiveCharge(
        structure=structure.Sr2TiO4(),
        charge_tensors=np.array([np.diag(charges) for charges in _BORN_CHARGES]),
    )


def dielectric_tensor_Sr2TiO4() -> raw.DielectricTensor:
    """Static dielectric tensors of the showcase Sr2TiO4 from density-functional theory."""
    return raw.DielectricTensor(
        electron=raw.VaspData(np.diag(_CLAMPED_ION)),
        ion=raw.VaspData(np.diag(_ION)),
        independent_particle=raw.VaspData(np.diag(_INDEPENDENT_PARTICLE)),
        method=b"dft",
        cell=None,
    )

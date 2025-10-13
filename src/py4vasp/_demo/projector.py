# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp import _demo, raw


def Sr2TiO4(use_orbitals):
    orbital_types = "s py pz px dxy dyz dz2 dxz x2-y2 fy3x2 fxyz fyz2 fz3 fxz2 fzx2 fx3"
    return raw.Projector(
        stoichiometry=_demo.stoichiometry.Sr2TiO4(),
        orbital_types=_demo.wrap_orbital_types(use_orbitals, orbital_types),
        number_spin_projections=1,
    )

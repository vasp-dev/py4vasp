# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp import _demo, raw

L_RESOLVED_ORBITALS = "s p d f"
LM_RESOLVED_ORBITALS = (
    "s py pz px dxy dyz dz2 dxz x2-y2 fy3x2 fxyz fyz2 fz3 fxz2 fzx2 fx3"
)


def Sr2TiO4(use_orbitals):
    return raw.Projector(
        stoichiometry=_demo.stoichiometry.Sr2TiO4(),
        orbital_types=_demo.wrap_orbital_types(use_orbitals, LM_RESOLVED_ORBITALS),
        number_spin_projections=_demo.NONPOLARIZED,
    )


def Fe3O4(use_orbitals):
    return raw.Projector(
        stoichiometry=_demo.stoichiometry.Fe3O4(),
        orbital_types=_demo.wrap_orbital_types(use_orbitals, L_RESOLVED_ORBITALS),
        number_spin_projections=_demo.COLLINEAR,
    )


def Ba2PbO4(use_orbitals):
    return raw.Projector(
        stoichiometry=_demo.stoichiometry.Ba2PbO4(),
        orbital_types=_demo.wrap_orbital_types(use_orbitals, L_RESOLVED_ORBITALS),
        number_spin_projections=_demo.NONCOLLINEAR,
    )

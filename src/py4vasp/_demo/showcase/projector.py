# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp import _demo, raw
from py4vasp._demo.showcase import stoichiometry


def Fe3O4(use_orbitals, magnetism="collinear") -> raw.Projector:
    """Projectors of magnetite, resolved by angular momentum and by spin."""
    spin_projections = (
        _demo.COLLINEAR if magnetism == "collinear" else _demo.NONCOLLINEAR
    )
    return raw.Projector(
        stoichiometry=stoichiometry.Fe3O4(),
        orbital_types=_demo.wrap_orbital_types(
            use_orbitals, _demo.projector.L_RESOLVED_ORBITALS
        ),
        number_spin_projections=spin_projections,
    )

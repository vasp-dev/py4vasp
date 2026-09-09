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
# Polar and azimuthal angle in degrees of the moment of every atom, for a calculation
# that resolves the direction. The tetrahedral iron already points against the
# octahedral iron, so the collinear sign is folded into the length of the moment and the
# canting only turns each moment away from the z axis.
_CANTING = 2 * [(22.0, 35.0)] + 4 * [(18.0, 30.0)] + 8 * [(25.0, 40.0)]


def Fe3O4(magnetism="collinear") -> raw.LocalMoment:
    """Charges and magnetic moments of magnetite over the showcase relaxation.

    The moments grow onto their converged values as the relaxation settles, the way the
    magnetization of a spin-polarized calculation does.

    Parameters
    ----------
    magnetism
        Pass ``"noncollinear"`` to resolve the moment of every atom along the three
        axes, and to add the orbital moments a calculation with spin-orbit coupling has.
    """
    converged = np.array(
        2 * [_TETRAHEDRAL_MOMENT] + 4 * [_OCTAHEDRAL_MOMENT] + 8 * [_OXYGEN_MOMENT]
    )
    charges = np.array(6 * [_IRON_CHARGE] + 8 * [_OXYGEN_CHARGE])
    # the moments start out too small, as they do before the magnetization is converged
    moments = showcase.converge(0.75 * converged, converged)
    steps = showcase.NUMBER_STEPS
    charges = np.broadcast_to(charges, (steps, 14, 3))
    magnetization = _magnetization(moments, magnetism)
    # VASP stores the components along the second axis, after the steps
    spin_moments = np.moveaxis(np.concatenate(([charges], magnetization)), 0, 1)
    moment = raw.LocalMoment(
        structure=structure.Fe3O4(),
        spin_moments=_demo.wrap_data(spin_moments),
    )
    if magnetism == "noncollinear":
        # spin-orbit coupling induces an orbital moment of a few percent of the spin one,
        # which VASP reports separately and without an s contribution
        orbital = np.moveaxis(0.06 * magnetization, 0, 1)
        moment.orbital_moments = _demo.wrap_data(orbital[:, :, :, 1:])
    return moment


def _magnetization(moments, magnetism):
    """Magnetization along every axis it is resolved on, ``(axis, step, atom, orbital)``."""
    if magnetism == "collinear":
        return moments[np.newaxis]
    # every atom keeps the length of its moment and turns it away from the z axis, so the
    # two sublattices differ by a direction rather than by a sign
    return np.einsum("saq,ax->xsaq", moments, _canted_directions())


def _canted_directions():
    """Unit vector the moment of every atom points along, shape ``(atom, axis)``."""
    polar, azimuthal = np.radians(np.array(_CANTING)).T
    return np.array(
        [
            np.sin(polar) * np.cos(azimuthal),
            np.sin(polar) * np.sin(azimuthal),
            np.cos(polar),
        ]
    ).T

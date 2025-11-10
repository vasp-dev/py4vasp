# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._calculation import base


class EffectiveCoulomb(base.Refinery):
    """Effective Coulomb interaction U obtained with the constrained random phase approximation (cRPA).

    This class provides post-processing routines to read and visualize first-principles
    results from constrained Random Phase Approximation (cRPA) calculations. After you
    have performed a cRPA calculation using VASP this class can visualize the effective
    Coulomb interaction *U* along the radial or frequency axis. Youy can use this *U*
    mean-field theories like DFT+*U* and Dynamical Mean Field Theory (DMFT).

    The cRPA method is essential for strongly correlated materials, where standard Density
    Functional Theory (DFT) often incorrectly predicts a metallic ground state or fails to
    capture magnetic order. You can activate the cRPA calculation in VASP by setting
    :tag:`ALGO` = `CRPAR` in the INCAR file. The method computes the effective Coulomb
    interaction *U* in real space by excluding screening processes within a predefined
    correlated subspace, typically associated with localized orbitals such as *d* or *f*
    states.

    While different flavors of cRPA exist, we recommend using the spectral cRPA (s-cRPA)
    method that you activate by setting :tag:`LSCRPA` = `.TRUE.`. in the INCAR file. This
    approach overcomes significant limitations of earlier cRPA formulations [1]_, in
    particular numerical instabilities for highly occupied correlated shells or unphysical
    results like negative *U* values.

    References
    ----------
    .. [1] Kaltak, M., *et al.*, Constrained Random Phase Approximation: the spectral
        method, https://arxiv.org/abs/2508.15368, 2025.
    """

    def to_dict(self):
        return {}

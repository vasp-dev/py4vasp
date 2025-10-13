# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def single_band():
    dispersion = _demo.dispersion.single_band()
    return raw.Band(
        dispersion=dispersion,
        fermi_energy=0.0,
        occupations=np.array([np.linspace([1], [0], dispersion.eigenvalues.size)]),
        projectors=_demo.projector.Sr2TiO4(use_orbitals=False),
    )

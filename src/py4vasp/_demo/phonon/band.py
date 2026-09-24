# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def Sr2TiO4():
    dispersion = _demo.dispersion.phonon()
    shape = (
        *dispersion.eigenvalues.shape,
        _demo.NUMBER_ATOMS,
        _demo.AXES,
        _demo.COMPLEX,
    )
    return raw.PhononBand(
        dispersion=dispersion,
        stoichiometry=_demo.stoichiometry.Sr2TiO4(),
        eigenvectors=np.linspace(0, 1, np.prod(shape)).reshape(shape),
    )

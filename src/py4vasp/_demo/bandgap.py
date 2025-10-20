# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def bandgap(selection):
    labels = (
        "valence band maximum",
        "conduction band minimum",
        "direct gap bottom",
        "direct gap top",
        "Fermi energy",
        "kx (VBM)",
        "ky (VBM)",
        "kz (VBM)",
        "kx (CBM)",
        "ky (CBM)",
        "kz (CBM)",
        "kx (direct)",
        "ky (direct)",
        "kz (direct)",
    )
    num_components = 3 if selection == "spin_polarized" else 1
    shape = (_demo.NUMBER_STEPS, num_components, len(labels))
    data = np.sqrt(np.arange(np.prod(shape)).reshape(shape))
    if num_components == 3:
        # only spin-independent Fermi energy implemented
        data[:, 1, 4] = data[:, 0, 4]
        data[:, 2, 4] = data[:, 0, 4]
    return raw.Bandgap(labels=np.array(labels, dtype="S"), values=data)

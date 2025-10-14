# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import raw


def Sr2TiO4(has_ion_types=True):
    if has_ion_types:
        return raw.Stoichiometry(
            number_ion_types=np.array((2, 1, 4)),
            ion_types=raw.VaspData(np.array(("Sr", "Ti", "O "), dtype="S")),
        )
    else:
        return raw.Stoichiometry(
            number_ion_types=raw.VaspData(np.array((2, 1, 4))),
            ion_types=raw.VaspData(None),
        )


def Fe3O4():
    return raw.Stoichiometry(
        number_ion_types=np.array((3, 4)), ion_types=np.array(("Fe", "O "), dtype="S")
    )


def Ba2PbO4():
    return raw.Stoichiometry(
        number_ion_types=np.array((2, 1, 4)),
        ion_types=raw.VaspData(np.array(("Ba", "Pb", "O "), dtype="S")),
    )


def Ca3AsBr3():
    return raw.Stoichiometry(
        number_ion_types=np.array((2, 1, 1, 1, 2)),
        ion_types=np.array(("Ca", "As", "Br", "Ca", "Br"), dtype="S"),
    )

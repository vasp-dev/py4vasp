# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import raw


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


def CaAs3_110():
    return raw.Stoichiometry(
        number_ion_types=np.array((6, 18)),
        ion_types=np.array(("Ca", "As"), dtype="S"),
    )


def Fe3O4():
    return raw.Stoichiometry(
        number_ion_types=np.array((3, 4)), ion_types=np.array(("Fe", "O "), dtype="S")
    )


def Graphite():
    return raw.Stoichiometry(
        number_ion_types=np.array((10,)),
        ion_types=np.array(("C",), dtype="S"),
    )


def Ni100():
    return raw.Stoichiometry(
        number_ion_types=np.array((5,)),
        ion_types=np.array(("Ni",), dtype="S"),
    )


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

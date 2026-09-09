# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import raw


def Fe3O4() -> raw.Stoichiometry:
    """Six iron and eight oxygen atoms, the primitive cell of the spinel.

    py4vasp._demo.stoichiometry.Fe3O4 describes seven atoms instead, because the test
    data pairs it with a made-up cell of one formula unit; a spinel needs fourteen.
    """
    return raw.Stoichiometry(
        number_ion_types=np.array((6, 8)),
        ion_types=raw.VaspData(np.array(("Fe", "O "), dtype="S")),
    )

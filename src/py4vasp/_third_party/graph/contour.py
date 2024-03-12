# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses

import numpy as np


@dataclasses.dataclass
class Contour:
    data: np.array
    lattice: np.array
    label: str

    def _generate_traces(self):
        return []

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses

import numpy as np
import plotly.graph_objects as go


@dataclasses.dataclass
class Contour:
    data: np.array
    lattice: np.array
    label: str
    supercell: np.array = (1, 1)
    "multiple of each lattice to be drawn"

    def _generate_traces(self):
        yield go.Heatmap(z=self.data), {}

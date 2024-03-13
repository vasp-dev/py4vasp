# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
import itertools

import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

from py4vasp import _config


@dataclasses.dataclass
class Contour:
    interpolation_factor = 2

    data: np.array
    lattice: np.array
    label: str
    supercell: np.array = (1, 1)
    "multiple of each lattice to be drawn"
    show_cell: bool = True

    def _generate_traces(self):
        lattice_supercell = np.diag(self.supercell) @ self.lattice
        data = np.tile(self.data, self.supercell)
        if self._interpolation_required():
            x, y, z = self._interpolate_data(lattice_supercell, data)
        else:
            x, y, z = self._use_data_without_interpolation(lattice_supercell, data)
        yield go.Heatmap(x=x, y=y, z=z, name=self.label), {}

    def _interpolation_required(self):
        return not np.allclose((self.lattice[1, 0], self.lattice[0, 1]), 0)

    def _interpolate_data(self, lattice, data):
        area_cell = abs(np.cross(lattice[0], lattice[1]))
        points_per_area = data.size / area_cell
        points_per_line = np.sqrt(points_per_area) * self.interpolation_factor
        lengths = np.sum(np.abs(lattice), axis=0)
        shape = np.ceil(points_per_line * lengths).astype(int)
        line_mesh_a = np.linspace(0, lattice[0], data.shape[0], endpoint=False)
        line_mesh_b = np.linspace(0, lattice[1], data.shape[1], endpoint=False)
        # this order is required so that itertools runs over a first
        x_in, y_in = np.array(
            [a + b for b, a in itertools.product(line_mesh_b, line_mesh_a)]
        ).T
        z_in = data.flatten()
        x_out, y_out = np.meshgrid(
            np.linspace(x_in.min(), x_in.max(), shape[0]),
            np.linspace(y_in.min(), y_in.max(), shape[1]),
        )
        interpolated_data = griddata((x_in, y_in), z_in, (x_out, y_out), method="cubic")
        return x_out[0], y_out[:, 0], interpolated_data

    def _use_data_without_interpolation(self, lattice, data):
        x = np.linspace(0, lattice[0, 0], data.shape[0], endpoint=False)
        y = np.linspace(0, lattice[1, 1], data.shape[1], endpoint=False)
        # plotly expects y-x order for data
        return x, y, data.T

    def _generate_shapes(self):
        if not self.show_cell:
            return ()
        pos_to_str = lambda pos: f"{pos[0]} {pos[1]}"
        corners = (self.lattice[0], self.lattice[0] + self.lattice[1], self.lattice[1])
        to_corners = (f"L {pos_to_str(corner)}" for corner in corners)
        path = f"M 0 0 {' '.join(to_corners)} Z"
        yield {"type": "path", "line": {"color": _config.VASP_GRAY}, "path": path}

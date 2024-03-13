# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses

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
        # swap a and b axes because that is the way plotly expects the data
        data = np.tile(self.data, self.supercell).T
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
        line_mesh_a = self._make_mesh(lattice, data.shape[1], 0)
        line_mesh_b = self._make_mesh(lattice, data.shape[0], 1)
        x_in, y_in = (line_mesh_a[:, np.newaxis] + line_mesh_b[np.newaxis, :]).T
        x_in = x_in.flatten()
        y_in = y_in.flatten()
        z_in = data.flatten()
        x_out, y_out = np.meshgrid(
            np.linspace(x_in.min(), x_in.max(), shape[0]),
            np.linspace(y_in.min(), y_in.max(), shape[1]),
        )
        z_out = griddata((x_in, y_in), z_in, (x_out, y_out), method="cubic")
        return x_out[0], y_out[:, 0], z_out

    def _use_data_without_interpolation(self, lattice, data):
        x = self._make_mesh(lattice, data.shape[1], 0)
        y = self._make_mesh(lattice, data.shape[0], 1)
        return x, y, data

    def _make_mesh(self, lattice, num_point, index):
        vector = index if self._interpolation_required() else (index, index)
        return (
            np.linspace(0, lattice[vector], num_point, endpoint=False)
            + 0.5 * lattice[vector] / num_point
        )

    def _generate_shapes(self):
        if not self.show_cell:
            return ()
        pos_to_str = lambda pos: f"{pos[0]} {pos[1]}"
        corners = (self.lattice[0], self.lattice[0] + self.lattice[1], self.lattice[1])
        to_corners = (f"L {pos_to_str(corner)}" for corner in corners)
        path = f"M 0 0 {' '.join(to_corners)} Z"
        yield {"type": "path", "line": {"color": _config.VASP_GRAY}, "path": path}

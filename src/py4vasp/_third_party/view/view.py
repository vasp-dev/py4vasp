# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Dict

import numpy as np

from py4vasp._util import import_

ase = import_.optional("ase")
ase_cube = import_.optional("ase.io.cube")
nglview = import_.optional("nglview")

CUBE_FILENAME = "quantity.cube"


@dataclass
class View:
    number_ion_types: array
    ion_types: array
    lattice_vectors: array
    positions: array

    grid_scalars: Dict = None

    def _ipython_display_(self):
        widget = self.to_ngl()
        widget._ipython_display_()

    def _create_atoms(self, idx_traj):
        ions_and_ion_types = (
            self.number_ion_types[idx_traj],
            self.ion_types[idx_traj],
        )
        symbols = [
            ion_type * number_ion_type
            for ion_type, number_ion_type in zip(*ions_and_ion_types)
        ]
        symbols = "".join(symbols)
        atoms = ase.Atoms(symbols)
        atoms.cell = self.lattice_vectors[idx_traj]
        atoms.set_scaled_positions(self.positions[idx_traj])
        atoms.set_pbc(True)
        return atoms

    def to_ngl(self):
        trajectory = []
        for idx_traj in range(len(self.number_ion_types)):
            atoms = self._create_atoms(idx_traj)
            trajectory.append(atoms)
        ngl_trajectory = nglview.ASETrajectory(trajectory)
        widget = nglview.NGLWidget(ngl_trajectory)
        return widget

    def show_isosurface(self, quantity):
        widget = self.to_ngl()
        data = self.grid_scalars[quantity][0, ...].astype(np.float32)
        atoms = self._create_atoms(-1)
        with tempfile.TemporaryDirectory() as tmp:
            filename = os.path.join(tmp, CUBE_FILENAME)
            ase_cube.write_cube(open(filename, "w"), atoms=atoms, data=data)
            widget.add_component(filename)
        return widget

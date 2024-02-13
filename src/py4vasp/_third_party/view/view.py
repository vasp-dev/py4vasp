# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import annotations

from dataclasses import dataclass

from py4vasp._util import import_

ase = import_.optional("ase")
nglview = import_.optional("nglview")


@dataclass
class View:
    number_ion_types: array
    ion_types: array
    lattice_vectors: array
    positions: array

    def _ipython_display_(self):
        widget = self.to_ngl()
        widget._ipython_display_()

    def to_ngl(self):
        trajectory = []
        for idx_traj in range(len(self.number_ion_types)):
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
            trajectory.append(atoms)
        ngl_trajectory = nglview.ASETrajectory(trajectory)
        widget = nglview.NGLWidget(ngl_trajectory)
        return widget

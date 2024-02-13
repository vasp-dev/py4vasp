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
        ions_and_ion_types = self.number_ion_types[0], self.ion_types[0]
        symbols = [
            ion_type * number_ion_type
            for ion_type, number_ion_type in zip(*ions_and_ion_types)
        ]
        symbols = "".join(symbols)
        atoms = ase.Atoms(symbols)
        atoms.set_scaled_positions(self.positions[0])
        atoms.cell = self.lattice_vectors[0]
        atoms.set_pbc(True)
        widget = nglview.show_ase(atoms)
        return widget

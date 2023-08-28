from dataclasses import dataclass

import numpy as np

from py4vasp._raw.data import Cell
from py4vasp._raw.data_wrapper import VaspData


@dataclass
class ParsePoscar:
    poscar: str

    def __post_init__(self):
        self.split_poscar = self.poscar.split("\n")

    @property
    def comment_line(self):
        return self.split_poscar[0]

    @property
    def cell(self):
        scaling_factor = self.split_poscar[1]
        scaling_factor = np.array(scaling_factor.split(), dtype=float)
        if len(scaling_factor) == 1:
            scaling_factor = scaling_factor[0]
        lattice_vectors = np.array(
            [x.split() for x in self.split_poscar[2:5]], dtype=float
        )
        lattice_vectors = VaspData(lattice_vectors)
        cell = Cell(lattice_vectors=lattice_vectors, scale=scaling_factor)
        return cell

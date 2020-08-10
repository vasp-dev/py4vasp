from py4vasp.data import _util
from py4vasp.data import Viewer3d
import ase
import numpy as np


class Structure:
    def __init__(self, raw_structure):
        self._raw = raw_structure

    def read(self):
        return self.to_dict()

    def to_dict(self):
        return {
            "cell": self._raw.cell.lattice_vectors[:],
            "cartesian_positions": self._raw.cartesian_positions[:],
            "species": self._raw.species,
        }

    def __len__(self):
        return len(self._raw.cartesian_positions)

    def to_ase(self, supercell=None):
        data = self.to_dict()
        species = [_util.decode_if_possible(sp) for sp in data["species"]]
        structure = ase.Atoms(
            symbols=species,
            cell=data["cell"],
            positions=data["cartesian_positions"],
            pbc=True,
        )
        if supercell is not None:
            structure *= supercell
        return structure

    def plot(self, *args):
        return self.to_ngl(*args)

    def to_ngl(self, supercell=None):
        viewer = Viewer3d.from_structure(self, supercell=supercell)
        viewer.show_cell()
        return viewer

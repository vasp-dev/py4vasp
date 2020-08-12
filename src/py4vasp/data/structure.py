from py4vasp.data import _util, Viewer3d, Topology
import ase
import numpy as np


class Structure:
    def __init__(self, raw_structure):
        self._raw = raw_structure

    @classmethod
    def from_file(cls, file=None):
        return _util.from_file(cls, file, "structure")

    def read(self):
        return self.to_dict()

    def to_dict(self):
        return {
            "cell": self._raw.cell.scale * self._raw.cell.lattice_vectors[:],
            "positions": self._raw.positions[:],
            "elements": Topology(self._raw.topology).elements(),
        }

    def __len__(self):
        return len(self._raw.positions)

    def to_ase(self, supercell=None):
        data = self.to_dict()
        structure = ase.Atoms(
            symbols=data["elements"],
            cell=data["cell"],
            scaled_positions=data["positions"],
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

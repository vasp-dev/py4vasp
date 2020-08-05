from py4vasp.data import _util
from py4vasp.data import Viewer3d


class Structure:
    def __init__(self, raw_structure):
        self._raw = raw_structure

    def read(self):
        return self.to_dict()

    def to_dict(self):
        return {
            "cell": self._raw.cell.lattice_vectors[:],
            "cartesian_positions": self._raw.cartesian_positions[:],
            "species": list(self._raw.species),
        }

    def __len__(self):
        return len(self._raw.cartesian_positions)

    def to_pymatgen(self):
        import pymatgen as mg

        return mg.Structure(
            lattice=mg.Lattice(self._raw.cell.lattice_vectors),
            species=[specie.decode("ascii") for specie in self._raw.species],
            coords=self._raw.cartesian_positions,
            coords_are_cartesian=True,
        )

    def plot(self, supercell=None):
        viewer = Viewer3d(self, supercell=supercell)
        viewer.show_cell()
        return viewer

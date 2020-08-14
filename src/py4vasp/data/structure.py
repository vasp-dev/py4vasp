from py4vasp.data import _util, Viewer3d, Topology
import ase
import numpy as np
import functools


class Structure:
    """ The structure of the crystal.

    You can use this class to process structural information from the Vasp
    calculation. Typically you want to do this to inspect the converged structure
    after an ionic relaxation.

    Parameters
    ----------
    raw_structure : raw.Structure
        Dataclass containing the raw data defining the structure.
    """

    def __init__(self, raw_structure):
        self._raw = raw_structure

    @classmethod
    @_util.add_doc(_util.from_file_doc("crystal structure"))
    def from_file(cls, file=None):
        return _util.from_file(cls, file, "structure")

    def to_dict(self):
        """ Read the structual information into a dictionary.

        Returns
        -------
        dict
            Contains the unit cell of the crystal, as well as the position of
            all the atoms in units of the lattice vectors and the elements of
            the atoms.
        """
        return {
            "cell": self._raw.cell.scale * self._raw.cell.lattice_vectors[:],
            "positions": self._raw.positions[:],
            "elements": Topology(self._raw.topology).elements(),
        }

    @functools.wraps(to_dict)
    def read(self):
        return self.to_dict()

    def __len__(self):
        return len(self._raw.positions)

    def to_ase(self, supercell=None):
        """ Convert the structure to an ase Atoms object.

        Parameters
        ----------
        supercell : int or np.ndarray
            If present the structure is replicated the specified number of times
            along each direction.

        Returns
        -------
        ase.Atoms
            Structural information for ase package.
        """
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

    def to_viewer3d(self, supercell=None):
        """ Generate a 3d representation of the structure.

        Parameters
        ----------
        supercell : int or np.ndarray
            If present the structure is replicated the specified number of times
            along each direction.

        Returns
        -------
        Viewer3d
            Visualize the structure as a 3d figure.
        """
        viewer = Viewer3d.from_structure(self, supercell=supercell)
        viewer.show_cell()
        return viewer

    @functools.wraps(to_viewer3d)
    def plot(self, *args):
        return self.to_viewer3d(*args)

from py4vasp.data import _util, Viewer3d, Topology, Magnetism
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

    length_moments = 1.5
    "Length in Å how a magnetic moment is displayed relative to the largest moment."

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
            **self._read_magnetic_moments(),
        }

    @functools.wraps(to_dict)
    def read(self):
        return self.to_dict()

    def _read_magnetic_moments(self):
        if self._raw.magnetism is not None:
            magnetism = Magnetism(self._raw.magnetism)
            return {"magnetic_moments": magnetism.total_moments(-1)}
        else:
            return {}

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
        if "magnetic_moments" in data:
            structure.set_initial_magnetic_moments(data["magnetic_moments"])
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
            Visualize the structure as a 3d figure, adding the magnetic momements
            of the atoms if present.
        """
        viewer = Viewer3d.from_structure(self, supercell=supercell)
        viewer.show_cell()
        moments = self._prepare_magnetic_moments_for_plotting()
        if moments is not None:
            viewer.show_arrows_at_atoms(moments)
        return viewer

    @functools.wraps(to_viewer3d)
    def plot(self, *args):
        return self.to_viewer3d(*args)

    def _prepare_magnetic_moments_for_plotting(self,):
        if self._raw.magnetism is None:
            return None
        moments = self._read_magnetic_moments()["magnetic_moments"]
        moments = self._convert_to_moment_to_3d_vector(moments)
        rescale_moments = self.length_moments / np.max(np.linalg.norm(moments, axis=1))
        return rescale_moments * moments

    def _convert_to_moment_to_3d_vector(self, moments):
        if moments.ndim == 2:
            return moments
        moments = moments.reshape((len(moments), 1))
        no_new_moments = (0, 0)
        add_zero_for_xy_axis = (2, 0)
        return np.pad(moments, (no_new_moments, add_zero_for_xy_axis))

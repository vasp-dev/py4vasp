from py4vasp.data import _util, Viewer3d, Topology, Magnetism
from IPython.lib.pretty import pretty
from dataclasses import dataclass
import py4vasp.exceptions as exception
import ase
import numpy as np
import functools


@dataclass
class _Format:
    begin: str = ""
    separator: str = " "
    row: str = "\n"
    end: str = ""
    newline: str = ""


@_util.add_specific_wrappers({"plot": "to_viewer3d"})
class Structure(_util.Data):
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
        super().__init__(raw_structure)

    @classmethod
    @_util.add_doc(_util.from_file_doc("crystal structure"))
    def from_file(cls, file=None):
        return _util.from_file(cls, file, "structure")

    def _repr_pretty_(self, p, cycle):
        p.text(self._create_repr(format_=_Format()))

    def _repr_html_(self):
        format_ = _Format(
            begin="<table>\n<tr><td>",
            separator="</td><td>",
            row="</td></tr>\n<tr><td>",
            end="</td></tr>\n</table>",
            newline="<br>",
        )
        return self._create_repr(format_)

    def to_poscar(self):
        " Generate a string representing this structure usable as a POSCAR file."
        return self._create_repr(format_=_Format())

    def _create_repr(self, format_):
        cell = self._raw.cell.scale * self._raw.cell.lattice_vectors[:]
        vec_to_string = lambda vec: format_.separator.join(str(v) for v in vec)
        vecs_to_string = lambda vecs: format_.row.join(vec_to_string(v) for v in vecs)
        vecs_to_table = lambda vecs: format_.begin + vecs_to_string(vecs) + format_.end
        return f"""
{pretty(Topology(self._raw.topology))}{format_.newline}
1.0{format_.newline}
{vecs_to_table(cell)}
{Topology(self._raw.topology).to_poscar(format_.newline)}{format_.newline}
Direct{format_.newline}
{vecs_to_table(self._raw.positions)}
    """.strip()

    def to_dict(self):
        """ Read the structual information into a dictionary.

        Returns
        -------
        dict
            Contains the unit cell of the crystal, as well as the position of
            all the atoms in units of the lattice vectors and the elements of
            the atoms.
        """
        moments = self._read_magnetic_moments()
        return {
            "cell": self._raw.cell.scale * self._raw.cell.lattice_vectors[:],
            "positions": self._raw.positions[:],
            "elements": Topology(self._raw.topology).elements(),
            **({"magnetic_moments": moments} if moments is not None else {}),
        }

    def _read_magnetic_moments(self):
        if self._raw.magnetism is not None:
            return Magnetism(self._raw.magnetism).total_moments(-1)
        else:
            return None

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
            try:
                structure *= supercell
            except (TypeError, IndexError) as err:
                error_message = (
                    "Generating the supercell failed. Please make sure the requested "
                    "supercell is either an integer or a list of 3 integers."
                )
                raise exception.IncorrectUsage(error_message) from err
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

    def _prepare_magnetic_moments_for_plotting(self):
        moments = self._read_magnetic_moments()
        moments = self._convert_to_moment_to_3d_vector(moments)
        max_length_moments = self._max_length_moments(moments)
        if max_length_moments > 1e-15:
            rescale_moments = self.length_moments / max_length_moments
            return rescale_moments * moments
        else:
            return None

    def _convert_to_moment_to_3d_vector(self, moments):
        if moments is not None and moments.ndim == 1:
            moments = moments.reshape((len(moments), 1))
            no_new_moments = (0, 0)
            add_zero_for_xy_axis = (2, 0)
            moments = np.pad(moments, (no_new_moments, add_zero_for_xy_axis))
        return moments

    def _max_length_moments(self, moments):
        if moments is not None:
            return np.max(np.linalg.norm(moments, axis=1))
        else:
            return 0.0

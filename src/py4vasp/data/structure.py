from py4vasp.data import Viewer3d, Topology
from py4vasp.data._base import RefinementDescriptor
from IPython.lib.pretty import pretty
from collections import Counter
from dataclasses import dataclass
import py4vasp.data._trajectory as _trajectory
import py4vasp.exceptions as exception
import py4vasp.raw as raw
import py4vasp._util.documentation as _documentation
import py4vasp._util.reader as _reader
import py4vasp._util.sanity_check as _check
from py4vasp.raw import RawStructure, RawCell
import ase.io
import io
import numpy as np
import functools
import mdtraj


@dataclass
class _Format:
    begin: str = ""
    separator: str = " "
    row: str = "\n"
    end: str = ""
    newline: str = ""


_structure_docs = f"""
The structure of the crystal for selected steps of the simulation.

You can use this class to process structural information from the Vasp
calculation. Typically you want to do this to inspect the converged structure
after an ionic relaxation or to visualize the changes of the structure along
the simulation.

Parameters
----------
raw_structure : RawStructure
    Dataclass containing the raw data defining the structure.

{_trajectory.trajectory_examples("structure")}
""".strip()


@_documentation.add(_structure_docs)
class Structure(_trajectory.DataTrajectory):

    A_to_nm = 0.1
    "Converting Å to nm used for mdtraj trajectories."

    read = RefinementDescriptor("_to_dict")
    to_dict = RefinementDescriptor("_to_dict")
    plot = RefinementDescriptor("_to_viewer3d")
    to_viewer3d = RefinementDescriptor("_to_viewer3d")
    to_ase = RefinementDescriptor("_to_ase")
    to_mdtraj = RefinementDescriptor("_to_mdtraj")
    to_POSCAR = RefinementDescriptor("_to_poscar")
    cartesian_positions = RefinementDescriptor("_cartesian_positions")
    volume = RefinementDescriptor("_volume")
    __str__ = RefinementDescriptor("_to_string")
    _repr_html_ = RefinementDescriptor("_to_html")
    number_atoms = RefinementDescriptor("_number_atoms")
    number_steps = RefinementDescriptor("_number_steps")

    @classmethod
    def from_POSCAR(cls, poscar):
        """Generate a structure from string in POSCAR format."""
        poscar = io.StringIO(str(poscar))
        structure = ase.io.read(poscar, format="vasp")
        return cls.from_ase(structure)

    @classmethod
    def from_ase(cls, structure):
        """Generate a structure from the ase Atoms class."""
        structure = raw.RawStructure(
            topology=_topology_from_ase(structure),
            cell=_cell_from_ase(structure),
            positions=structure.get_scaled_positions()[np.newaxis],
        )
        return cls(structure)

    def _to_string(self):
        "Generate a string representing the final structure usable as a POSCAR file."
        return self._create_repr()

    def _to_html(self):
        format_ = _Format(
            begin="<table>\n<tr><td>",
            separator="</td><td>",
            row="</td></tr>\n<tr><td>",
            end="</td></tr>\n</table>",
            newline="<br>",
        )
        return self._create_repr(format_)

    def _create_repr(self, format_=_Format()):
        step = self._last_step_in_slice
        vec_to_string = lambda vec: format_.separator.join(str(v) for v in vec)
        vecs_to_string = lambda vecs: format_.row.join(vec_to_string(v) for v in vecs)
        vecs_to_table = lambda vecs: format_.begin + vecs_to_string(vecs) + format_.end
        return f"""
{pretty(self._topology())}{self._step_string()}{format_.newline}
1.0{format_.newline}
{vecs_to_table(self._raw_data.cell.lattice_vectors[step])}
{self._topology().to_poscar(format_.newline)}{format_.newline}
Direct{format_.newline}
{vecs_to_table(self._raw_data.positions[step])}
        """.strip()

    @_documentation.add(
        f"""Read the structual information into a dictionary.

Returns
-------
dict
    Contains the unit cell of the crystal, as well as the position of
    all the atoms in units of the lattice vectors and the elements of
    the atoms for all selected steps.

{_trajectory.trajectory_examples("structure", "read")}"""
    )
    def _to_dict(self):
        return {
            "lattice_vectors": self._lattice_vectors(),
            "positions": self._raw_data.positions[self._steps],
            "elements": self._topology().elements(),
            "names": self._topology().names(),
        }

    @_documentation.add(
        f"""Generate a 3d representation of the structure(s).

Parameters
----------
supercell : int or np.ndarray
    If present the structure is replicated the specified number of times
    along each direction.

Returns
-------
Viewer3d
    Visualize the structure(s) as a 3d figure.

{_trajectory.trajectory_examples("structure", "plot")}"""
    )
    def _to_viewer3d(self, supercell=None):
        if self._is_slice:
            return self._viewer_from_trajectory()
        else:
            return self._viewer_from_structure(supercell)

    @_documentation.add(
        f"""Convert the structure to an ase Atoms object.

Parameters
----------
supercell : int or np.ndarray
    If present the structure is replicated the specified number of times
    along each direction.

Returns
-------
ase.Atoms
    Structural information for ase package.

{_trajectory.trajectory_examples("structure", "to_ase")}"""
    )
    def _to_ase(self, supercell=None):
        if self._is_slice:
            message = (
                "Converting multiple structures to ASE trajectories is not implemented."
            )
            raise exception.NotImplemented(message)
        data = self._to_dict()
        structure = ase.Atoms(
            symbols=data["elements"],
            cell=data["lattice_vectors"],
            scaled_positions=data["positions"],
            pbc=True,
        )
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

    @_documentation.add(
        f"""Convert the trajectory to mdtraj.Trajectory

Returns
-------
mdtraj.Trajectory
    The mdtraj package offers many functionalities to analyze a MD
    trajectory. By converting the Vasp data to their format, we facilitate
    using all functions of that package.

{_trajectory.trajectory_examples("structure", "to_mdtraj")}"""
    )
    def _to_mdtraj(self):
        if not self._is_slice:
            message = "Converting a single structure to mdtraj is not implemented."
            raise exception.NotImplemented(message)
        data = self._to_dict()
        xyz = data["positions"] @ data["lattice_vectors"] * self.A_to_nm
        trajectory = mdtraj.Trajectory(
            xyz, Topology(self._raw_data.topology).to_mdtraj()
        )
        trajectory.unitcell_vectors = data["lattice_vectors"] * Structure.A_to_nm
        return trajectory

    @_documentation.add(
        f"""Convert the structure(s) to a POSCAR format

Returns
-------
str or list[str]
    Returns the POSCAR of the current or all selected steps.

{_trajectory.trajectory_examples("structure", "to_POSCAR")}"""
    )
    def _to_poscar(self):
        if not self._is_slice:
            return self._create_repr()
        else:
            message = "Converting multiple structures to a POSCAR is currently not implemented."
            raise exception.NotImplemented(message)

    @_documentation.add(
        f"""Convert the positions from direct coordinates to cartesian ones.

Returns
-------
np.ndarray
    Position of all atoms in cartesian coordinates in Å.

{_trajectory.trajectory_examples("structure", "cartesian_positions")}"""
    )
    def _cartesian_positions(self):
        return self._raw_data.positions[self._steps] @ self._lattice_vectors()

    @_documentation.add(
        f"""Return the volume of the unit cell for the selected steps.

Returns
-------
float or np.ndarray
    The volume(s) of the selected step(s) in Å³.

{_trajectory.trajectory_examples("structure", "volume")}"""
    )
    def _volume(self):
        return np.abs(np.linalg.det(self._lattice_vectors()))

    def _number_atoms(self):
        """Return the total number of atoms in the structure."""
        return self._raw_data.positions.shape[1]

    def _number_steps(self):
        """Return the number of structures in the trajectory."""
        return len(self._raw_data.positions[self._slice])

    def _topology(self):
        return Topology(self._raw_data.topology)

    def _lattice_vectors(self):
        lattice_vectors = _LatticeVectors(self._raw_data.cell.lattice_vectors)
        return lattice_vectors[self._steps]

    def _viewer_from_structure(self, supercell):
        viewer = Viewer3d.from_structure(self, supercell=supercell)
        viewer.show_cell()
        return viewer

    def _viewer_from_trajectory(self):
        viewer = Viewer3d.from_trajectory(self)
        viewer.show_cell()
        return viewer

    def _step_string(self):
        if self._is_slice:
            range_ = range(len(self._raw_data.positions))[self._steps]
            return f" from step {range_.start + 1} to {range_.stop + 1}"
        elif self._steps == -1:
            return ""
        else:
            return f" (step {self._steps + 1})"


class _LatticeVectors(_reader.Reader):
    def error_message(self, key, err):
        key = np.array(key)
        steps = key if key.ndim == 0 else key[0]
        return (
            f"Error reading the lattice vectors. Please check if the steps "
            f"`{steps}` are properly formatted and within the boundaries. "
            "Additionally, you may consider the original error message:\n" + err.args[0]
        )


def _topology_from_ase(structure):
    # TODO: this should be moved to Topology
    ion_types_and_numbers = Counter(structure.get_chemical_symbols())
    return raw.RawTopology(
        number_ion_types=list(ion_types_and_numbers.values()),
        ion_types=list(ion_types_and_numbers.keys()),
    )


def _cell_from_ase(structure):
    return raw.RawCell(lattice_vectors=np.array([structure.get_cell()]))

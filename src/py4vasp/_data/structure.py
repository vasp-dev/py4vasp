# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import io
from collections import Counter
from dataclasses import dataclass

import ase.io
import mdtraj
import numpy as np
from IPython.lib.pretty import pretty

from py4vasp import data, exception, raw
from py4vasp._data import base, slice_, topology
from py4vasp._data.viewer3d import Viewer3d
from py4vasp._util import documentation, reader


@dataclass
class _Format:
    begin_table: str = ""
    column_separator: str = " "
    row_separator: str = "\n"
    end_table: str = ""
    newline: str = ""

    def comment_line(self, topology, step_string):
        return f"{pretty(topology)}{step_string}{self.newline}"

    def scaling_factor(self):
        return f"1.0{self.newline}"

    def ion_list(self, topology):
        return f"{topology.to_POSCAR(self.newline)}{self.newline}"

    def coordinate_system(self):
        return f"Direct{self.newline}"

    def vectors_to_table(self, vectors):
        rows = (self._vector_to_row(vector) for vector in vectors)
        return f"{self.begin_table}{self.row_separator.join(rows)}{self.end_table}"

    def _vector_to_row(self, vector):
        elements = (self._element_to_string(element) for element in vector)
        return self.column_separator.join(elements)

    def _element_to_string(self, element):
        return f"{element:21.16f}"


@documentation.format(examples=slice_.examples("structure"))
class Structure(slice_.Mixin, base.Refinery):
    """The structure of the crystal for selected steps of the simulation.

    You can use this class to process structural information from the Vasp
    calculation. Typically you want to do this to inspect the converged structure
    after an ionic relaxation or to visualize the changes of the structure along
    the simulation.

    {examples}
    """

    A_to_nm = 0.1
    "Converting Å to nm used for mdtraj trajectories."

    @classmethod
    def from_POSCAR(cls, poscar):
        """Generate a structure from string in POSCAR format."""
        poscar = io.StringIO(str(poscar))
        structure = ase.io.read(poscar, format="vasp")
        return cls.from_ase(structure)

    @classmethod
    def from_ase(cls, structure):
        """Generate a structure from the ase Atoms class."""
        structure = raw.Structure(
            topology=topology.raw_topology_from_ase(structure),
            cell=_cell_from_ase(structure),
            positions=structure.get_scaled_positions()[np.newaxis],
        )
        return cls.from_data(structure)

    @base.data_access
    def __str__(self):
        "Generate a string representing the final structure usable as a POSCAR file."
        return self._create_repr()

    @base.data_access
    def _repr_html_(self):
        format_ = _Format(
            begin_table="<table>\n<tr><td>",
            column_separator="</td><td>",
            row_separator="</td></tr>\n<tr><td>",
            end_table="</td></tr>\n</table>",
            newline="<br>",
        )
        return self._create_repr(format_)

    def _create_repr(self, format_=_Format()):
        step = self._last_step_in_slice
        lines = (
            format_.comment_line(self._topology(), self._step_string()),
            format_.scaling_factor(),
            format_.vectors_to_table(self._raw_data.cell.lattice_vectors[step]),
            format_.ion_list(self._topology()),
            format_.coordinate_system(),
            format_.vectors_to_table(self._raw_data.positions[step]),
        )
        return "\n".join(lines)

    @base.data_access
    @documentation.format(examples=slice_.examples("structure", "to_dict"))
    def to_dict(self):
        """Read the structural information into a dictionary.

        Returns
        -------
        dict
            Contains the unit cell of the crystal, as well as the position of
            all the atoms in units of the lattice vectors and the elements of
            the atoms for all selected steps.

        {examples}
        """
        return {
            "lattice_vectors": self._lattice_vectors(),
            "positions": self._raw_data.positions[self._steps],
            "elements": self._topology().elements(),
            "names": self._topology().names(),
        }

    @base.data_access
    @documentation.format(examples=slice_.examples("structure", "to_graph"))
    def plot(self, supercell=None):
        """Generate a 3d representation of the structure(s).

        Parameters
        ----------
        supercell : int or np.ndarray
            If present the structure is replicated the specified number of times
            along each direction.

        Returns
        -------
        Viewer3d
            Visualize the structure(s) as a 3d figure.

        {examples}
        """
        if self._is_slice:
            return self._viewer_from_trajectory()
        else:
            return self._viewer_from_structure(supercell)

    @base.data_access
    @documentation.format(examples=slice_.examples("structure", "to_ase"))
    def to_ase(self, supercell=None):
        """Convert the structure to an ase Atoms object.

        Parameters
        ----------
        supercell : int or np.ndarray
            If present the structure is replicated the specified number of times
            along each direction.

        Returns
        -------
        ase.Atoms
            Structural information for ase package.

        {examples}
        """
        if self._is_slice:
            message = (
                "Converting multiple structures to ASE trajectories is not implemented."
            )
            raise exception.NotImplemented(message)
        data = self.to_dict()
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

    @base.data_access
    @documentation.format(examples=slice_.examples("structure", "to_mdtraj"))
    def to_mdtraj(self):
        """Convert the trajectory to mdtraj.Trajectory

        Returns
        -------
        mdtraj.Trajectory
            The mdtraj package offers many functionalities to analyze a MD
            trajectory. By converting the Vasp data to their format, we facilitate
            using all functions of that package.

        {examples}
        """
        if not self._is_slice:
            message = "Converting a single structure to mdtraj is not implemented."
            raise exception.NotImplemented(message)
        data = self.to_dict()
        xyz = data["positions"] @ data["lattice_vectors"] * self.A_to_nm
        trajectory = mdtraj.Trajectory(xyz, self._topology().to_mdtraj())
        trajectory.unitcell_vectors = data["lattice_vectors"] * Structure.A_to_nm
        return trajectory

    @base.data_access
    @documentation.format(examples=slice_.examples("structure", "to_POSCAR"))
    def to_POSCAR(self):
        """Convert the structure(s) to a POSCAR format

        Returns
        -------
        str or list[str]
            Returns the POSCAR of the current or all selected steps.

        {examples}
        """
        if not self._is_slice:
            return self._create_repr()
        else:
            message = "Converting multiple structures to a POSCAR is currently not implemented."
            raise exception.NotImplemented(message)

    @base.data_access
    @documentation.format(examples=slice_.examples("structure", "cartesian_positions"))
    def cartesian_positions(self):
        """Convert the positions from direct coordinates to cartesian ones.

        Returns
        -------
        np.ndarray
            Position of all atoms in cartesian coordinates in Å.

        {examples}
        """
        return self._raw_data.positions[self._steps] @ self._lattice_vectors()

    @base.data_access
    @documentation.format(examples=slice_.examples("structure", "volume"))
    def volume(self):
        """Return the volume of the unit cell for the selected steps.

        Returns
        -------
        float or np.ndarray
            The volume(s) of the selected step(s) in Å³.

        {examples}
        """
        return np.abs(np.linalg.det(self._lattice_vectors()))

    @base.data_access
    def number_atoms(self):
        """Return the total number of atoms in the structure."""
        return self._raw_data.positions.shape[1]

    @base.data_access
    def number_steps(self):
        """Return the number of structures in the trajectory."""
        return len(self._raw_data.positions[self._slice])

    def _topology(self):
        return data.Topology.from_data(self._raw_data.topology)

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


class _LatticeVectors(reader.Reader):
    def error_message(self, key, err):
        key = np.array(key)
        steps = key if key.ndim == 0 else key[0]
        return (
            f"Error reading the lattice vectors. Please check if the steps "
            f"`{steps}` are properly formatted and within the boundaries. "
            "Additionally, you may consider the original error message:\n" + err.args[0]
        )


def _cell_from_ase(structure):
    return raw.Cell(lattice_vectors=np.array([structure.get_cell()]))


class Mixin:
    @property
    def _structure(self):
        return Structure.from_data(self._raw_data.structure)

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import io
from dataclasses import dataclass

import numpy as np

from py4vasp import calculation, exception, raw
from py4vasp._third_party import view
from py4vasp._util import documentation, import_, reader
from py4vasp.calculation import _base, _slice, _topology

ase = import_.optional("ase")
ase_io = import_.optional("ase.io")
mdtraj = import_.optional("mdtraj")


@dataclass
class _Format:
    begin_table: str = ""
    column_separator: str = " "
    row_separator: str = "\n"
    end_table: str = ""
    newline: str = ""

    def comment_line(self, topology, step_string):
        return f"{topology}{step_string}{self.newline}"

    def scaling_factor(self, scale):
        return f"{self._element_to_string(scale)}{self.newline}".lstrip()

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


@documentation.format(examples=_slice.examples("structure"))
class Structure(_slice.Mixin, _base.Refinery, view.Mixin):
    """The structure contains the unit cell and the position of all ions within.

    The crystal structure is the specific arrangement of ions in a three-dimensional
    repeating pattern. This spatial arrangement is characterized by the unit cell and
    the relative position of the ions. The unit cell is repeated periodically in three
    dimensions to form the crystal. The combination of unit cell and ion positions
    determines the symmetry of the crystal. This symmetry helps understanding the
    material properties because some symmetries do not allow for the presence of some
    properties, e.g., you cannot observe a ferroelectric :data:`~py4vasp.calculation.polarization`
    in a system with inversion symmetry. Therefore relaxing the crystal structure with
    VASP is an important first step in analyzing materials properties.

    When you run a relaxation or MD simulation, this class allows to access all
    individual steps of the trajectory. Typically, you would study the converged
    structure after an ionic relaxation or to visualize the changes of the structure
    along the simulation. Moreover, you could take snapshots along the trajectory
    and further process them by computing more properties.

    {examples}
    """

    A_to_nm = 0.1
    "Converting Å to nm used for mdtraj trajectories."

    @classmethod
    def from_POSCAR(cls, poscar, *, elements=None):
        """Generate a structure from string in POSCAR format.

        Parameters
        ----------
        elements : list[str]
            Name of the elements in the order they appear in the POSCAR file. If the
            elements are specified in the POSCAR file, this argument is optional and
            if set it will overwrite the choice in the POSCAR file. Old POSCAR files
            do not specify the name of the elements; in that case this argument is
            required.
        """
        poscar = _replace_or_set_elements(str(poscar), elements)
        poscar = io.StringIO(poscar)
        structure = ase_io.read(poscar, format="vasp")
        return cls.from_ase(structure)

    @classmethod
    def from_ase(cls, structure):
        """Generate a structure from the ase Atoms class."""
        structure = raw.Structure(
            topology=_topology.raw_topology_from_ase(structure),
            cell=_cell_from_ase(structure),
            positions=structure.get_scaled_positions()[np.newaxis],
        )
        return cls.from_data(structure)

    @_base.data_access
    def __str__(self):
        "Generate a string representing the final structure usable as a POSCAR file."
        return self._create_repr()

    @_base.data_access
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
        step = self._get_last_step()
        lines = (
            format_.comment_line(self._topology(), self._step_string()),
            format_.scaling_factor(self._scale()),
            format_.vectors_to_table(self._raw_data.cell.lattice_vectors[step]),
            format_.ion_list(self._topology()),
            format_.coordinate_system(),
            format_.vectors_to_table(self._raw_data.positions[step]),
        )
        return "\n".join(lines)

    @_base.data_access
    @documentation.format(examples=_slice.examples("structure", "to_dict"))
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
            "positions": self._positions(),
            "elements": self._topology().elements(),
            "names": self._topology().names(),
        }

    @_base.data_access
    @documentation.format(examples=_slice.examples("structure", "to_view"))
    def to_view(self, supercell=None):
        """Generate a 3d representation of the structure(s).

        Parameters
        ----------
        supercell : int or np.ndarray
            If present the structure is replicated the specified number of times
            along each direction.

        Returns
        -------
        View
            Visualize the structure(s) as a 3d figure.

        {examples}
        """
        make_3d = lambda array: array if array.ndim == 3 else array[np.newaxis]
        return view.View(
            elements=np.atleast_2d(self._topology().elements()),
            lattice_vectors=make_3d(self._lattice_vectors()),
            positions=make_3d(self._positions()),
            supercell=self._parse_supercell(supercell),
        )

    @_base.data_access
    @documentation.format(examples=_slice.examples("structure", "to_ase"))
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
        num_atoms_prim = len(structure)
        if supercell is not None:
            try:
                structure *= supercell
            except (TypeError, IndexError) as err:
                error_message = (
                    "Generating the supercell failed. Please make sure the requested "
                    "supercell is either an integer or a list of 3 integers."
                )
                raise exception.IncorrectUsage(error_message) from err
        num_atoms_super = len(structure)
        order = sorted(range(num_atoms_super), key=lambda n: n % num_atoms_prim)
        return structure[order]

    @_base.data_access
    @documentation.format(examples=_slice.examples("structure", "to_mdtraj"))
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

    @_base.data_access
    @documentation.format(examples=_slice.examples("structure", "to_POSCAR"))
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

    @_base.data_access
    @documentation.format(examples=_slice.examples("structure", "cartesian_positions"))
    def cartesian_positions(self):
        """Convert the positions from direct coordinates to cartesian ones.

        Returns
        -------
        np.ndarray
            Position of all atoms in cartesian coordinates in Å.

        {examples}
        """
        return self._positions() @ self._lattice_vectors()

    @_base.data_access
    @documentation.format(examples=_slice.examples("structure", "volume"))
    def volume(self):
        """Return the volume of the unit cell for the selected steps.

        Returns
        -------
        float or np.ndarray
            The volume(s) of the selected step(s) in Å³.

        {examples}
        """
        return np.abs(np.linalg.det(self._lattice_vectors()))

    @_base.data_access
    def number_atoms(self):
        """Return the total number of atoms in the structure."""
        if self._is_trajectory:
            return self._raw_data.positions.shape[1]
        else:
            return self._raw_data.positions.shape[0]

    @_base.data_access
    def number_steps(self):
        """Return the number of structures in the trajectory."""
        if self._is_trajectory:
            range_ = range(len(self._raw_data.positions))
            return len(range_[self._slice])
        else:
            return 1

    def _parse_supercell(self, supercell):
        if supercell is None:
            return np.ones(3, np.int_)
        try:
            integer_supercell = np.round(supercell).astype(np.int_)
        except TypeError as error:
            message = (
                f"Could not convert supercell='{supercell}' to an integer numpy array."
            )
            raise exception.IncorrectUsage(message) from error
        if not np.allclose(supercell, integer_supercell):
            message = f"supercell='{supercell}' contains noninteger values."
            raise exception.IncorrectUsage(message)
        if np.isscalar(integer_supercell):
            return np.full(3, integer_supercell)
        if integer_supercell.shape == (3,):
            return integer_supercell
        message = (
            f"supercell='{supercell}' is not a scalar or a three component vector."
        )
        raise exception.IncorrectUsage(message)

    def _topology(self):
        return calculation.topology.from_data(self._raw_data.topology)

    def _lattice_vectors(self):
        lattice_vectors = _LatticeVectors(self._raw_data.cell.lattice_vectors)
        return self._scale() * lattice_vectors[self._get_steps()]

    def _scale(self):
        if isinstance(self._raw_data.cell.scale, np.float_):
            return self._raw_data.cell.scale
        if not self._raw_data.cell.scale.is_none():
            return self._raw_data.cell.scale[()]
        else:
            return 1.0

    def _positions(self):
        return self._raw_data.positions[self._get_steps()]

    def _get_steps(self):
        return self._steps if self._is_trajectory else ()

    def _get_last_step(self):
        return self._last_step_in_slice if self._is_trajectory else ()

    def _step_string(self):
        if self._is_slice:
            range_ = range(len(self._raw_data.positions))[self._steps]
            return f" from step {range_.start + 1} to {range_.stop + 1}"
        elif self._steps == -1:
            return ""
        else:
            return f" (step {self._steps + 1})"

    @_base.data_access
    def __getitem__(self, steps):
        if not self._is_trajectory:
            message = "The structure is not a Trajectory so accessing individual elements is not allowed."
            raise exception.IncorrectUsage(message)
        return super().__getitem__(steps)

    @property
    def _is_trajectory(self):
        return self._raw_data.positions.ndim == 3


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
    lattice_vectors = np.array([structure.get_cell()])
    return raw.Cell(lattice_vectors, scale=raw.VaspData(1.0))


def _replace_or_set_elements(poscar, elements):
    line_with_elements = 5
    elements = "" if not elements else " ".join(elements)
    lines = poscar.split("\n")
    if _elements_not_in_poscar(lines[line_with_elements]):
        _raise_error_if_elements_not_set(elements)
        lines.insert(line_with_elements, elements)
    elif elements:
        lines[line_with_elements] = elements
    return "\n".join(lines)


def _elements_not_in_poscar(elements):
    elements = elements.split()
    return any(element.isdecimal() for element in elements)


def _raise_error_if_elements_not_set(elements):
    if not elements:
        message = """The POSCAR file does not specify the elements needed to create a
            Structure. Please pass `elements=[...]` to the `from_POSCAR` routine where
            ... are the elements in the same order as in the POSCAR."""
        raise exception.IncorrectUsage(message)


class Mixin:
    @property
    def _structure(self):
        return Structure.from_data(self._raw_data.structure)

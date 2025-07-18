# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import io
from dataclasses import dataclass

import numpy as np

from py4vasp import exception, raw
from py4vasp._calculation import _stoichiometry, base, slice_
from py4vasp._third_party import view
from py4vasp._util import documentation, import_, parse, reader

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

    def comment_line(self, stoichiometry, step_string, ion_types):
        return f"{stoichiometry.to_string(ion_types)}{step_string}{self.newline}"

    def scaling_factor(self, scale):
        return f"{self._element_to_string(scale)}{self.newline}".lstrip()

    def ion_list(self, stoichiometry, ion_types):
        return f"{stoichiometry.to_POSCAR(self.newline, ion_types)}{self.newline}"

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
class Structure(slice_.Mixin, base.Refinery, view.Mixin):
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
        poscar = parse.POSCAR(poscar)
        return cls.from_data(poscar.structure)

    @classmethod
    def from_ase(cls, structure):
        """Generate a structure from the ase Atoms class."""
        structure = raw.Structure(
            stoichiometry=_stoichiometry.raw_stoichiometry_from_ase(structure),
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

    def _create_repr(self, format_=_Format(), ion_types=None):
        step = self._get_last_step()
        lines = (
            format_.comment_line(self._stoichiometry(), self._step_string(), ion_types),
            format_.scaling_factor(self._scale()),
            format_.vectors_to_table(self._raw_data.cell.lattice_vectors[step]),
            format_.ion_list(self._stoichiometry(), ion_types),
            format_.coordinate_system(),
            format_.vectors_to_table(self._raw_data.positions[step]),
        )
        return "\n".join(lines)

    @base.data_access
    @documentation.format(
        examples=slice_.examples("structure", "to_dict"),
        ion_types=_stoichiometry.ion_types_documentation,
    )
    def to_dict(self, ion_types=None):
        """Read the structural information into a dictionary.

        Parameters
        ----------
        {ion_types}

        Returns
        -------
        dict
            Contains the unit cell of the crystal, as well as the position of
            all the atoms in units of the lattice vectors and the elements of
            the atoms for all selected steps.

        {examples}
        """
        return {
            "lattice_vectors": self.lattice_vectors(),
            "positions": self.positions(),
            "elements": self._stoichiometry().elements(ion_types),
            "names": self._stoichiometry().names(ion_types),
        }

    @base.data_access
    @documentation.format(
        examples=slice_.examples("structure", "to_view"),
        ion_types=_stoichiometry.ion_types_documentation,
    )
    def to_view(self, supercell=None, ion_types=None):
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
        positions = make_3d(self.positions())
        elements_single_step = self._stoichiometry().elements(ion_types)
        elements_all_steps = np.tile(elements_single_step, (len(positions), 1))
        return view.View(
            elements=elements_all_steps,
            lattice_vectors=make_3d(self.lattice_vectors()),
            positions=positions,
            supercell=self._parse_supercell(supercell),
        )

    @base.data_access
    @documentation.format(
        examples=slice_.examples("structure", "to_ase"),
        ion_types=_stoichiometry.ion_types_documentation,
    )
    def to_ase(self, supercell=None, ion_types=None):
        """Convert the structure to an ase Atoms object.

        Parameters
        ----------
        supercell : int or np.ndarray
            If present the structure is replicated the specified number of times
            along each direction.
        {ion_types}

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
        data = self.to_dict(ion_types)
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
                raise exception.IncorrectUsage(error_message) from None
        num_atoms_super = len(structure)
        order = sorted(range(num_atoms_super), key=lambda n: n % num_atoms_prim)
        return structure[order]

    @base.data_access
    @documentation.format(
        examples=slice_.examples("structure", "to_mdtraj"),
        ion_types=_stoichiometry.ion_types_documentation,
    )
    def to_mdtraj(self, ion_types=None):
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
        data = self.to_dict(ion_types)
        xyz = data["positions"] @ data["lattice_vectors"] * self.A_to_nm
        trajectory = mdtraj.Trajectory(xyz, self._stoichiometry().to_mdtraj(ion_types))
        trajectory.unitcell_vectors = data["lattice_vectors"] * Structure.A_to_nm
        return trajectory

    @base.data_access
    @documentation.format(
        examples=slice_.examples("structure", "to_POSCAR"),
        ion_types=_stoichiometry.ion_types_documentation,
    )
    def to_POSCAR(self, ion_types=None):
        """Convert the structure(s) to a POSCAR format

        Parameters
        ----------
        {ion_types}

        Returns
        -------
        str or list[str]
            Returns the POSCAR of the current or all selected steps.

        {examples}
        """
        if not self._is_slice:
            return self._create_repr(ion_types=ion_types)
        else:
            message = "Converting multiple structures to a POSCAR is currently not implemented."
            raise exception.NotImplemented(message)

    @base.data_access
    @documentation.format(examples=slice_.examples("structure", "to_lammps"))
    def to_lammps(self, standard_form=True):
        """Convert the structure to LAMMPS format

        Parameters
        ----------
        standard_form : bool
            Determines whether the structure is standardize, i.e., the lattice vectors
            are a triagonal matrix.

        Returns
        -------
        str
            Returns a string describing the structure for LAMMPS

        {examples}
        """
        if self._is_slice:
            message = "Converting multiple structures to LAMMPS is not implemented."
            raise exception.NotImplemented(message)
        number_ion_types = self._raw_data.stoichiometry.number_ion_types
        cell_string, transformation = self._cell_and_transformation(standard_form)
        position_lines = self._position_lines(number_ion_types, transformation)
        return f"""\
Configuration 1: system "{self._stoichiometry()}"

{self.number_atoms()} atoms
{len(number_ion_types)} atom types

{cell_string}

Atoms # atomic

{position_lines}"""

    def _cell_and_transformation(self, standard_form):
        if standard_form:
            cell = ase.cell.Cell(self.lattice_vectors())
            cell, transformation = cell.standard_form()
            cell_string = f"""\
0.0 {self._format_number(cell[0,0])} xlo xhi
0.0 {self._format_number(cell[1,1])} ylo yhi
0.0 {self._format_number(cell[2,2])} zlo zhi
{self._format_number((cell[1,0], cell[2,0], cell[2,1]))} xy xz yz"""
        else:
            lattice_vectors = self.lattice_vectors()
            cell_string = f"""\
{self._format_number(lattice_vectors[0])} avec
{self._format_number(lattice_vectors[1])} bvec
{self._format_number(lattice_vectors[2])} cvec
0.0 0.0 0.0 abc origin"""
            transformation = np.eye(3)
        return cell_string, transformation

    def _position_lines(self, number_ion_types, transformation):
        positions = self.cartesian_positions() @ transformation.T
        ion_type_labels = [
            str(ion_type + 1)
            for ion_type, number in enumerate(number_ion_types)
            for _ in range(number)
        ]
        return "\n".join(
            f"{i + 1} {ion_type_labels[i]} {self._format_number(position)}"
            for i, position in enumerate(positions)
        )

    def _format_number(self, number):
        number = np.atleast_1d(number)
        return " ".join(f"{x:24.16E}" for x in number)

    @base.data_access
    def lattice_vectors(self):
        """Return the lattice vectors spanning the unit cell

        Returns
        -------
        np.ndarray
            Lattice vectors of the unit cell in Å.
        """
        lattice_vectors = _LatticeVectors(self._raw_data.cell.lattice_vectors)
        return self._scale() * lattice_vectors[self._get_steps()]

    @base.data_access
    def positions(self):
        """Return the direct coordinates of all ions in the unit cell.

        Direct or fractional coordinates measure the position of the ions in terms of
        the lattice vectors. Hence they are dimensionless quantities.

        Returns
        -------
        np.ndarray
            Positions of all ions in terms of the lattice vectors.
        """
        return self._raw_data.positions[self._get_steps()]

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
        return self.positions() @ self.lattice_vectors()

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
        return np.abs(np.linalg.det(self.lattice_vectors()))

    @base.data_access
    def number_atoms(self):
        """Return the total number of atoms in the structure."""
        if self._is_trajectory:
            return self._raw_data.positions.shape[1]
        else:
            return self._raw_data.positions.shape[0]

    @base.data_access
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
            raise exception.IncorrectUsage(message) from None
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

    def _stoichiometry(self):
        return _stoichiometry.Stoichiometry.from_data(self._raw_data.stoichiometry)

    def _scale(self):
        if isinstance(self._raw_data.cell.scale, np.float64):
            return self._raw_data.cell.scale
        if not self._raw_data.cell.scale.is_none():
            return self._raw_data.cell.scale[()]
        else:
            return 1.0

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

    @base.data_access
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

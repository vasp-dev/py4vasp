# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import copy
from contextlib import suppress
from dataclasses import dataclass
from typing import Union

import numpy as np

from py4vasp import exception, raw
from py4vasp._calculation import _stoichiometry
from py4vasp._calculation._stoichiometry import StoichiometryHandler
from py4vasp._calculation.cell import CellHandler
from py4vasp._calculation.dispatch import (
    DataSource,
    SuppressErrorsSourceWrapper,
    _result_has_data,
    merge_default,
    merge_strings,
    quantity,
)
from py4vasp._raw.definition import unique_selections as _schema_unique_selections
from py4vasp._raw.models import StoichiometryModel, StructureModel
from py4vasp._third_party import view
from py4vasp._util import check, import_, parse

ase = import_.optional("ase")
ase_io = import_.optional("ase.io")
mdtraj = import_.optional("mdtraj")
spglib = import_.optional("spglib")

__all__ = ["Structure"]

_TO_DATABASE_SUPPRESSED_EXCEPTIONS = (
    exception.Py4VaspError,
    np.linalg.LinAlgError,
    AttributeError,
    TypeError,
    ValueError,
    IndexError,
    # a source may reference datasets that are absent/malformed in the HDF5 file;
    # reading them can raise a low-level RuntimeError which we treat as "no data"
    RuntimeError,
)


class StructureHandler:
    """Processes structural data from a single raw.Structure object."""

    A_to_nm = 0.1

    def __init__(self, raw_structure: raw.Structure, steps=None):
        self._raw_structure = raw_structure
        self._steps = steps if steps is not None else -1
        self._is_slice = isinstance(self._steps, slice)
        if self._is_slice:
            self._slice = self._steps
        elif self._steps == -1:
            self._slice = slice(-1, None)
        else:
            try:
                self._slice = slice(self._steps, self._steps + 1)
            except TypeError as error:
                raise exception.IncorrectUsage(
                    f"Steps must be an integer or slice, got {type(self._steps).__name__!r}."
                ) from error

    @classmethod
    def from_data(cls, raw_structure: raw.Structure, steps=None) -> "StructureHandler":
        return cls(raw_structure, steps=steps)

    def to_dict(self, ion_types=None) -> dict:
        """Read the structural information into a dictionary.

        The returned dictionary contains the following keys:
        - 'lattice_vectors': The lattice vectors of the unit cell.
        - 'positions': The positions of the atoms in the unit cell.
        - 'elements': The chemical elements of the atoms in the unit cell.
        - 'names': The names of the atoms in the unit cell.

        Note that 'elements' and 'names' have the same length as the number of atoms in
        the unit cell.

        Parameters
        ----------
        ion_types : Sequence
            Overwrite the ion types present in the raw data. You can use this to quickly
            generate different stoichiometries without modifying the underlying raw data.

        Returns
        -------
        dict
            Contains the unit cell of the crystal, as well as the position of
            all the atoms in units of the lattice vectors and the elements of
            the atoms for all selected steps.
        """
        return {
            "lattice_vectors": self.lattice_vectors(),
            "positions": self.positions(),
            "elements": self._stoichiometry().elements(ion_types),
            "names": self._stoichiometry().names(ion_types),
        }

    def __str__(self):
        """Generate a string representing the final structure usable as a POSCAR file."""
        return self._create_repr()

    def _repr_html_(self):
        format_ = _Format(
            begin_table="<table>\n<tr><td>",
            column_separator="</td><td>",
            row_separator="</td></tr>\n<tr><td>",
            end_table="</td></tr>\n</table>",
            newline="<br>",
        )
        return self._create_repr(format_)

    def to_POSCAR(self, ion_types=None) -> str:
        """Convert the structure(s) to a POSCAR format."""
        if not self._is_slice:
            return self._create_repr(ion_types=ion_types)
        else:
            message = "Converting multiple structures to a POSCAR is currently not implemented."
            raise exception.NotImplemented(message)

    def to_view(self, supercell=None, ion_types=None):
        """Generate a 3d representation of the structure(s)."""
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

    def to_ase(self, supercell=None, ion_types=None):
        """Convert the structure to an ASE Atoms object."""
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

    def to_mdtraj(self, ion_types=None):
        """Convert the trajectory to mdtraj.Trajectory."""
        if not self._is_slice:
            message = "Converting a single structure to mdtraj is not implemented."
            raise exception.NotImplemented(message)
        data = self.to_dict(ion_types)
        xyz = data["positions"] @ data["lattice_vectors"] * self.A_to_nm
        trajectory = mdtraj.Trajectory(xyz, self._stoichiometry().to_mdtraj(ion_types))
        trajectory.unitcell_vectors = data["lattice_vectors"] * self.A_to_nm
        return trajectory

    def to_lammps(self, standard_form=True) -> str:
        """Convert the structure to LAMMPS format."""
        if self._is_slice:
            message = "Converting multiple structures to LAMMPS is not implemented."
            raise exception.NotImplemented(message)
        number_ion_types = self._raw_structure.stoichiometry.number_ion_types
        cell_string, transformation = self._cell_and_transformation(standard_form)
        position_lines = self._position_lines(number_ion_types, transformation)
        return f"""\
Configuration 1: system "{self._stoichiometry()}"

{self.number_atoms()} atoms
{len(number_ion_types)} atom types

{cell_string}

Atoms # atomic

{position_lines}"""

    def lattice_vectors(self):
        """Return the lattice vectors spanning the unit cell."""
        return self._cell().lattice_vectors()

    def positions(self):
        """Return the direct coordinates of all ions in the unit cell."""
        return self._raw_structure.positions[self._get_steps()]

    def cartesian_positions(self):
        """Convert the positions from direct coordinates to cartesian ones."""
        return self.positions() @ self.lattice_vectors()

    def volume(self):
        """Return the volume of the unit cell for the selected steps."""
        return np.abs(np.linalg.det(self.lattice_vectors()))

    def number_atoms(self) -> int:
        """Return the total number of atoms in the structure."""
        if self._is_trajectory:
            return self._raw_structure.positions.shape[1]
        else:
            return self._raw_structure.positions.shape[0]

    def number_steps(self) -> int:
        """Return the number of structures in the trajectory."""
        if self._is_trajectory:
            range_ = range(len(self._raw_structure.positions))
            return len(range_[self._slice])
        else:
            return 1

    def equivalent_atoms(self) -> np.ndarray:
        """Return the orbit index of every atom under VASP's symmetry operations."""
        return self._orbit_labels()

    def _raw_symmetry(self):
        """Return the raw symmetry, raising if the structure does not provide it."""
        symmetry = self._raw_structure.symmetry
        if check.is_none(symmetry):
            message = (
                "The structure does not provide symmetry information; it requires VASP "
                "6.6 or later. Symmetry-derived properties such as the Wyckoff "
                "positions or the equivalent atoms cannot be computed."
            )
            raise exception.NoData(message)
        return symmetry

    def _orbit_labels(self) -> np.ndarray:
        """Group the atoms into orbits (equivalence classes) of VASP's operations.

        Two atoms belong to the same orbit if some symmetry operation maps one onto
        the other. The classes are computed from ``atom_permutations`` with a union-find
        pass and relabeled to consecutive indices starting at 0.
        """
        symmetry = self._raw_symmetry()
        permutations = np.array(symmetry.atom_permutations) - 1  # Fortran to 0-based
        permutations = permutations.reshape(-1, permutations.shape[-1])
        number_atoms = permutations.shape[-1]
        if number_atoms != self.number_atoms():
            message = (
                f"The symmetry describes {number_atoms} atoms but the structure has "
                f"{self.number_atoms()}; the structure and its symmetry are inconsistent."
            )
            raise exception.DataMismatch(message)
        labels = np.arange(number_atoms)
        for permutation in permutations:
            for atom, image in enumerate(permutation):
                low, high = sorted((labels[atom], labels[image]))
                labels[labels == high] = low
        _, orbits = np.unique(labels, return_inverse=True)
        return orbits

    def to_database(self, steps=-1) -> StructureModel:
        """Return database-ready data for a single structure geometry.

        *steps* selects which geometry to describe (default ``-1``, the final step).
        The database splits a calculation into separate ``initial`` and ``final``
        structure models, so each :class:`StructureModel` holds one geometry with
        unprefixed fields.
        """
        # Temporarily override the steps used for the database
        saved_steps = self._steps
        saved_is_slice = self._is_slice
        saved_slice = self._slice
        self._steps = steps
        self._is_slice = isinstance(steps, slice)
        if self._is_slice:
            self._slice = steps
        elif steps == -1:
            self._slice = slice(-1, None)
        else:
            self._slice = slice(steps, steps + 1)

        try:
            return self._single_geometry_database()
        finally:
            self._steps = saved_steps
            self._is_slice = saved_is_slice
            self._slice = saved_slice

    def _single_geometry_database(self) -> StructureModel:
        lattice = [None, None, None]
        with suppress(*_TO_DATABASE_SUPPRESSED_EXCEPTIONS):
            lattices = self.lattice_vectors()
            lattice = lattices[-1] if lattices.ndim == 3 else lattices
            if lattice.ndim != 2:
                lattice = [None, None, None]

        volume = None
        with suppress(*_TO_DATABASE_SUPPRESSED_EXCEPTIONS):
            volumes = self.volume()
            volume = (
                volumes[-1]
                if not isinstance(volumes, (float, np.float64, np.float32))
                else volumes
            )

        lengths = angles = None
        cell_area_2d = cell_area_2d_span = None
        dimensionality = 3
        with suppress(*_TO_DATABASE_SUPPRESSED_EXCEPTIONS):
            dimensionality = self._dimensionality()

        with suppress(*_TO_DATABASE_SUPPRESSED_EXCEPTIONS):
            cell_ = self._cell()
            all_lengths = cell_.lengths()
            lengths = all_lengths[-1] if all_lengths.ndim == 2 else all_lengths
            all_angles = cell_.angles()
            angles = all_angles[-1] if all_angles.ndim == 2 else all_angles
            if dimensionality == 2:
                area, span = cell_._area_2d()
                cell_area_2d = area[-1] if isinstance(area, np.ndarray) else area
                cell_area_2d_span = span[-1] if isinstance(span, list) else span

        num_atoms = self.number_atoms() or None

        stoichiometry = StoichiometryModel()
        with suppress(*_TO_DATABASE_SUPPRESSED_EXCEPTIONS):
            stoichiometry = self._stoichiometry().to_database()

        return StructureModel(
            num_ions=num_atoms,
            dimensionality=dimensionality,
            ion_types=stoichiometry.ion_types,
            num_ion_types=stoichiometry.num_ion_types,
            num_ion_types_primitive=stoichiometry.num_ion_types_primitive,
            formula=stoichiometry.formula,
            compound=stoichiometry.compound,
            cell_volume=volume,
            cell_area_2d=cell_area_2d,
            cell_area_2d_span=cell_area_2d_span,
            lattice_vector_1=list(lattice[0]) if lattice[0] is not None else None,
            lattice_vector_2=list(lattice[1]) if lattice[1] is not None else None,
            lattice_vector_3=list(lattice[2]) if lattice[2] is not None else None,
            lattice_vector_1_length=lengths[0] if lengths is not None else None,
            lattice_vector_2_length=lengths[1] if lengths is not None else None,
            lattice_vector_3_length=lengths[2] if lengths is not None else None,
            angle_alpha=angles[0] if angles is not None else None,
            angle_beta=angles[1] if angles is not None else None,
            angle_gamma=angles[2] if angles is not None else None,
        )

    def _dimensionality(self) -> Union[int, np.ndarray]:
        """Heuristic check for dimensionality of system."""
        if not check.is_none(self._raw_structure.idipol):
            if self._raw_structure.idipol < 1:
                return 3
            elif self._raw_structure.idipol in [1, 2, 3]:
                return 2
            elif self._raw_structure.idipol == 4:
                return 0
        cell_ = self._cell()
        if bool(np.all(np.array(cell_.is_suspected_2d_system))):
            return 2
        return 3

    def _stoichiometry(self) -> StoichiometryHandler:
        return StoichiometryHandler.from_data(self._raw_structure.stoichiometry)

    def _cell(self) -> CellHandler:
        return CellHandler.from_data(self._raw_structure.cell, steps=self._steps)

    def _get_steps(self):
        return self._steps if self._is_trajectory else ()

    def _get_last_step(self):
        return self._last_step_in_slice if self._is_trajectory else ()

    @property
    def _last_step_in_slice(self):
        return (self._slice.stop or 0) - 1

    def _step_string(self):
        if self._is_slice:
            range_ = range(len(self._raw_structure.positions))[self._steps]
            return f" from step {range_.start + 1} to {range_.stop + 1}"
        elif self._steps == -1:
            return ""
        else:
            return f" (step {self._steps + 1})"

    def _create_repr(self, format_=None, ion_types=None):
        if format_ is None:
            format_ = _Format()
        step = self._get_last_step()
        stoichiometry = self._stoichiometry()
        lines = (
            format_.comment_line(stoichiometry, self._step_string(), ion_types),
            format_.scaling_factor(self._cell().scale()),
            format_.vectors_to_table(self._raw_structure.cell.lattice_vectors[step]),
            format_.ion_list(stoichiometry, ion_types),
            format_.coordinate_system(),
            format_.vectors_to_table(self._raw_structure.positions[step]),
        )
        return "\n".join(lines)

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

    def _cell_and_transformation(self, standard_form):
        if standard_form:
            cell_obj = ase.cell.Cell(self.lattice_vectors())
            cell_obj, transformation = cell_obj.standard_form()
            cell_string = f"""\
0.0 {self._format_number(cell_obj[0,0])} xlo xhi
0.0 {self._format_number(cell_obj[1,1])} ylo yhi
0.0 {self._format_number(cell_obj[2,2])} zlo zhi
{self._format_number((cell_obj[1,0], cell_obj[2,0], cell_obj[2,1]))} xy xz yz"""
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

    @property
    def _is_trajectory(self):
        return self._raw_structure.positions.ndim == 3


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


@quantity("structure")
class Structure(view.Mixin):
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

    Examples
    --------
    Let us create some example data so that we can illustrate how to use this class.
    Of course you can also use your own VASP calculation data if you have it available.

    >>> from py4vasp import demo
    >>> calculation = demo.calculation(path)

    If you access the structure, the result will depend on the steps that you selected
    with the [] operator. Without any selection the results from the final step will be
    used.

    >>> calculation.structure.number_steps()
    1

    To select the results for all steps, you don't specify the array boundaries.

    >>> calculation.structure[:].number_steps()
    4

    You can also select specific {step}s or a subset of {step}s as follows

    >>> calculation.structure[3].number_steps()
    1
    >>> calculation.structure[1:4].number_steps()
    3
    """

    A_to_nm = 0.1
    "Converting Å to nm used for mdtraj trajectories."

    def __init__(self, source, quantity_name="structure", steps=None):
        self._source = source
        self._quantity_name = quantity_name
        self._steps = steps

    @classmethod
    def from_data(cls, raw_structure) -> "Structure":
        return cls(source=DataSource(raw_structure))

    def __repr__(self):
        return f"{type(self).__name__}.from_path({self._path!r})"

    def _handler_factory(self, raw):
        return StructureHandler.from_data(raw, steps=self._steps)

    def __getitem__(self, steps) -> "Structure":
        with self._source.access(self._quantity_name) as raw_structure:
            is_trajectory = raw_structure.positions.ndim == 3
        if not is_trajectory:
            message = (
                "The structure is not a Trajectory so accessing individual "
                "elements is not allowed."
            )
            raise exception.IncorrectUsage(message)
        new = copy.copy(self)
        new._steps = steps
        return new

    @classmethod
    def from_POSCAR(cls, poscar, *, elements=None):
        """Generate a structure from string in POSCAR format.

        The POSCAR format is the standard format to represent crystal structures in
        VASP. This method allows to create a structure from a POSCAR string.
        To read more about the POSCAR format, please refer to the `VASP manual <https://vasp.at/wiki/POSCAR>`_.

        Parameters
        ----------
        elements : list[str]
            Name of the elements in the order they appear in the POSCAR file. If the
            elements are specified in the POSCAR file, this argument is optional and
            if set it will overwrite the choice in the POSCAR file. Old POSCAR files
            do not specify the name of the elements; in that case this argument is
            required.

        Examples
        --------
        We can create a GaAs structure from a POSCAR string as follows

        >>> poscar = '''\\
        ... GaAs
        ... 5.65325
        ... 0.0 0.5 0.5
        ... 0.5 0.0 0.5
        ... 0.5 0.5 0.0
        ... 1 1
        ... fractional
        ... 0.0 0.0 0.0
        ... 0.25 0.25 0.25'''
        >>> structure = py4vasp.calculation.structure.from_POSCAR(poscar, elements=['Ga', 'As'])
        >>> print(structure.to_POSCAR())
        GaAs
        5.6532...
        0.0... 0.5... 0.5...
        0.5... 0.0... 0.5...
        0.5... 0.5... 0.0...
        Ga As
        1 1
        Direct
        0.00... 0.00... 0.00...
        0.25... 0.25... 0.25...
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

    def __str__(self, selection=None):
        "Generate a string representing the final structure usable as a POSCAR file."
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            StructureHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else "...")

    def _repr_html_(self):
        return merge_strings(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StructureHandler._repr_html_,
        )

    def read(self, ion_types=None):
        """Read the structural information into a dictionary.

        The returned dictionary contains the following keys:
        - 'lattice_vectors': The lattice vectors of the unit cell.
        - 'positions': The positions of the atoms in the unit cell.
        - 'elements': The chemical elements of the atoms in the unit cell.
        - 'names': The names of the atoms in the unit cell.

        Note that 'elements' and 'names' have the same length as the number of atoms in
        the unit cell.

        Parameters
        ----------
        ion_types : Sequence
            Overwrite the ion types present in the raw data. You can use this to quickly
            generate different stoichiometries without modifying the underlying raw data.

        Returns
        -------
        dict
            Contains the unit cell of the crystal, as well as the position of
            all the atoms in units of the lattice vectors and the elements of
            the atoms for all selected steps.

        Examples
        --------
        First, we create some example data so that we can illustrate how to use this method.
        You can also use your own VASP calculation data if you have it available.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        If you use the `read` method, the result will depend on the steps that you
        selected with the [] operator. Without any selection the results from the final
        step will be used.

        >>> calculation.structure.read()
        {'lattice_vectors': array([[...]]), 'positions': array([[...]]),
            'elements': [...], 'names': [...]}

        To select the results for all steps, you don't specify the array boundaries.
        Notice that in this case the lattice vectors and positions contain an additional
        dimension for the different steps.

        >>> calculation.structure[:].read()
        {'lattice_vectors': array([[[...]]]), 'positions': array([[[...]]]),
            'elements': [...], 'names': [...]}

        You can also select specific steps or a subset of steps as follows

        >>> calculation.structure[1].read()
        {'lattice_vectors': array([[...]]), 'positions': array([[...]]),
            'elements': [...], 'names': [...]}
        >>> calculation.structure[0:2].read()
        {'lattice_vectors': array([[[...]]]), 'positions': array([[[...]]]),
            'elements': [...], 'names': [...]}
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StructureHandler.to_dict,
            ion_types,
        )

    def to_dict(self, ion_types=None):
        """Convenient alias for :py:meth:`read`. Please read the documentation there."""
        return self.read(ion_types=ion_types)

    def to_view(self, supercell=None, ion_types=None):
        """Generate a 3d representation of the structure(s).

        This method uses the `View` class to create a 3d visualization of the atomic
        structure(s) in the unit cell.

        Parameters
        ----------
        supercell : int or np.ndarray
            If present the structure is replicated the specified number of times
            along each direction.
        ion_types : Sequence
            Overwrite the ion types present in the raw data. You can use this to quickly
            generate different stoichiometries without modifying the underlying raw data.

        Returns
        -------
        View
            Visualize the structure(s) as a 3d figure.

        Examples
        --------
        First, we create some example data so that we can illustrate how to use this method.
        You can also use your own VASP calculation data if you have it available.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        If you use the `to_view` method, the result will depend on the steps that you
        selected with the [] operator. Without any selection the results from the final
        step will be used.

        >>> calculation.structure.to_view()
        View(elements=array([[...]], dtype=...), lattice_vectors=array([[[...]]]),
            positions=array([[[...]]]), ...)

        To select the results for all steps, you don't specify the array boundaries.
        Notice that in this case the lattice vectors and positions contain an additional
        dimension for the different steps.

        >>> calculation.structure[:].to_view()
        View(elements=array([[...], ..., [...]], dtype=...), lattice_vectors=array([[[...]], ..., [[...]]]),
            positions=array([[[...]], ..., [[...]]]), ...)

        You can also select specific steps or a subset of steps as follows

        >>> calculation.structure[1].to_view()
        View(elements=array([[...]], dtype=...), lattice_vectors=array([[[...]]]),
            positions=array([[[...]]]), ...)
        >>> calculation.structure[0:2].to_view()
        View(elements=array([[...], [...]], dtype=...), lattice_vectors=array([[[...]], [[...]]]),
            positions=array([[[...]], [[...]]]), ...)

        You may also replicate the structure by specifying a supercell.

        >>> calculation.structure.to_view(supercell=2)
        View(..., supercell=array([2, 2, 2]), ...)

        The supercell size can also be different for the different directions.

        >>> calculation.structure.to_view(supercell=[2,3,1])
        View(..., supercell=array([2, 3, 1]), ...)
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StructureHandler.to_view,
            supercell,
            ion_types,
        )

    def to_ase(self, supercell=None, ion_types=None):
        """Convert the structure to an ASE Atoms object.

        ASE (the Atomic Simulation Environment) is a popular Python package for atomistic
        simulations. This method converts the VASP structure to an ASE Atoms object,
        which can be used for further analysis and visualization.

        Parameters
        ----------
        supercell : int or np.ndarray
            If present the structure is replicated the specified number of times
            along each direction.
        ion_types : Sequence
            Overwrite the ion types present in the raw data. You can use this to quickly
            generate different stoichiometries without modifying the underlying raw data.

        Returns
        -------
        Atoms
            Structural information for ASE package. Read more about ASE `here <https://ase-lib.org>`_.

        Examples
        --------
        First, we create some example data so that we can illustrate how to use this method.
        You can also use your own VASP calculation data if you have it available.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        If you use the `to_ase` method, the result will depend on the steps that you
        selected with the [] operator. Without any selection the results from the final
        step will be used.

        >>> calculation.structure.to_ase()
        Atoms(symbols='...', pbc=True, cell=[[...]])

        You can also select specific steps as follows

        >>> calculation.structure[1].to_ase()
        Atoms(symbols='...', pbc=True, cell=[[...]])

        Notice that converting multiple steps to ASE trajectories is not implemented.

        You may also replicate the structure by specifying a supercell. If you compare
        the cell size with the previous example, you will see that it is doubled in all
        directions.

        >>> calculation.structure.to_ase(supercell=2)
        Atoms(symbols='...', pbc=True, cell=[[...]])

        The supercell size can also be different for the different directions. The three
        lattice vectors will be scaled accordingly.

        >>> calculation.structure.to_ase(supercell=[2,3,1])
        Atoms(symbols='...', pbc=True, cell=[[...]])
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StructureHandler.to_ase,
            supercell,
            ion_types,
        )

    def to_mdtraj(self, ion_types=None):
        """Convert the trajectory to mdtraj.Trajectory

        mdtraj is a popular Python package to analyze molecular dynamics trajectories.
        This method converts the VASP structure trajectory to an mdtraj.Trajectory
        object, which can be used for further analysis and visualization.

        Parameters
        ----------
        ion_types : Sequence
            Overwrite the ion types present in the raw data. You can use this to quickly
            generate different stoichiometries without modifying the underlying raw data.

        Returns
        -------
        mdtraj.Trajectory
            The mdtraj package offers many functionalities to analyze a MD
            trajectory. By converting the VASP data to their format, we facilitate
            using all functions of that package.

        Examples
        --------
        First, we create some example data so that we can illustrate how to use this method.
        You can also use your own VASP calculation data if you have it available.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        To convert the whole trajectory (all steps), you don't specify the array boundaries.

        >>> calculation.structure[:].to_mdtraj()
        <mdtraj.Trajectory with ... frames, ... atoms, ...>

        You can also select a subset of steps as follows

        >>> calculation.structure[0:2].to_mdtraj()
        <mdtraj.Trajectory with 2 frames, ... atoms, ...>

        You cannot convert a single structure to mdtraj.Trajectory.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StructureHandler.to_mdtraj,
            ion_types,
        )

    def to_POSCAR(self, ion_types=None):
        """Convert the structure(s) to a POSCAR format.

        Use this method to generate a string in POSCAR format representing the
        structure(s). You can use this string to write a POSCAR file for VASP. This
        can be useful if you want to use the relaxed structure from a VASP calculation
        or a snapshot from an MD simulation as input for a new VASP calculation.

        Parameters
        ----------
        ion_types : Sequence
            Overwrite the ion types present in the raw data. You can use this to quickly
            generate different stoichiometries without modifying the underlying raw data.

        Returns
        -------
        str
            Returns the POSCAR of the selected steps.

        Examples
        --------
        First, we create some example data so that we can illustrate how to use this method.
        You can also use your own VASP calculation data if you have it available.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        If you use the `to_POSCAR` method, the result will depend on the steps that you
        selected with the [] operator. Without any selection the results from the final
        step will be used.

        >>> poscar = calculation.structure.to_POSCAR()
        >>> assert poscar == str(calculation.structure)

        You can also select specific steps as follows

        >>> poscar = calculation.structure[1].to_POSCAR()
        >>> assert poscar == str(calculation.structure[1])

        Notice that converting multiple steps to POSCAR format is not implemented.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StructureHandler.to_POSCAR,
            ion_types,
        )

    def to_lammps(self, standard_form=True):
        """Convert the structure to LAMMPS format.

        LAMMPS is a popular molecular dynamics simulation software. This method
        converts the structure to a string in LAMMPS format, which can be used as
        input for LAMMPS simulations.

        Parameters
        ----------
        standard_form : bool
            Determines whether the structure is standardize, i.e., the lattice vectors
            are a triagonal matrix.

        Returns
        -------
        str
            Returns a string describing the structure for LAMMPS

        Examples
        --------
        First, we create some example data so that we can illustrate how to use this method.
        You can also use your own VASP calculation data if you have it available.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        If you use the `to_lammps` method, the result will depend on the steps that you
        selected with the [] operator. Without any selection the results from the final
        step will be used.

        >>> print(calculation.structure.to_lammps())
        Configuration 1: system "..."
        ... atoms
        ... atom types
        ... xlo xhi
        ... ylo yhi
        ... zlo zhi
        ... xy xz yz
        Atoms # atomic
        1 1 ...

        You can also select specific steps as follows

        >>> print(calculation.structure[1].to_lammps())
        Configuration 1: system "..."
        ... atoms
        ... atom types
        ... xlo xhi
        ... ylo yhi
        ... zlo zhi
        ... xy xz yz
        Atoms # atomic
        1 1 ...

        Notice that converting multiple steps to LAMMPS format is not implemented.

        LAMMPS requires either a standard form of the unit cell or the transformation
        from the original cell to the standard form. By default, the standard form is
        used. You can disable this behavior as follows

        >>> print(calculation.structure.to_lammps(standard_form=False))
        Configuration 1: system "..."
        ... atoms
        ... atom types
        ... avec
        ... bvec
        ... cvec
        ... abc origin
        Atoms # atomic
        1 1 ...

        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StructureHandler.to_lammps,
            standard_form,
        )

    def lattice_vectors(self):
        """Return the lattice vectors spanning the unit cell

        Returns
        -------
        np.ndarray
            Lattice vectors of the unit cell in Å.

        Examples
        --------
        First, we create some example data so that we can illustrate how to use this method.
        You can also use your own VASP calculation data if you have it available.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        If you use the `lattice_vectors` method, the result will depend on the steps that you
        selected with the [] operator. Without any selection the results from the final
        step will be used.

        >>> calculation.structure.lattice_vectors()
        array([[...], [...], [...]])

        To select the results for all steps, you don't specify the array boundaries.

        >>> calculation.structure[:].lattice_vectors()
        array([[[...]], [[...]], [[...]], [[...]]])
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StructureHandler.lattice_vectors,
        )

    def positions(self):
        """Return the direct coordinates of all ions in the unit cell.

        Direct or fractional coordinates measure the position of the ions in terms of
        the lattice vectors. Hence they are dimensionless quantities.

        Returns
        -------
        np.ndarray
            Positions of all ions in terms of the lattice vectors.

        Examples
        --------
        First, we create some example data so that we can illustrate how to use this method.
        You can also use your own VASP calculation data if you have it available.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        If you use the `positions` method, the result will depend on the steps that you
        selected with the [] operator. Without any selection the results from the final
        step will be used.

        >>> calculation.structure.positions()
        array([[...]])

        To select the results for all steps, you don't specify the array boundaries.

        >>> calculation.structure[:].positions()
        array([[[...]]])
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StructureHandler.positions,
        )

    def cartesian_positions(self):
        """Convert the positions from direct coordinates to cartesian ones.

        Returns
        -------
        np.ndarray
            Position of all atoms in cartesian coordinates in Å.

        Examples
        --------
        First, we create some example data so that we can illustrate how to use this method.
        You can also use your own VASP calculation data if you have it available.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        If you use the `cartesian_positions` method, the result will depend on the steps that you
        selected with the [] operator. Without any selection the results from the final
        step will be used.

        >>> calculation.structure.cartesian_positions()
        array([[...]])

        To select the results for all steps, you don't specify the array boundaries.

        >>> calculation.structure[:].cartesian_positions()
        array([[[...]]])
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StructureHandler.cartesian_positions,
        )

    def volume(self):
        """Return the volume of the unit cell for the selected steps.

        Returns
        -------
        float or np.ndarray
            The volume(s) of the selected step(s) in Å³.

        Examples
        --------
        First, we create some example data so that we can illustrate how to use this method.
        You can also use your own VASP calculation data if you have it available.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        If you use the `volume` method, the result will depend on the steps that you
        selected with the [] operator. Without any selection the results from the final
        step will be used.

        >>> calculation.structure.volume()
        np.float...

        To select the results for all steps, you don't specify the array boundaries.

        >>> calculation.structure[:].volume()
        array([...])
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StructureHandler.volume,
        )

    def number_atoms(self):
        """Return the total number of atoms in the structure."""
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StructureHandler.number_atoms,
        )

    def number_steps(self):
        """Return the number of structures in the trajectory."""
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StructureHandler.number_steps,
        )

    def equivalent_atoms(self):
        """Group the atoms into orbits of the symmetry operations VASP determined.

        Atoms that some symmetry operation maps onto each other are equivalent and
        share an orbit index. This uses the symmetry VASP recognized for the crystal,
        so any symmetry lowering (e.g. from magnetic order) is reflected here. It
        requires the structure to carry symmetry information (VASP 6.6 or later).

        Returns
        -------
        np.ndarray
            The orbit index of every atom. Atoms with the same index are equivalent
            under the symmetry operations.

        Examples
        --------
        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path, "perovskite")

        In cubic perovskite SrTiO3 the strontium and titanium atoms each sit on their
        own site while the three oxygen atoms are equivalent.

        >>> calculation.structure.equivalent_atoms()
        array([0, 1, 2, 2, 2]...)
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StructureHandler.equivalent_atoms,
        )

    def _to_database(self) -> dict:
        """Return ``{"structure": {selection: StructureModel}}`` for the database.

        Uses a custom merge instead of the generic :func:`merge_to_database`: the
        final structure is always stored, the initial structure (first ionic step
        of the ``default`` trajectory) only when it differs from the final one, and
        any additional source only when it differs from both.
        """
        models = _collect_structure_database(self._source, self._quantity_name)
        return {"structure": models} if models else {}


def _collect_structure_database(source, quantity_name):
    """Build ``{selection: StructureModel}`` from the available structure sources.

    The ``final`` source (or, when absent, the last step of the ``default``
    trajectory) provides the final structure, which is always included. The first
    ionic step of the ``default`` trajectory provides the initial structure, kept
    only when it differs from the final one. Every other source is kept only when
    it differs from both.
    """
    entries = _read_structure_entries(source, quantity_name)
    return _assemble_structure_models(entries)


def _read_structure_entries(source, quantity_name):
    """Materialize ``{source: {step: (model, geometry)}}`` for every schema source.

    The reads happen while the source's access context is open (raw data holds
    lazy references that become invalid once the file is closed) and the arrays
    are copied so they survive after the context exits. The ``default`` trajectory
    is read at both its first (``0``) and last (``-1``) step; every other source
    only at its final step (``-1``).
    """
    wrapped = SuppressErrorsSourceWrapper(source)
    entries = {}
    for name in _schema_unique_selections(quantity_name.lstrip("_")):
        selection = None if name == "default" else name
        steps = (-1, 0) if name == "default" else (-1,)
        with wrapped.access(quantity_name, selection=selection) as raw_structure:
            if raw_structure is None:
                continue
            per_step = {}
            for step in steps:
                entry = _build_structure_entry(raw_structure, step)
                if entry is not None:
                    per_step[step] = entry
            if per_step:
                entries[name] = per_step
    return entries


def _build_structure_entry(raw_structure, step):
    """Return ``(StructureModel, (lattice, positions))`` for a single step, or None."""
    with suppress(*_TO_DATABASE_SUPPRESSED_EXCEPTIONS):
        handler = StructureHandler.from_data(raw_structure, steps=step)
        geometry = (np.array(handler.lattice_vectors()), np.array(handler.positions()))
        model = StructureHandler.from_data(raw_structure).to_database(steps=step)
        if not _result_has_data(model):
            return None
        return model, geometry
    return None


def _assemble_structure_models(entries):
    """Apply the final/initial/other rules to the collected per-source entries."""
    models, geometries = {}, {}
    final = entries.get("final", {}).get(-1) or entries.get("default", {}).get(-1)
    if final is not None:
        models["final"], geometries["final"] = final
    initial = entries.get("default", {}).get(0)
    if initial is not None and not _duplicate_geometry(initial[1], geometries):
        models["initial"], geometries["initial"] = initial
    for name, per_step in entries.items():
        if name in ("default", "final") or -1 not in per_step:
            continue
        model, geometry = per_step[-1]
        if _duplicate_geometry(geometry, geometries):
            continue
        models[name], geometries[name] = model, geometry
    return models


def _duplicate_geometry(geometry, geometries):
    """Whether *geometry* matches any already-collected geometry."""
    return any(_same_geometry(geometry, other) for other in geometries.values())


def _same_geometry(first, second):
    """Two geometries are equal when both lattice vectors and positions match."""
    return all(
        np.shape(a) == np.shape(b) and np.allclose(a, b) for a, b in zip(first, second)
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

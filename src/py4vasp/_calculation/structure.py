# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from dataclasses import dataclass

import numpy as np

from py4vasp import exception, raw
from py4vasp._calculation import _stoichiometry, base, cell, slice_
from py4vasp._third_party import view
from py4vasp._util import database, import_, parse

ase = import_.optional("ase")
ase_io = import_.optional("ase.io")
mdtraj = import_.optional("mdtraj")

__all__ = ["Structure"]


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
            format_.scaling_factor(self._cell().scale()),
            format_.vectors_to_table(self._raw_data.cell.lattice_vectors[step]),
            format_.ion_list(self._stoichiometry(), ion_types),
            format_.coordinate_system(),
            format_.vectors_to_table(self._raw_data.positions[step]),
        )
        return "\n".join(lines)

    @base.data_access
    def to_dict(self, ion_types=None):
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

        If you use the `to_dict` method, the result will depend on the steps that you
        selected with the [] operator. Without any selection the results from the final
        step will be used.

        >>> calculation.structure.to_dict()
        {'lattice_vectors': array([[...]]), 'positions': array([[...]]),
            'elements': [...], 'names': [...]}

        To select the results for all steps, you don't specify the array boundaries.
        Notice that in this case the lattice vectors and positions contain an additional
        dimension for the different steps.

        >>> calculation.structure[:].to_dict()
        {'lattice_vectors': array([[[...]]]), 'positions': array([[[...]]]),
            'elements': [...], 'names': [...]}

        You can also select specific steps or a subset of steps as follows

        >>> calculation.structure[1].to_dict()
        {'lattice_vectors': array([[...]]), 'positions': array([[...]]),
            'elements': [...], 'names': [...]}
        >>> calculation.structure[0:2].to_dict()
        {'lattice_vectors': array([[[...]]]), 'positions': array([[[...]]]),
            'elements': [...], 'names': [...]}
        """
        return {
            "lattice_vectors": self.lattice_vectors(),
            "positions": self.positions(),
            "elements": self._stoichiometry().elements(ion_types),
            "names": self._stoichiometry().names(ion_types),
        }

    @base.data_access
    def _to_database(self, *args, **kwargs):
        stoichiometry = self._stoichiometry()._read_to_database(*args, **kwargs)

        # TODO DISCUSS add more structure properties
        lattice = self.lattice_vectors()
        if lattice.ndim != 2:
            lattice = [None, None, None]
        volume = None
        try:
            volume = self.volume()
        except Exception:
            pass

        return database.combine_db_dicts(
            {
                "structure": {
                    "total_ion_count": self.number_atoms(),
                    "cell_volume": volume,
                    "lattice_a": list(lattice[0]) if lattice[0] is not None else None,
                    "lattice_b": list(lattice[1]) if lattice[1] is not None else None,
                    "lattice_c": list(lattice[2]) if lattice[2] is not None else None,
                },
            },
            stoichiometry,
        )

    @base.data_access
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
        if not self._is_slice:
            message = "Converting a single structure to mdtraj is not implemented."
            raise exception.NotImplemented(message)
        data = self.to_dict(ion_types)
        xyz = data["positions"] @ data["lattice_vectors"] * self.A_to_nm
        trajectory = mdtraj.Trajectory(xyz, self._stoichiometry().to_mdtraj(ion_types))
        trajectory.unitcell_vectors = data["lattice_vectors"] * Structure.A_to_nm
        return trajectory

    @base.data_access
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
        if not self._is_slice:
            return self._create_repr(ion_types=ion_types)
        else:
            message = "Converting multiple structures to a POSCAR is currently not implemented."
            raise exception.NotImplemented(message)

    @base.data_access
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
        return self._cell().lattice_vectors()

    @base.data_access
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
        return self._raw_data.positions[self._get_steps()]

    @base.data_access
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
        return self.positions() @ self.lattice_vectors()

    @base.data_access
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

    def _cell(self):
        return cell.Cell.from_data(self._raw_data.cell)[self._steps]

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

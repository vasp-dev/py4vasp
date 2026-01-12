# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import importlib
import pathlib
from typing import Any, List, Optional, Tuple, Union

from py4vasp import exception
from py4vasp._raw.access import access
from py4vasp._raw.data import CalculationMetaData, DatabaseData
from py4vasp._raw.definition import schema, selections
from py4vasp._util import convert, import_

DEFAULT_VASP_DB_NAME = "vasp_database.db"

vaspdb = import_.optional("vaspdb")

INPUT_FILES = ("INCAR", "KPOINTS", "POSCAR")
QUANTITIES = (
    "band",
    "bandgap",
    "born_effective_charge",
    "current_density",
    "density",
    "dielectric_function",
    "dielectric_tensor",
    "dos",
    "effective_coulomb",
    "elastic_modulus",
    "electronic_minimization",
    "energy",
    "force",
    "force_constant",
    "internal_strain",
    "kpoint",
    "local_moment",
    "nics",
    "pair_correlation",
    "partial_density",
    "piezoelectric_tensor",
    "polarization",
    "potential",
    "projector",
    "stress",
    "structure",
    "system",
    "velocity",
    "workfunction",
    "_CONTCAR",
    "_dispersion",
    "_stoichiometry",
)
GROUPS = {
    "electron_phonon": ("bandgap", "chemical_potential", "self_energy", "transport"),
    "exciton": ("density", "eigenvector"),
    "phonon": ("band", "dos", "mode"),
}
GROUP_TYPE_ALIAS = {
    convert.to_camelcase(f"{group}_{member}"): f"{group}.{member}"
    for group, members in GROUPS.items()
    for member in members
}

AUTOSUMMARY_QUANTITIES = [
    (quantity, f"~py4vasp.Calculation.{quantity}")
    for quantity in QUANTITIES
    if not quantity.startswith("_")
]
AUTOSUMMARY_GROUPS = [
    (
        f"{group}.{member}",
        f"~py4vasp._calculation.{group}_{member}.{convert.to_camelcase(f'{group}_{member}')}",
    )
    for group, members in GROUPS.items()
    for member in members
]
AUTOSUMMARIES = sorted(AUTOSUMMARY_QUANTITIES + AUTOSUMMARY_GROUPS)

__all__ = QUANTITIES


class Calculation:
    """Provide refinement functions for a the raw data of a VASP calculation run in any directory.

    The :data:`~py4vasp.calculation` object always reads the VASP calculation from the current
    working directory. This class gives you a more fine grained control so that you
    can use a Python script or Jupyter notebook in a different folder or rename the
    files that VASP produces.

    To create a new instance, you should use the classmethod :meth:`from_path` or
    :meth:`from_file` and *not* the constructor. This will ensure that the path to
    your VASP calculation is properly set and all features work as intended.
    The two methods allow you to read VASP results from a specific folder other
    than the working directory or a nondefault file name.

    With the Calculation instance, you can access the quantities VASP computes via
    the attributes of the object. The attributes are the same provided by the
    :data:`~py4vasp.calculation` object. You can find links to how to use these quantities
    below.

    Examples
    --------

    Let's first create some example data in a temporary directory. Please define `path`
    as the path to a temporary directory that does not exist yet. This command create
    example data in that directory and will return a Calculation but we ignore the result.

    >>> _ = py4vasp.demo.calculation(path)

    We can now, generate a new calculation object to access the data from this path

    >>> calculation = Calculation.from_path(path)

    Plot the density of states (DOS) of the calculation

    >>> calculation.dos.plot()
    Graph(series=[Series(x=array(...), y=array(...), label='total', ...)],
        xlabel='Energy (eV)', ..., ylabel='DOS (1/eV)', ...)

    Read the energies for a structure relaxation run into a Python dictionary

    >>> calculation.energy[:].read()
    {'free energy    TOTEN': array(...), 'energy without entropy': array(...),
        'energy(sigma->0)': array(...)}

    Convert the structure to a POSCAR format

    >>> poscar_string = calculation.structure.to_POSCAR()
    """

    def __init__(self, *args, **kwargs):
        if not kwargs.get("_internal"):
            message = """\
Please setup new Calculation instances using the classmethod Calculation.from_path()
instead of the constructor Calculation()."""
            raise exception.IncorrectUsage(message)

    @classmethod
    def from_path(cls, path_name):
        """Set up a Calculation for a particular path and so that all files are opened there.

        py4vasp knows to which files the relevant information is written. It will
        automatically open the files as necessary and extract the required data from
        them. Then the raw data is refined according to the selected methods.

        Importantly, the creation of the Calculation object does not require that the
        VASP calculation was already finished. It does not even need the path to exist.
        All data is lazily loaded at the moment when it is needed. This also means that
        if you change the data, e.g. by rerunning VASP in the same path, py4vasp will
        directly read the new results. If you want to keep the old results, please run
        the new calculation in a new path.

        Parameters
        ----------
        path_name : str or pathlib.Path
            Name of the path associated with the calculation.

        Returns
        -------
        Calculation
            A calculation associated with the given path.

        Examples
        --------

        Create a new Calculation object from a specific path.

        >>> calculation = Calculation.from_path("path/to/calculation")

        You can also pass in pathlib Path objects or anything else that can be converted
        into it.

        >>> calculation = Calculation.from_path(pathlib.Path.cwd())
        """
        calc = cls(_internal=True)
        calc._path = pathlib.Path(path_name).expanduser().resolve()
        calc._file = None
        return calc

    @classmethod
    def from_file(cls, file_name):
        """Set up a Calculation from a particular file.

        Typically this limits the amount of information, you have access to, so prefer
        creating the instance with the :meth:`from_path` if possible. Most data is
        found in the vaspout.h5 file, so if you renamed it for backup purposes most
        functions of py4vasp will work, when you pass it into this constructor.

        Please keep in mind that creating a new Calculation will not read any data.
        You can create an instance for a specific file and create or modify it
        afterwards. py4vasp access the data in the moment when it is needed e.g. to
        generate a plot or read it to a dictionary. However, this also means that you
        need to make sure to keep track of any changes, because the Calculation object
        is always a representation of the current contents of the file not the ones
        at creation of the Calculation object.

        Parameters
        ----------
        file_name : str or pathlib.Path
            Name of the file from which the data is read.

        Returns
        -------
        Calculation
            A calculation accessing the data in the given file.

        Examples
        --------

        Create a new Calculation object to the vaspout.h5 file. For the most parts this
        is equivalent to the :meth:`from_path` method so you should typically use that
        instead.

        >>> calculation = Calculation.from_file("vaspout.h5")

        Sometime you rename the VASP output as a backup. Then the `from_file` constructor
        is your only option.

        >>> calculation = Calculation.from_file("path/to/file/backup.h5")
        """
        calc = cls(_internal=True)
        calc._path = pathlib.Path(file_name).expanduser().resolve().parent
        calc._file = file_name
        return calc

    def to_database(
        self,
        tags: Optional[Union[str, list[str]]] = None,
    ):
        """
        Retrieve the data of the calculation needed to write it to a VASP database.

        The actual database write is handled by external modules, e.g., the `vaspdb`
        package. This method prepares all the data that is needed for the database.

        Parameters
        ----------
        tags
            Tags to associate with the calculation in the database.
            Can be a single string or a list of strings.

        Examples
        --------
        Prepare the calculation data for the default database:

        >>> from py4vasp import Calculation
        >>> calculation = Calculation.from_path("path/to/calculation")
        >>> calc_data = calculation.to_database()

        Tag your calculation when writing it to the database:

        >>> from py4vasp import Calculation
        >>> calculation = Calculation.from_path("path/to/calculation")
        >>> calc_data = calculation.to_database(tags=["relaxation", "vaspdb", "testing some stuff"])
        """
        hdf5_path: pathlib.Path = self._path / (self._file or "vaspout.h5")

        # Check h5 file existence
        if not hdf5_path.exists():
            raise exception.FileAccessError(
                f"The HDF5 file {hdf5_path} does not exist."
            )

        # Obtain DatabaseData instance
        # Obtain runtime data from h5 file
        database_data = None
        with access("runtime_data", file=self._file, path=self._path) as runtime_data:
            database_data = DatabaseData(
                metadata=CalculationMetaData(
                    hdf5=hdf5_path,
                    tags=tags,
                    infer_none_files=True,
                    runtime_data=runtime_data,
                )
            )

        # Check available quantities and compute additional properties
        database_data.available_quantities, database_data.additional_properties = (
            self._compute_database_data()
        )

        # Return DatabaseData object for VaspDB to process
        return database_data

    def path(self):
        "Return the path in which the calculation is run."
        return self._path

    # Input files are not in current release
    # @property
    # def INCAR(self):
    #     "The INCAR file of the VASP calculation."
    #     return self._INCAR
    #
    # @INCAR.setter
    # def INCAR(self, incar):
    #     self._INCAR.write(str(incar))
    #
    # @property
    # def KPOINTS(self):
    #     "The KPOINTS file of the VASP calculation."
    #     return self._KPOINTS
    #
    # @KPOINTS.setter
    # def KPOINTS(self, kpoints):
    #     self._KPOINTS.write(str(kpoints))
    #
    # @property
    # def POSCAR(self):
    #     "The POSCAR file of the VASP calculation."
    #     return self._POSCAR
    #
    # @POSCAR.setter
    # def POSCAR(self, poscar):
    #     self._POSCAR.write(str(poscar))

    def _compute_quantity_db_data(
        self, group, selection, quantity_name, group_name=None
    ) -> Tuple[bool, dict]:
        "Compute additional data to be stored in the database."
        is_available = False
        additional_properties = {}
        quantity_data = None

        try:
            # attempt to read
            quantity_data = getattr(group, quantity_name).read(selection=str(selection))
            is_available = True
            # attempt to compute additional properties if any are requested
        except exception.NoData:
            pass  # happens when some required data is missing
        except exception.OutdatedVaspVersion:
            pass  # happens when VASP version is too old for this quantity
        except exception.FileAccessError:
            pass  # happens when vaspout.h5 or vaspwave.h5 (where relevant) are missing
        except Exception as e:
            # print(f"Unexpected error on {quantity_name} (group={type(group)}) with selection {selection}:", e)
            pass  # catch any other errors during reading

        if quantity_data is not None:
            try:
                with access(
                    str.strip(quantity_name, "_"),
                    selection=selection,
                    file=self._file,
                    path=self._path,
                ) as quantity_data:
                    try:
                        full_key_prefix = (
                            (f"{group_name}_" if group_name else "")
                            + (f"{quantity_name}_" if quantity_name else "")
                            + (
                                f"{selection}_"
                                if (selection and selection != "default")
                                else ""
                            )
                        )
                        additional_properties = {
                            f"{full_key_prefix}{k}": v
                            for k, v in quantity_data.to_database().items()
                        }
                    except:
                        pass
            except exception.FileAccessError as e:
                pass

            # TODO find a way to ensure properties are only re-computed if they don't already exist
            #       --> probably via private functions and setting private variables on first computation
            # TODO think about whether additional properties should be specified in data.py or the implementation class
            #       --> probably better to have it close to the implementation so that data.py stays clean
            #       --> so instead of to_dict, we implement a to_database function in the implementation class that calls _to_database
            #       --> TODO BUT then how do we avoid duplicate recomputes?
            #       --> duplicate recomputes can also occur right now, see e.g., structure.to_database being called multiple times
            #       --> unless I am mistaken, a new Structure instance is created every time structure.read() is called
            #       --> so I could write a wrapper function that calls a function for computing only if a given (specified) key does not yet exist in additional_properties
            #       --> TODO AND how do I ensure selections are correct when calling to_database?
            #       --> I guess I pass selection on the class implementation of to_database, via the base.data_access decorator.
        return is_available, additional_properties

    def _loop_quantities(
        self, quantities, available_quantities, additional_properties, group_name=None
    ):
        group_instance = self if group_name is None else getattr(self, group_name)
        for quantity in quantities:
            try:
                _selections = (
                    selections(quantity)
                    if group_name is None
                    else selections(f"{group_name}_{quantity}")
                )
            except exception.FileAccessError:
                _selections = ["default"]
            for selection in _selections:
                is_available, props = self._compute_quantity_db_data(
                    group_instance, selection, quantity, group_name
                )
                if is_available:
                    available_quantities.append(
                        (f"{group_name}." if group_name else "")
                        + quantity
                        + (
                            f":{selection}"
                            if (selection and selection != "default")
                            else ""
                        )
                    )
                    additional_properties = additional_properties | props
        return available_quantities, additional_properties

    def _compute_database_data(self) -> Tuple[List[str], dict]:
        "Return a list of all available quantities in the calculation."
        available_quantities = []
        additional_properties = {}
        available_quantities, additional_properties = self._loop_quantities(
            QUANTITIES, available_quantities, additional_properties
        )
        for group, quantities in GROUPS.items():
            available_quantities, additional_properties = self._loop_quantities(
                quantities,
                available_quantities,
                additional_properties,
                group_name=group,
            )
        return available_quantities, additional_properties


def _add_all_refinement_classes(calc):
    for quantity in QUANTITIES:
        setattr(calc, quantity, _make_property(quantity))
    for group, quantities in GROUPS.items():
        setattr(calc, group, _make_group(group, quantities))
    return calc


def _make_property(quantity):
    # Be careful when refactoring this code, if the class_ is in a scope where it gets
    # changed like in the _add_all_refinement_classes routine, it may overwrite the
    # previous setting and all properties point to the same quantity.
    class_name = convert.to_camelcase(quantity)
    module = importlib.import_module(f"py4vasp._calculation.{quantity}")
    class_ = getattr(module, class_name)

    def get_quantity(self):
        if self._file is None:
            return class_.from_path(self._path)
        else:
            return class_.from_file(self._file)

    return property(get_quantity, doc=class_.__doc__)


def _make_group(group_name, quantities):
    # The Group class for each group is constructed on the fly. Alternatively you could
    # create a Group for every instance needed and construct the required properties as
    # needed. It is important that they are distinct classes.
    def get_group(self):
        class Group:
            def __init__(self, calculation):
                self._path = calculation._path
                self._file = calculation._file

        for quantity in quantities:
            full_name = f"{group_name}_{quantity}"
            setattr(Group, quantity, _make_property(full_name))
        return Group(self)

    return property(get_group)


Calculation = _add_all_refinement_classes(Calculation)


class DefaultCalculationFactory:
    """Provide refinement functions for a the raw data of a VASP calculation run in the
    current directory.

    Usually one is not directly interested in the raw data that is produced but
    wants to produce either a figure for a publication or some post-processing of
    the data. `calculation` contains multiple quantities that enable these kinds of
    workflows by extracting the relevant data from the HDF5 file and transforming
    them into an accessible format.

    Generally, all quantities provide a `read` function that extracts the data from the
    HDF5 file and puts it into a Python dictionary. Where it makes sense in addition
    a `plot` function is available that converts the data into a figure for Jupyter
    notebooks. In addition, data conversion routines `to_X` may be available
    transforming the data into another format or file, which may be useful to
    generate plots with tools other than Python. For the specifics, please refer to
    the documentation of the individual quantities.

    `calculation` reads the raw data from the current directory and from the default
    VASP output files. With the :class:`~py4vasp.Calculation` class, you can tailor
    the location of the files to your needs and both have access to the same quantities.

    We demonstrate this setting up some example data in a temporary directory `path` and
    changing to it.

    >>> import os
    >>> from py4vasp import demo
    >>> _ = demo.calculation(path)
    >>> os.chdir(path)  # change to a temporary directory

    Then the two following examples are equivalent:

    .. rubric:: using :data:`~py4vasp.calculation` object

    >>> from py4vasp import calculation
    >>> calculation.dos.read()
    {'energies': array(...), 'total': array(...), 'fermi_energy': ...}

    .. rubric:: using :class:`~py4vasp.Calculation` class

    >>> from py4vasp import Calculation
    >>> calculation = Calculation.from_path(".")
    >>> calculation.dos.read()
    {'energies': array(...), 'total': array(...), 'fermi_energy': ...}

    In the latter example, you could directly provide a path and do not need to have
    the data in the current directory.
    """

    def __getattr__(self, attr):
        calc = Calculation.from_path(".")
        return getattr(calc, attr)

    def __setattr__(self, attr, value):
        calc = Calculation.from_path(".")
        return setattr(calc, attr, value)


# we use a factory instead of an instance of Calculation here so that changing the
# directory works -> calculation will always point to the current directory
calculation = DefaultCalculationFactory()

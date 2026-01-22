# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import importlib
import pathlib
from typing import Any, List, Optional, Tuple, Union

import h5py

from py4vasp import exception
from py4vasp._raw.access import access
from py4vasp._raw.data import CalculationMetaData, _DatabaseData
from py4vasp._raw.definition import (
    DEFAULT_SOURCE,
    schema,
    selections,
    unique_selections,
)
from py4vasp._raw.schema import Link
from py4vasp._util import convert, database, import_

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
    "run_info",
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

    def _to_database(
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

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)
        >>> calc_data = calculation._to_database()

        Tag your calculation when writing it to the database:

        >>> calc_data = calculation._to_database(tags=["relaxation", "vaspdb", "testing some stuff"])

        Notes
        -----
        To add a variable to the database data, implement a `_to_database` method
        in a `base.Refinery` subclass (see `base.Refinery._read_to_database` for reference) listed in QUANTITIES.
        For example, the `_calculation.energy.Energy` class implements such a method to add energy-related
        data to the database.

        If you do not know which quantity to add it to, consider `_calculation.run_info.RunInfo` if you can
        guarantee the quantity will always be available. If not, implement your own dataclass - just make sure
        to implement the base dataclass in `_raw.data`, add its schema in `_raw.definition`, write an implementation
        for `_calculation.your_quantity.YourQuantity` and add it to the QUANTITIES list.
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
            database_data = _DatabaseData(
                metadata=CalculationMetaData(
                    hdf5_original_path=hdf5_path,
                    tags=tags,
                    infer_none_files=True,
                )
            )

        # Check available quantities and compute additional properties
        database_data.available_quantities, database_data.additional_properties = (
            self._compute_database_data(hdf5_path)
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

    def _compute_database_data(
        self, hdf5_path: pathlib.Path
    ) -> Tuple[dict[str, tuple[bool, list[str]]], dict[str, dict]]:
        """Computes a dict of available py4vasp dataclasses and all available database data.

        Returns
        -------
        Tuple[dict[str, tuple[bool, list[str]]], dict[str, dict]]
            A tuple containing:
            - dict[str, tuple[bool, list[str]]]
                A dictionary indicating the availability of each quantity (and selection) in the calculation.
                The keys may take the form 'group.quantity', 'quantity', 'group.quantity:selection', or 'quantity:selection'.
                The default selection is always omitted from the key.
                Also includes a list of aliases for each quantity/selection combination.
            - dict[str, dict]
                A dictionary containing all additional properties to be stored in the database.
                The keys follow the same convention as above. The values are dictionaries with the actual data to be stored.
        """
        available_quantities = {}
        additional_properties = {}

        # clear cached calls to should_load
        database.should_load.cache_clear()

        # Obtain quantities
        # --- MAIN LOOP FOR QUANTITIES ---
        available_quantities, additional_properties = self._loop_quantities(
            hdf5_path, QUANTITIES, available_quantities, additional_properties
        )
        for group, quantities in GROUPS.items():
            available_quantities, additional_properties = self._loop_quantities(
                hdf5_path,
                quantities,
                available_quantities,
                additional_properties,
                group_name=group,
            )
        # --------------------------------

        # clear cached calls to should_load
        database.should_load.cache_clear()

        # post-process dictionary keys
        available_quantities = _clean_db_dict_keys(available_quantities)
        additional_properties = _clean_db_dict_keys(additional_properties)

        return available_quantities, additional_properties

    def _loop_quantities(
        self,
        hdf5_path: pathlib.Path,
        quantities,
        available_quantities,
        additional_properties,
        group_name=None,
    ) -> Tuple[dict[str, tuple[bool, list[str]]], dict[str, dict]]:
        group_instance = self if group_name is None else getattr(self, group_name)
        for quantity in quantities:
            try:
                _selections = (
                    unique_selections(quantity.lstrip("_"))
                    if group_name is None
                    else unique_selections(f"{group_name}_{quantity.lstrip('_')}")
                )
            except exception.FileAccessError:
                _selections = ["default"]

            for selection in _selections:
                is_available, props, aliases_ = self._compute_quantity_db_data(
                    hdf5_path,
                    group_instance,
                    selection,
                    quantity,
                    group_name,
                    additional_properties,
                )
                availability_key, _ = database.construct_database_data_key(
                    group_name, quantity, selection
                )
                available_quantities[availability_key] = (is_available, aliases_)
                if is_available:
                    additional_properties = database.combine_db_dicts(
                        additional_properties, props
                    )
        return available_quantities, additional_properties

    def _compute_quantity_db_data(
        self,
        hdf5_path: pathlib.Path,
        group,
        selection: Optional[str],
        quantity_name: str,
        group_name: Optional[str] = None,
        current_db: dict = {},
    ) -> Tuple[bool, dict, list[str]]:
        "Compute additional data to be stored in the database."
        is_available = False
        schema_quantity_name = quantity_name.lstrip("_")
        aliases_ = schema._aliases(
            (
                f"{group_name}_{schema_quantity_name}"
                if group_name is not None
                else schema_quantity_name
            ),
            selection,
        )
        additional_properties = {}

        try:
            # check if readable
            expected_key = (
                f"{group_name}_{schema_quantity_name}"
                if group_name is not None
                else schema_quantity_name
            )
            should_load = True
            with h5py.File(hdf5_path, "r") as h5file:
                should_load, _, should_attempt_read = database.should_load(
                    expected_key, selection, h5file, schema
                )

                if should_attempt_read:
                    raw_class = getattr(group, quantity_name)
            # should_load = True
            if should_load or should_attempt_read:
                if should_attempt_read:
                    # attempt to read; if it passes: available
                    # this is relevant for quantities that read from files other than h5
                    quantity_data = getattr(group, quantity_name).read(
                        selection=str(selection)
                    )
                is_available = True
            # attempt to compute additional properties if any are requested
        except exception.NoData:
            pass  # happens when some required data is missing
        except exception.OutdatedVaspVersion:
            pass  # happens when VASP version is too old for this quantity
        except exception.FileAccessError:
            pass  # happens when vaspout.h5 or vaspwave.h5 (where relevant) are missing
        except Exception as e:
            # print(
            #     f"[CHECK] Unexpected error on {quantity_name} (group={type(group)}) with selection {selection}:",
            #     e,
            # )
            pass  # catch any other errors during reading

        if is_available:
            try:
                additional_properties: dict[str, dict[str, Any]] = getattr(
                    group, quantity_name
                )._read_to_database(
                    selection=str(selection),
                    current_db=current_db,
                    original_group_name=group_name,
                )
            except Exception as e:
                raise Exception(
                    f"[ADD] Unexpected error on {quantity_name} (group={type(group)}) with selection {selection} (please consider filing a bug report):",
                    e,
                ) from e
                # pass  # catch any other errors during reading

            # TODO can I load POSCAR and CONTCAR with correct available_quantities representations on nested quantities?
        return is_available, additional_properties, aliases_


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


def _clean_db_dict_keys(
    dict_to_clean: dict, rid_default_selection: bool = True
) -> dict:
    if rid_default_selection:
        # Fix keys to remove default selection suffixes
        dict_to_clean = dict(
            zip(
                [
                    (
                        key
                        if not (key.endswith(f":{DEFAULT_SOURCE}"))
                        else key[: -len(f":{DEFAULT_SOURCE}")]
                    )
                    for key in dict_to_clean.keys()
                ],
                dict_to_clean.values(),
            )
        )

    # Find private quantities
    private_quantities = [
        (None, quantity) for quantity in QUANTITIES if quantity.startswith("_")
    ] + [
        (group, quantity)
        for group, quantities in GROUPS.items()
        for quantity in quantities
        if quantity.startswith("_")
    ]

    # Fix keys to change private quantity keys back to private
    relevant_keys = []
    for group, quantity in private_quantities:
        if group is None:
            expected_key = quantity.lstrip("_")
        else:
            expected_key = f"{group}.{quantity.lstrip('_')}"
        relevant_keys = relevant_keys + [
            key
            for key in dict_to_clean.keys()
            if key.startswith(f"{expected_key}:") or key == expected_key
        ]
    relevant_keys = set(relevant_keys)
    for key in relevant_keys:
        if key in dict_to_clean:
            dict_to_clean[f"_{key}"] = dict_to_clean.pop(key)

    # Fix keys to resolve group selections
    relevant_keys = []
    for group, _ in GROUPS.items():
        expected_key = group
        relevant_keys = [
            key for key in dict_to_clean.keys() if key.endswith(f":{group}")
        ]
        for key in relevant_keys:
            split1, split2 = key.rsplit(":", 1)
            dict_to_clean[f"{split2}._{split1}"] = dict_to_clean.pop(key)

    return dict_to_clean


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

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import contextlib
import importlib
import pathlib
from typing import Any, List, Optional, Tuple, Union

from py4vasp import exception
from py4vasp._calculation.dispatch import _REGISTRY, FileSource, Group
from py4vasp._raw.data import CalculationMetaData, _DatabaseData
from py4vasp._raw.data_db import __SCHEMA_VERSION__
from py4vasp._util import convert, import_, loadable


def _append_database_error(
    encountered_errors: dict[str, list[str]],
    key: str,
    error: Exception,
    context: str,
):
    message = f"{context} | {type(error).__name__}: {error}"
    encountered_errors.setdefault(key, []).append(message)


_REGISTRY_MODULES_IMPORTED = False

_SUPPRESSED_DB_EXCEPTIONS = (
    exception.Py4VaspError,
    exception.OutdatedVaspVersion,
    exception.NoData,
    exception.FileAccessError,
    AttributeError,
    TypeError,
    ValueError,
)


INPUT_FILES = ("INCAR", "KPOINTS", "POSCAR")
QUANTITIES = (
    "structure",
    "_stoichiometry",
)
GROUPS = {}
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
        calc._source = FileSource(calc._path)
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
        calc._source = FileSource(calc._path, file=file_name)
        return calc

    def _to_database(self):
        """Retrieve the data of the calculation needed to write it to a VASP database.

        The actual database write is handled by external modules, e.g., the `vaspdb`
        package. This method prepares all the data that is needed for the database.

        Examples
        --------
        Prepare the calculation data for the default database:

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)
        >>> calc_data = calculation._to_database()
        """
        metadata = CalculationMetaData(
            path=self._path,
            schema_version=__SCHEMA_VERSION__,
        )
        properties = self._compute_database_data()
        return _DatabaseData(metadata=metadata, properties=properties)

    def path(self):
        "Return the path in which the calculation is run."
        return self._path

    def selections(self, method: Optional[str] = None) -> dict[str, dict[str, str]]:
        """Determine which quantities and selections can be loaded for this calculation.

        This inspects the VASP output files of the calculation and compares them against
        the schema defined in :mod:`py4vasp._raw.definition`. For every quantity that
        py4vasp can access (e.g. ``"structure"``, ``"band"``, or grouped quantities like
        ``"exciton.density"``) it collects all selections (sources) whose data is
        actually present and loadable.

        Candidate selections are first filtered cheaply against the schema (only the
        existence of the relevant datasets is checked). Every remaining candidate is then
        confirmed by genuinely invoking the requested method, so the result only lists
        selections that truly load. Because the access convention differs between
        quantities (some take ``selection=...``, others are indexed via ``[...]``), the
        result reports, for each selection, a ready-to-evaluate snippet showing exactly
        how to obtain it. The snippets assume the calculation is bound to a variable
        named ``calculation``.

        Parameters
        ----------
        method : str, optional
            The method the snippets should call and that is used to confirm loadability.
            Defaults to ``"read"``. Pass e.g. ``"to_view"`` to restrict the result to
            quantities that can be visualized and to obtain plotting snippets.

        Returns
        -------
        dict[str, dict[str, str]]
            Maps each quantity call name to a dictionary that maps every loadable
            selection (the default source is reported as ``"default"``) to a
            ready-to-evaluate snippet calling *method*. Quantities for which no selection
            can be loaded are omitted entirely.

        Examples
        --------

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)
        >>> calculation.selections()
        {'band': {'default': 'calculation.band.read()', 'kpoints_opt': "calculation.band.read(selection='kpoints_opt')"}, ...}

        Obtain snippets that implement 3d visualization instead:

        >>> calculation.selections(method="to_view")
        {'density': {'default': 'calculation.density.to_view()'}, ...}
        """
        _ensure_all_quantities_imported()
        result = {}
        with contextlib.ExitStack() as stack:
            open_files = {}
            cache = {}
            for call_name, schema_name in _public_quantities():
                snippets = loadable.loadable_selections(
                    self,
                    call_name,
                    schema_name,
                    method,
                    open_files,
                    stack,
                    cache,
                    QUANTITIES,
                )
                if snippets:
                    result[call_name] = snippets
        return dict(sorted(result.items()))

    def __getattr__(self, name):
        # Only called when normal attribute lookup (including class-level properties
        # set by _add_all_refinement_classes) has already failed.
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in _REGISTRY:
            try:
                importlib.import_module(f"py4vasp._calculation.{name}")
            except ImportError:
                pass
        if name not in _REGISTRY:
            # Could be a group name (e.g. "electron_phonon") whose member modules
            # have different file names — import all to populate the full registry.
            _ensure_all_quantities_imported()
        if name in _REGISTRY:
            entry = _REGISTRY[name]
            if isinstance(entry, dict):
                return Group(self._source, entry)
            return entry(source=self._source, quantity_name=entry._quantity_name)
        raise AttributeError(f"'Calculation' has no attribute '{name}'")

    def __dir__(self):
        names = set(super().__dir__())
        names.update(_REGISTRY.keys())
        return sorted(names)

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

    def _compute_database_data(self) -> dict:
        """Iterate over all quantities in _REGISTRY and collect database properties.

        Returns a flat dict mapping property keys to handler result dicts.
        Keys use the format:
        - ``<quantity>`` for the default selection
        - ``<quantity>_<selection>`` for non-default selections
        - ``<group>_<quantity>`` / ``<group>_<quantity>_<selection>`` for groups
        Leading underscores are stripped from private quantity names.
        """
        _ensure_all_quantities_imported()
        properties = {}
        for name, entry in _REGISTRY.items():
            if isinstance(entry, dict):
                # group
                for member_name, dispatcher_cls in entry.items():
                    _collect_to_database(
                        name, member_name, dispatcher_cls, self._source, properties
                    )
            else:
                _collect_to_database(None, name, entry, self._source, properties)
        return properties


def _public_quantities():
    """List (call_name, schema_name) pairs for all user-facing quantities.

    Combines the dispatcher registry (new architecture) with the legacy ``QUANTITIES``
    that are not yet ported. Private quantities (leading underscore) are excluded.
    """
    pairs = []
    for key, entry in _REGISTRY.items():
        if key.startswith("_"):
            continue
        if isinstance(entry, dict):  # group of quantities, e.g. exciton.density
            for member, dispatcher_cls in entry.items():
                if member.startswith("_"):
                    continue
                pairs.append((f"{key}.{member}", dispatcher_cls._quantity_name))
        else:
            pairs.append((key, entry._quantity_name))
    for quantity in QUANTITIES:
        if quantity.startswith("_") or quantity in _REGISTRY:
            continue
        pairs.append((quantity, quantity))
    return pairs


def _ensure_all_quantities_imported():
    """Import all quantity modules so that _REGISTRY is fully populated."""
    calc_pkg = importlib.import_module("py4vasp._calculation")
    calc_dir = pathlib.Path(calc_pkg.__file__).parent
    for module_file in sorted(calc_dir.glob("*.py")):
        name = module_file.stem
        if name.startswith("__"):
            continue
        try:
            importlib.import_module(f"py4vasp._calculation.{name}")
        except Exception:
            pass


def _collect_to_database(group_name, quantity_name, dispatcher_cls, source, properties):
    """Call dispatcher._to_database() and merge results into *properties*."""
    base = quantity_name.lstrip("_")
    dispatcher = dispatcher_cls(
        source=source, quantity_name=dispatcher_cls._quantity_name
    )
    if not hasattr(dispatcher, "_to_database"):
        return
    try:
        result = dispatcher._to_database()
    except _SUPPRESSED_DB_EXCEPTIONS:
        return
    except Exception:
        return
    # result is {quantity[_selection]: handler_result}
    for key, handler_result in result.items():
        if group_name is not None:
            key = f"{group_name}_{key}"
        properties[key] = handler_result


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

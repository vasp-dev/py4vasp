# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import importlib
import pathlib
from typing import Any, List, Optional, Tuple, Union

from py4vasp import exception
from py4vasp._calculation.dispatch import (
    _REGISTRY,
    FileSource,
    Group,
    _availability_quantity_of,
)
from py4vasp._raw.data import CalculationMetaData, _DatabaseData
from py4vasp._raw.definition import unique_selections as _schema_unique_selections
from py4vasp._raw.models import schema_version
from py4vasp._util import convert, import_


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

# QUANTITIES, GROUPS, GROUP_TYPE_ALIAS, AUTOSUMMARY_QUANTITIES, AUTOSUMMARY_GROUPS,
# AUTOSUMMARIES, and __all__ are derived from the dispatcher _REGISTRY by
# _rebuild_public_registry_views() at the bottom of this module.


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
            schema_version=schema_version(),
        )
        properties = self._compute_database_data()
        return _DatabaseData(metadata=metadata, properties=properties)

    def path(self):
        "Return the path in which the calculation is run."
        return self._path

    def selections(
        self, method: Optional[str] = None, only_available: bool = False
    ) -> dict[str, list[str]]:
        """Determine which quantities and selections this calculation exposes.

        For every quantity that py4vasp can access (e.g. ``"structure"``, ``"band"``,
        or grouped quantities like ``"exciton.density"``) this collects the selections
        (sources) defined in the schema. Only the schema and the existence of the
        relevant datasets are inspected; the data itself is never loaded. There are
        some exceptions for quantities & methods that require knowledge of specific data
        to determine whether they might fail, but even then only the relevant subset
        of the data is loaded.

        Parameters
        ----------
        method : str, optional
            Restrict the result to quantities that implement this method, e.g.
            ``"to_view"``. Defaults to ``None``, which includes all quantities.
        only_available : bool, optional
            If False (default), report all schema-defined selections for each
            quantity. If True, report only the selections whose data is actually
            present in the output (via :meth:`is_available`), omitting quantities
            without an available selection.

        Returns
        -------
        dict[str, list[str]]
            Maps each quantity call name to a list of selection names (the primary
            source is reported as ``"default"``).

        Examples
        --------

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        Get all public quantities and their schema-defined selections (default):

        >>> calculation.selections()
        {'band': ['default', 'kpoints_opt', 'kpoints_wan'], ...}

        Restrict to quantities implementing a specific method:

        >>> calculation.selections(method="to_view")
        {...}

        Report only the quantities and selections whose data is present:

        >>> calculation.selections(only_available=True)
        {...}
        """
        _ensure_all_quantities_imported()
        result = {}
        for call_name, schema_name in _public_quantities():
            quantity = _quantity_object(self, call_name)
            if method is not None and not _implements(quantity, method):
                continue
            sources = _sources_for(quantity, schema_name)
            if only_available:
                availability = quantity.is_available(sources, method=method)
                sources = [source for source in sources if availability.get(source)]
                if not sources:
                    continue
            result[call_name] = sources
        return dict(sorted(result.items()))

    def is_available(self, method: Optional[str] = None) -> dict[str, dict[str, bool]]:
        """Report which quantities and selections are available for this calculation.

        For every quantity (and every one of its sources), this checks whether the
        data needed is present in the VASP output, comparing against the schema
        without loading the (potentially large) arrays. The result is a nested
        dictionary that mirrors the database layout, so it can be stored and
        filtered later.

        Parameters
        ----------
        method : str, optional
            Restrict the report to quantities implementing this method (e.g.
            ``"to_view"``) and evaluate availability for that method. Defaults to
            ``None``, which reports every quantity for its ``read`` method.

        Returns
        -------
        dict[str, dict[str, bool]]
            Maps each quantity call name to a dictionary of ``{source: available}``,
            e.g. ``{"structure": {"default": True, "final": False}, ...}``.

        Examples
        --------
        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)
        >>> calculation.is_available()
        {'band': {...}, ...}
        """
        _ensure_all_quantities_imported()
        result = {}
        for call_name, schema_name in _public_quantities():
            quantity = _quantity_object(self, call_name)
            if method is not None and not _implements(quantity, method):
                continue
            sources = _sources_for(quantity, schema_name)
            result[call_name] = quantity.is_available(sources, method=method)
        return dict(sorted(result.items()))

    def __getattr__(self, name):
        # Resolves a quantity (or group) by name from the dispatcher _REGISTRY. Called
        # only when normal attribute lookup has already failed.
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in _REGISTRY:
            module_name = f"py4vasp._calculation.{name}"
            try:
                importlib.import_module(module_name)
            except ImportError as err:
                # Missing quantity modules are expected for unknown names; however,
                # re-raise ImportError originating from inside an existing module.
                if err.name != module_name:
                    raise
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

        Returns a nested dict ``{quantity: {selection: model}}``. The outer key is
        the (underscore-stripped) quantity name; the inner dict is keyed by
        selection with the default source keyed ``"default"``. Group members use
        ``<group>_<quantity>`` (e.g. ``phonon_mode``) as their outer key.
        """
        _ensure_all_quantities_imported()
        properties = {}
        for entry in _REGISTRY.values():
            members = entry.values() if isinstance(entry, dict) else [entry]
            for dispatcher_cls in members:
                _collect_to_database(dispatcher_cls, self._source, properties)
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


def _quantity_object(calculation, call_name):
    """Resolve a (possibly grouped) call name to its quantity dispatcher."""
    if "." in call_name:
        group_name, member = call_name.split(".", 1)
        return getattr(getattr(calculation, group_name), member)
    return getattr(calculation, call_name)


def _implements(quantity, method):
    """Return whether the quantity provides the requested method."""
    return callable(getattr(quantity, method, None))


def _sources_for(quantity, schema_name):
    """Return the schema sources of the quantity that actually holds the data.

    Derived quantities (e.g. ``optics``) read another quantity's data, so their
    sources come from that quantity rather than their own (empty) schema entry.
    """
    try:
        availability_quantity = _availability_quantity_of(quantity)
    except AttributeError:
        availability_quantity = schema_name
    try:
        return list(_schema_unique_selections(availability_quantity))
    except exception.FileAccessError:
        return []


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


def _collect_to_database(dispatcher_cls, source, properties):
    """Call dispatcher._to_database() and deep-merge results into *properties*.

    Each dispatcher returns a nested ``{quantity: {selection: model}}`` dict keyed by
    the full quantity name; group members use ``<group>_<member>`` via their
    ``_quantity_name`` (e.g. ``phonon_mode``). The nested dicts are merged so that
    quantities and their selections accumulate without clobbering each other.
    """
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
    for quantity, selections in result.items():
        properties.setdefault(quantity, {}).update(selections)


def _rebuild_public_registry_views():
    """Derive the public quantity/group views from the dispatcher ``_REGISTRY``.

    These module-level names drive documentation generation (``_sphinx``) and database
    key extraction (``_util.database``). They are computed from ``_REGISTRY`` so that
    every public dispatcher quantity is exposed without a separate hardcoded list.
    Private quantities (leading-underscore registry keys) are excluded.
    """
    global QUANTITIES, GROUPS, GROUP_TYPE_ALIAS
    global AUTOSUMMARY_QUANTITIES, AUTOSUMMARY_GROUPS, AUTOSUMMARIES, __all__
    _ensure_all_quantities_imported()
    QUANTITIES = tuple(
        sorted(
            name
            for name, entry in _REGISTRY.items()
            if not isinstance(entry, dict) and not name.startswith("_")
        )
    )
    GROUPS = {
        group: tuple(sorted(m for m in members if not m.startswith("_")))
        for group, members in _REGISTRY.items()
        if isinstance(members, dict) and not group.startswith("_")
    }
    GROUP_TYPE_ALIAS = {
        convert.to_camelcase(f"{group}_{member}"): f"{group}.{member}"
        for group, members in GROUPS.items()
        for member in members
    }
    AUTOSUMMARY_QUANTITIES = [
        (quantity, f"~py4vasp.Calculation.{quantity}") for quantity in QUANTITIES
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


_rebuild_public_registry_views()


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

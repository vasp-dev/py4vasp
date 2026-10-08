# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import importlib
import pathlib
import pkgutil
import warnings
from typing import Any, List, Optional, Tuple, Union

from py4vasp import exception
from py4vasp._calculation.dispatch import (
    _REGISTRY,
    INPUT_FILES,
    ArchiveSource,
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


_SUPPRESSED_DB_EXCEPTIONS = (
    exception.Py4VaspError,
    exception.OutdatedVaspVersion,
    exception.NoData,
    exception.FileAccessError,
    AttributeError,
    TypeError,
    ValueError,
)


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

    Let's first create some example data in a temporary directory and print where it
    is, so that you can look at the files. Keep the returned calculation around, because
    the temporary directory is removed once it is no longer used.

    >>> example = py4vasp.demo.calculation()
    >>> print("The example data is in", example.path())
    The example data is in ...

    We can now generate a new calculation object to access the data from this path

    >>> calculation = Calculation.from_path(example.path())

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
        return cls._from_source(FileSource(path_name))

    @classmethod
    def _from_source(cls, source, file=None):
        "Set up a Calculation that reads its data from the given source."
        calc = cls(_internal=True)
        calc._source = source
        calc._path = source.path
        calc._file = file
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
        file_path = pathlib.Path(file_name).expanduser().resolve()
        source = FileSource(file_path.parent, file=file_path.name)
        return cls._from_source(source, file=file_path.name)

    @classmethod
    def from_archive(cls, archive_name, path=None, file=None):
        """Set up a Calculation from a VASP calculation stored in an archive.

        If you archive a finished VASP calculation as a zip or tar file, you can
        analyze it without unpacking it first. The archive may contain the files of the
        calculation directly (INCAR, vaspout.h5, ...) or inside a directory
        (folder/INCAR, folder/vaspout.h5, ...). py4vasp finds the calculation in either
        case. If the archive contains more than one calculation, use the path argument
        to select one of them; py4vasp reports the available choices otherwise.

        The files that py4vasp reads are extracted to a temporary directory when you
        access the data for the first time. Large files that py4vasp does not read, e.g.
        the WAVECAR, remain in the archive. The temporary directory is removed when the
        Calculation is deleted, so if you want to work with the files directly you
        should unpack the archive yourself and use :meth:`from_path`.

        Note that :meth:`path` reports the directory in which the archive is stored and
        not the temporary directory. This way any output that py4vasp generates, e.g.
        an image of a plot, is written next to the archive.

        Parameters
        ----------
        archive_name : str or pathlib.Path
            Name of the archive containing the VASP calculation. py4vasp reads zip and
            tar archives including the compressed variants tar.gz (tgz), tar.bz2, and
            tar.xz. The format is determined from the content of the file so that a
            renamed archive is read correctly, too.
        path : str or pathlib.Path, optional
            Directory inside the archive in which the calculation is stored. You only
            need this if the archive contains more than one calculation.
        file : str or pathlib.Path, optional
            Name of the file inside the archive from which the data is read. Use this
            if you renamed the vaspout.h5 file before archiving the calculation.

        Returns
        -------
        Calculation
            A calculation accessing the data inside the archive.

        Examples
        --------

        Let's create an example calculation in a new temporary directory and archive it.
        The directory is printed so that you can look at the archive.

        >>> import pathlib, shutil, tempfile
        >>> path = pathlib.Path(tempfile.mkdtemp())
        >>> print("The archive is created in", path)
        The archive is created in ...
        >>> _ = py4vasp.demo.calculation(path / "data" / "calculation")
        >>> archive = shutil.make_archive(str(path / "archive"), "zip", path / "data")

        You can now analyze the data in the archive without unpacking it.

        >>> calculation = Calculation.from_archive(archive)
        >>> calculation.dos.plot()
        Graph(series=[Series(x=array(...), y=array(...), label='total', ...)],
            xlabel='Energy (eV)', ..., ylabel='DOS (1/eV)', ...)

        If the archive contains more than one calculation, select one with the path.

        >>> calculation = Calculation.from_archive(archive, path="calculation")
        """
        source = ArchiveSource(archive_name, path=path, file=file)
        return cls._from_source(source, file=file)

    def _to_database(self):
        """Retrieve the data of the calculation needed to write it to a VASP database.

        The actual database write is handled by external modules, e.g., the `vaspdb`
        package. This method prepares all the data that is needed for the database.

        Examples
        --------
        Prepare the calculation data for the default database:

        >>> from py4vasp import demo
        >>> calculation = demo.calculation()
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
        >>> calculation = demo.calculation()

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
        >>> calculation = demo.calculation()
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
    # Ask the package's loader rather than the filesystem, so discovery also works
    # when the modules are not .py files on disk (zipimport, PyInstaller).
    names = sorted(module.name for module in pkgutil.iter_modules(calc_pkg.__path__))
    if not names:
        message = (
            "The package loader could not list the modules of py4vasp._calculation, "
            "so no quantities are registered and Calculation.selections() and "
            "py4vasp._calculation.QUANTITIES will be empty. Accessing a quantity "
            "directly, e.g. calculation.structure, still works."
        )
        warnings.warn(message, UserWarning)
    for name in names:
        importlib.import_module(f"py4vasp._calculation.{name}")


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

    We demonstrate this by setting up some example data in a temporary directory and
    changing to it. Keep the returned calculation around, because the temporary
    directory is removed once it is no longer used.

    >>> import os
    >>> from py4vasp import demo
    >>> example = demo.calculation()
    >>> os.chdir(example.path())

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

    .. rubric:: Common tasks

    The quantities below are listed by name, which only helps once you know the name.
    These are the questions people arrive with, and the call that answers each.

    *How far apart are these atoms?* Use
    :py:class:`~py4vasp._calculation.neighbor_list.NeighborList`. It takes the periodic
    images into account and measures the perpendicular width of the cell, so it stays
    correct for the tilted cells where the minimum-image convention ``d - np.rint(d)``
    silently does not

    >>> calculation.neighbor_list.selections()
    ['Sr~Sr', 'Sr~Ti', 'Sr~O', 'Ti~Sr', 'Ti~Ti', 'Ti~O', 'O~Sr', 'O~Ti', 'O~O']

    Each of these is a selection for ``read(selection, cutoff=...)``, which returns the
    distance, the distance vector and the periodic image of every pair within the cutoff.

    *How do I build the supercell for a finite-difference phonon run?* Replicate the
    cell while writing the structure file. You do not need a finished VASP run to start
    from: :py:meth:`~py4vasp._calculation.structure.Structure.from_POSCAR` reads a
    POSCAR you already have

    >>> poscar = calculation.structure.to_POSCAR()
    >>> structure = py4vasp.calculation.structure.from_POSCAR(poscar)
    >>> structure.to_POSCAR(supercell=(2, 2, 1)).splitlines()[6]
    '8 4 16'

    *How do I pass ε∞ and the Born effective charges to a polar phonon run?* Let the
    linear-response calculation write the INCAR tags for you. Both strings end with a
    newline, so you can join them and append them to the INCAR file of the phonon
    calculation, where you also switch on the polar correction with
    ``LPHON_POLAR = .TRUE.``. The orientation of the tensors is already the one VASP
    reads

    >>> tags = (
    ...     calculation.dielectric_tensor.to_INCAR()
    ...     + calculation.born_effective_charge.to_INCAR()
    ... )
    >>> [line.split()[0] for line in tags.splitlines() if "=" in line]
    ['PHON_DIELECTRIC', 'PHON_BORN_CHARGES']

    Append them to the INCAR of the *phonon* run, not to the one of the linear-response
    calculation you read them from. Here the phonon run lives in a subdirectory of the
    example data

    >>> phonon_run = example.path() / "phonon"
    >>> phonon_run.mkdir(exist_ok=True)
    >>> print("The phonon run is in", phonon_run)
    The phonon run is in ...phonon
    >>> with open(phonon_run / "INCAR", "a") as incar:
    ...     _ = incar.write("LPHON_POLAR = .TRUE.\\n" + tags)

    *How do I average a tensor over the directions?* Select the average instead of
    computing it, here for the dielectric function

    >>> sorted(calculation.dielectric_function.read("isotropic"))
    ['energies', 'isotropic']

    *Which elastic constants does my crystal have, in GPa?* Read the
    :py:class:`~py4vasp._calculation.elastic_modulus.ElasticModulus` and divide by ten,
    because py4vasp returns it in kBar as VASP writes it. The result is the full tensor
    with four Cartesian indices, so C_11 is ``[0, 0, 0, 0]``, C_12 is ``[0, 0, 1, 1]``
    and C_44 is ``[1, 2, 1, 2]``

    >>> elastic_modulus = calculation.elastic_modulus.read()["relaxed_ion"] / 10
    >>> c11, c12, c44 = (0, 0, 0, 0), (0, 0, 1, 1), (1, 2, 1, 2)
    >>> [float(elastic_modulus[index]) for index in (c11, c12, c44)]
    [297.0, 119.0, 57.0]

    *How do I compare several calculations in one figure?* Add the graphs together.
    :py:meth:`~py4vasp.graph.Graph.label` names each contribution, and the axis labels
    and ticks are reconciled for you

    >>> coarse = calculation.dos.plot().label("coarse mesh")
    >>> dense = calculation.dos.plot().label("dense mesh")
    >>> [series.label for series in coarse + dense]
    ['coarse mesh', 'dense mesh']

    *How do I watch a phonon mode move?* Let the viewer animate the eigenvector rather
    than reading a table of frequencies

    >>> view = calculation.phonon.mode.to_view()
    >>> view.phonon.frequencies.shape
    (1, 21)

    *How do I get vibrational frequencies from the force constants?* The force
    constants diagonalize into ħω in eV once the masses enter; multiply by 8065.610420
    for cm⁻¹. ``to_molden`` writes the same modes for a molecular viewer

    >>> frequencies = calculation.force_constant.frequencies()
    >>> round(float(frequencies[-1].real * 8065.610420))
    647
    >>> molden = calculation.force_constant.to_molden()

    *How do I turn a list of peaks into a spectrum?* Give every peak a width and add
    them up with :py:mod:`py4vasp.broadening`, which is what every quantity that plots
    a spectrum uses

    >>> import numpy as np
    >>> from py4vasp.broadening import broaden, Lorentzian
    >>> mesh = np.linspace(0, 10, 200)
    >>> spectrum = broaden(mesh, [3.0, 7.0], [1.0, 2.0], shape=Lorentzian(fwhm=0.5))
    >>> spectrum.shape
    (200,)

    *How strongly does each vibration scatter light?* Read the Raman tensor, which
    :py:class:`~py4vasp._calculation.raman.Raman` averages over the orientations of a
    crystallite for you rather than leaving you to write the invariants out

    >>> activity = calculation.raman.activity()
    >>> sorted(activity)
    ['frequencies', 'laser', 'powder']

    Pass a laser energy in eV to see how the lines change as it approaches an
    electronic transition, and a temperature to get the intensity a spectrometer
    measures rather than the bare activity.
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

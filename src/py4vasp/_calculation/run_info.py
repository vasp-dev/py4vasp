# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from contextlib import contextmanager

import numpy as np

from py4vasp import raw
from py4vasp._calculation import exception
from py4vasp._calculation.dispatch import (
    DataSource,
    _dispatch,
    merge_default,
    merge_strings,
    merge_to_database,
    quantity,
)
from py4vasp._raw.data_wrapper import VaspData
from py4vasp._raw.models import RunInfoModel
from py4vasp._util import check, convert

# core settings of raw.RunInfo; a key that is exactly an INCAR tag is spelled in upper
# case, every other one in lower case, both here and as field of raw.RunInfo
_CORE_SETTINGS = (
    "SIGMA",
    "ISPIN",
    "ISTART",
    "ICHARG",
    "NELM",
    "LMAXMIX",
    "POTIM",
    "TEBEG",
    "TEEND",
    "MDALGO",
    "ISYM",
    "LORBIT",
    "IMAGES",
    "peak_memory",
    "NCORE",
    "KPAR",
    "IBRION",
    "ISIF",
    "NSW",
    "EDIFFG",
    "ENCUT",
    "EDIFF",
    "NBANDS",
    "nkpts",
    "NELECT",
    "ISMEAR",
    "AEXX",
    "HFSCREEN",
    "IVDW",
    "NUPDOWN",
    "IDIPOL",
    "mpi_ranks",
    "wall_time",
    "cpu_time",
    "omp_threads",
    "nodes",
    "gpus",
)
_CORE_LOGICALS = (
    "LDAU",
    "LNONCOLLINEAR",
    "LSORBIT",
    "LKPOINTS_OPT",
    "LASPH",
    "LHFCALC",
    "LDIPOL",
    "LEPSILON",
    "build_gpu",
    "build_openmp",
)
# a bug in VASP is the most severe message, so it counts as an error as well
_ERROR_SEVERITIES = ("error", "bug")
_REQUIRES_NEWER_VASP = "unknown (requires a newer VASP)"
# print groups the core settings like the sections VASP writes them to, so that the
# units can be looked up in the extensions; the keys are the ones of read()
_PHONON_KEYS = (
    "phonon_method",
    "phonon_num_modes",
    "phonon_num_qpoints",
    "phonon_num_branches",
)
_PRINTED_GROUPS = (
    (
        "ionic",
        "ionic",
        (
            "IBRION",
            "ISIF",
            "NSW",
            "EDIFFG",
            "POTIM",
            "TEBEG",
            "TEEND",
            "MDALGO",
            "ISYM",
        ),
    ),
    (
        "electronic",
        "electronic",
        (
            *("ENCUT", "PREC", "ALGO", "EDIFF", "NELM", "NELECT", "NUPDOWN", "ISPIN"),
            *("LNONCOLLINEAR", "LSORBIT", "LASPH", "LREAL", "LMAXMIX", "ISTART"),
            "ICHARG",
        ),
    ),
    ("dimensions", "dimensions", ("NBANDS", "nkpts")),
    ("dos", "dos", ("ISMEAR", "SIGMA", "EFERMI")),
    (
        "exchange_correlation",
        "exchange_correlation",
        ("XC", "LDAU", "LHFCALC", "AEXX", "HFSCREEN"),
    ),
    ("vdw", "vdw", ("IVDW",)),
    ("dipole", "dipole", ("LDIPOL", "IDIPOL")),
    ("linear_response", "linear_response", ("LEPSILON",)),
    ("kpoints_opt", "kpoints_opt", ("LKPOINTS_OPT",)),
    ("write_flags", "write_flags", ("LORBIT",)),
    (
        "resources",
        "metadata/resources",
        ("mpi_ranks", "omp_threads", "nodes", "gpus", "KPAR", "NCORE", "IMAGES"),
    ),
    ("performance", "metadata/performance", ("wall_time", "cpu_time", "peak_memory")),
    ("build", "metadata/build", ("build_variant", "build_gpu", "build_openmp")),
)
_CORE_STRINGS = (
    "LREAL",
    "PREC",
    "ALGO",
    "XC",
    "build_variant",
)


class RunInfoHandler:
    """Handler for run info. Works with exactly one raw.RunInfo object."""

    def __init__(self, raw_run_info: raw.RunInfo):
        self._raw_run_info = raw_run_info

    @classmethod
    def from_data(cls, raw_run_info: raw.RunInfo) -> "RunInfoHandler":
        return cls(raw_run_info)

    def to_dict(self) -> dict:
        """Convert the run information to a dictionary."""
        return {
            **self._dict_from_runtime(),
            **self._dict_from_system(),
            **self._dict_from_structure(),
            **self._dict_additional_collection(),
            **self._dict_from_selective_dynamics(),
            **self._dict_from_phonons(),
            **self._dict_from_messages(),
            "extensions": self._dict_from_extension(),
        }

    def _dict_from_extension(self) -> dict:
        "All settings of the /setup group as {section: {KEY: value}}."
        extension = getattr(self._raw_run_info, "extension", None)
        if check.is_none(extension):
            return {}
        return {
            name: dict(section.values) for name, section in extension.sections.items()
        }

    def __str__(self) -> str:
        data = self.to_dict()
        system = data["system_tag"]
        header = f"run info for {system}:" if system else "run info:"
        lines = [header, *self._summary_lines(data)]
        return "\n".join(lines)

    def _summary_lines(self, data):
        """Report the quantities VASP wrote, skipping the ones it did not."""
        version = data["vasp_version"]
        if version is not None:
            yield f"    VASP version: {convert.text_to_string(version)}"
        # VASP writes the exit status at startup, so without it the file is too old
        has_setup = data["exit_status"] is not None
        exit_status = data["exit_status"] if has_setup else _REQUIRES_NEWER_VASP
        yield f"    exit status: {exit_status}"
        if data["num_ionic_steps"] is not None:
            yield f"    ionic steps: {data['num_ionic_steps']}"
        spin = self._spin_string(data)
        if spin is not None:
            yield f"    spin: {spin}"
        if data["XC"] is not None:
            yield f"    functional: {data['XC']}" + ("+U" if data["LDAU"] else "")
        yield from self._group_lines(data, "phonons", "phonons", _PHONON_KEYS)
        if has_setup:
            yield from self._setup_summary_lines(data)
        else:
            yield f"    settings: {_REQUIRES_NEWER_VASP}"

    def _setup_summary_lines(self, data):
        coarse, fine = data["grid_coarse_shape"], data["grid_fine_shape"]
        if coarse is not None:
            grid = " x ".join(map(str, coarse))
            if fine is not None:
                grid += f" (fine {' x '.join(map(str, fine))})"
            yield f"    FFT grid: {grid}"
        for group, section, keys in _PRINTED_GROUPS:
            yield from self._group_lines(data, group, section, keys)
        yield from self._error_message_lines(data)
        if data["extensions"]:
            yield f"    extensions: {', '.join(sorted(data['extensions']))}"

    @staticmethod
    def _error_message_lines(data):
        messages, counts = data["error_messages"], data["error_message_counts"]
        if messages is None:
            return
        if not messages:
            yield "    error messages: none"
            return
        yield "    error messages:"
        for message, count in zip(messages, counts):
            yield f"        - {message} ({count}×)"

    def _group_lines(self, data, group, section, keys):
        "Print a group of keys in YAML style; leave it out if VASP wrote none of them."
        lines = [
            f"        {key}: {data[key]}" + self._unit_suffix(section, key)
            for key in keys
            if data[key] is not None
        ]
        if lines:
            yield f"    {group}:"
            yield from lines

    def _unit_suffix(self, section, key):
        "The units VASP attached to the dataset, if any."
        extension = getattr(self._raw_run_info, "extension", None)
        if check.is_none(extension) or section not in extension.sections:
            return ""
        unit = extension.sections[section].units.get(key)
        return f" {unit}" if unit else ""

    @staticmethod
    def _spin_string(data):
        if data["LNONCOLLINEAR"]:
            return "noncollinear"
        if data["LNONCOLLINEAR"] is None or data["ISPIN"] is None:
            return None
        return "collinear" if data["ISPIN"] == 2 else "nonpolarized"

    def to_database(self) -> dict:
        """Serialize run info for the database; only the core settings are stored."""
        data = self.to_dict()
        data.pop("extensions")
        return RunInfoModel(**data)

    def _dict_additional_collection(self) -> dict:
        return {
            **self._dict_from_setup(),
            "EFERMI": self._efermi(),
            "fermi_energy": self._setup_value("fermi_energy"),
        }

    def _efermi(self):
        """The EFERMI tag as set in the INCAR file: MIDGAP, LEGACY, or a number.

        VASP always writes the method, which is FORCE if the user set a number. Only
        then it writes the number, in eV, as well. A database column holds a single
        type, so the number is reported as text, too."""
        value = self._setup_value("EFERMI")
        if value is not None:
            return str(value)
        return self._setup_string("efermi_method")

    def _dict_from_setup(self) -> dict:
        return {
            "grid_coarse_shape": self._setup_grid("NGX", "NGY", "NGZ"),
            "grid_fine_shape": self._setup_grid("NGXF", "NGYF", "NGZF"),
            "exit_status": self._setup_string("exit_status"),
            **{key: self._setup_value(key) for key in _CORE_SETTINGS},
            **{key: self._setup_bool(key) for key in _CORE_LOGICALS},
            **{key: self._setup_string(key) for key in _CORE_STRINGS},
        }

    def _dict_from_messages(self) -> dict:
        """Collect the messages of severity error or bug.

        VASP writes the messages together with their number when it stops regularly
        and omits the arrays if there are none. A killed run wrote only the exit
        status, so its messages are unknown."""
        number_messages = self._setup_value("number_messages")
        if number_messages is None:
            return {"error_messages": None, "error_message_counts": None}
        if number_messages == 0:
            return {"error_messages": [], "error_message_counts": []}
        raw_run_info = self._raw_run_info
        arrays = (
            raw_run_info.message_text,
            raw_run_info.message_severity,
            raw_run_info.message_count,
        )
        messages = zip(*(np.atleast_1d(_unwrap(array)) for array in arrays))
        errors = [
            (convert.fortran_text_to_string(text), int(count))
            for text, severity, count in messages
            if convert.fortran_text_to_string(severity) in _ERROR_SEVERITIES
        ]
        return {
            "error_messages": [text for text, _ in errors],
            "error_message_counts": [count for _, count in errors],
        }

    def _setup_bool(self, key):
        "VASP stores logicals as integers 0/1."
        value = self._setup_value(key)
        return None if value is None else bool(value)

    def _setup_string(self, key):
        "VASP pads fixed-length strings with blanks."
        value = self._setup_value(key)
        return None if value is None else convert.fortran_text_to_string(value)

    def _setup_value(self, key):
        "Return a scalar of the core setup as a Python object, None if VASP did not write it."
        value = getattr(self._raw_run_info, key)
        if check.is_none(value):
            return None
        value = _unwrap(value)
        return value.item() if hasattr(value, "item") else value

    def _setup_grid(self, *keys):
        shape = [self._setup_value(key) for key in keys]
        return None if None in shape else shape

    def _dict_from_system(self) -> dict:
        system_tag = self._raw_run_info.system_tag
        if check.is_none(system_tag):
            return {"system_tag": None}
        return {"system_tag": convert.fortran_text_to_string(_unwrap(system_tag))}

    def _dict_from_runtime(self) -> dict:
        version = self._raw_run_info.vasp_version
        if check.is_none(version):
            return {"vasp_version": None}
        parts = (version.major, version.minor, version.patch)
        if any(check.is_none(part) for part in parts):
            return {"vasp_version": None}
        return {"vasp_version": ".".join(str(int(_unwrap(part))) for part in parts)}

    def _dict_from_structure(self) -> dict:
        steps = self._raw_run_info.number_ionic_steps
        return {"num_ionic_steps": None if check.is_none(steps) else int(steps)}

    def _dict_from_selective_dynamics(self) -> dict:
        "Older VASP versions write the flag only for the POSCAR file."
        selective_dynamics = self._setup_bool("selective_dynamics")
        if selective_dynamics is None:
            selective_dynamics = self._setup_bool("poscar_selective_dynamics")
        return {"has_selective_dynamics": selective_dynamics}

    def _dict_from_phonons(self) -> dict:
        """VASP versions without the setup of the phonons only write the dispersion,
        whose shape gives the number of q points and branches."""
        qpoints = self._setup_value("phonon_num_qpoints")
        branches = self._setup_value("phonon_num_branches")
        frequencies = self._raw_run_info.phonon_frequencies
        if qpoints is None and branches is None and not check.is_none(frequencies):
            qpoints, branches = frequencies.shape
        return {
            "phonon_method": self._setup_string("phonon_method"),
            "phonon_num_modes": self._setup_value("phonon_num_modes"),
            "phonon_num_qpoints": qpoints,
            "phonon_num_branches": branches,
        }


def _unwrap(data):
    return data[()] if isinstance(data, VaspData) else data


@contextmanager
def _hint_at_extensions(selection):
    "The section names of the setup look like sources, so point to the extensions."
    try:
        yield
    except exception.IncorrectUsage as error:
        if selection is None:
            raise
        hint = (
            " The settings VASP wrote are grouped by section in the extensions, "
            f'e.g. read()["extensions"]["{selection}"].'
        )
        raise exception.IncorrectUsage(f"{error}{hint}") from error


@quantity("run_info")
class RunInfo:
    """Contains information about the VASP run.

    The run info gathers a summary of the calculation in one place: the VASP version
    and the system, properties of the results such as the Fermi energy, the number
    of ionic steps, and the phonon calculation.
    If VASP records how the run was set up, it adds the core INCAR settings, e.g.,
    the spin (ISPIN and LNONCOLLINEAR), the FFT grids, the parallelization, the run
    time and peak memory, the build of the executable, whether VASP terminated
    normally, the error messages, and, as extensions, every setting and metadata
    entry VASP wrote to the HDF5 file. Use :py:meth:`read` to get all of it as a dictionary and
    :py:meth:`print` for a short overview.
    """

    def __init__(self, source, quantity_name: str = "run_info"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_run_info: raw.RunInfo) -> "RunInfo":
        """Create a RunInfo dispatcher from raw data (convenience for testing)."""
        return cls(source=DataSource(raw_run_info))

    def read(self, selection: str | None = None) -> dict:
        """Convert the run information to a dictionary.

        The dictionary summarizes the calculation: the VASP version
        ``"vasp_version"``, the SYSTEM tag ``"system_tag"``, the Fermi energy
        ``"fermi_energy"`` in eV VASP computed, the number of ionic steps
        ``"num_ionic_steps"``, whether the ions move with selective dynamics
        ``"has_selective_dynamics"``, and the phonon calculation: how the force
        constants were computed ``"phonon_method"``, the number of modes at the Gamma
        point of the supercell ``"phonon_num_modes"``, and the number of q points
        ``"phonon_num_qpoints"`` and frequencies per q point ``"phonon_num_branches"``
        of the dispersion. If VASP records how the run was set up, it also contains

        - the core settings of the run under the name of their INCAR tag: the kind
          of run (``"IBRION"``, ``"ISIF"``, ``"NSW"``, ``"EDIFFG"``), the basis
          and accuracy (``"ENCUT"`` in eV, ``"PREC"``, ``"EDIFF"`` in eV,
          ``"ALGO"``, ``"NBANDS"``, ``"nkpts"``, ``"NELECT"``, ``"LASPH"``), the
          smearing and Fermi level (``"ISMEAR"``, ``"SIGMA"`` in eV,
          ``"EFERMI"``), the Hamiltonian (``"XC"``, ``"LHFCALC"``, ``"AEXX"``,
          ``"HFSCREEN"``, ``"IVDW"``, ``"NUPDOWN"``, ``"LDIPOL"``, ``"IDIPOL"``,
          ``"LEPSILON"``), and further ones like ``"ISPIN"`` or ``"NELM"``.
          ``"POTIM"`` is in fs, ``"TEBEG"`` and ``"TEEND"`` are in K, and
          ``"EDIFFG"`` is in eV if positive (energy) or eV/Å if negative (forces).
          Logical tags are Python bools; string tags keep VASP's spelling, e.g.,
          ``"LREAL"`` is "Auto", "On", "T", or "F" and ``"XC"`` is a lower-case
          name such as "pbe", "hse06", or "mixture". ``"ALGO"`` is the value as
          typed in the INCAR file, e.g. "Fast", "VERYFAST", or "V", so compare it
          ignoring case; in response-function runs it is that algorithm, e.g.
          "EVGW0". It is "Normal" if the INCAR sets neither ALGO nor IALGO and None
          if it sets only IALGO, which ``read()["extensions"]["electronic"]`` holds.
          ``"EFERMI"`` is the value set
          in the INCAR file, not the computed ``"fermi_energy"``: "LEGACY", "MIDGAP", or
          the Fermi energy in eV the user fixed, given as text, e.g. "5.5",
        - ``"LDAU"``, which is True if DFT+U is used; the U and J values are then
          in ``read()["extensions"]["ldau"]``,
        - the FFT grids ``"grid_coarse_shape"`` and ``"grid_fine_shape"``,
        - the resources ``"mpi_ranks"``, ``"omp_threads"``, ``"nodes"``,
          ``"gpus"``, ``"KPAR"``, and ``"NCORE"`` and the performance
          ``"wall_time"`` and ``"cpu_time"`` in s and ``"peak_memory"`` in kB,
        - the build of the executable: ``"build_variant"`` ("gam", "std", or
          "ncl") and whether it was built with ``"build_gpu"`` or
          ``"build_openmp"`` support,
        - ``"exit_status"``, how VASP finished: "success" if it terminated
          normally, or "error", "killed", or "dry-run"; a run can succeed while
          error messages exist,
        - ``"error_messages"`` and ``"error_message_counts"``, the text of every
          distinct message of severity "error" or "bug" and how often VASP
          reported it. Warnings and alerts are not reported here; they remain in
          the HDF5 file under /messages. Both are None if the run was killed,
          because VASP then writes no messages.

        A key that is exactly an INCAR tag is spelled like the tag in upper case,
        e.g. ``"SIGMA"``; every other key is lower case, e.g. ``"nkpts"``, which VASP
        takes from the KPOINTS file. A setting VASP did not write is None; in
        particular, for files of VASP versions that do not record the setup, the core
        settings, ``"exit_status"``, the error messages, the resources, and the build
        are None. Serial builds write no resources.

        Parameters
        ----------
        selection : str | None
            Select the source of the data. This quantity has only the default source;
            the argument exists for compatibility with older versions of py4vasp.

        Returns
        -------
        -
            A dictionary with the run information. The key ``"extensions"`` contains
            every entry VASP wrote as ``{section: {key: value}}``, including sections
            py4vasp does not know about. The sections of the /setup group keep their
            name (e.g. ``"dos"``, ``"electronic"``) and their keys keep VASP's
            spelling, i.e., upper-case INCAR tags. The sections ``"metadata/build"``,
            ``"metadata/resources"``, and ``"metadata/performance"`` describe the
            executable, the resources, and the performance of the run with lower-case
            keys. Nested groups are flattened into keys like ``"functional_1/NAME"``.
            The extensions are empty for files of VASP versions that do not record
            the setup.

        Examples
        --------
        Read the run information of a calculation and look at some core settings

        >>> from py4vasp import demo
        >>> calculation = demo.calculation()
        >>> run_info = calculation.run_info.read()
        >>> run_info["grid_coarse_shape"], run_info["SIGMA"], run_info["exit_status"]
        ([32, 32, 40], 0.05, 'success')
        >>> run_info["mpi_ranks"], run_info["KPAR"], run_info["NCORE"]
        (64, 4, 8)

        The kind of run, the basis, and the functional tell how the results came about

        >>> run_info["IBRION"], run_info["NSW"], run_info["ENCUT"], run_info["PREC"]
        (2, 40, 520.0, 'accurate')
        >>> run_info["XC"], run_info["LHFCALC"], run_info["LDAU"]
        ('pbe', False, False)
        >>> run_info["wall_time"], run_info["build_variant"]
        (1834.5, 'std')

        Only messages of severity "error" or "bug" are reported, so the warning of
        this run is missing; the run still succeeded

        >>> run_info["error_messages"], run_info["error_message_counts"]
        (['The basis set is incomplete.'], [1])

        Every entry VASP wrote is available in the extensions, grouped by section

        >>> sorted(run_info["extensions"])
        ['dimensions', 'dipole', 'dos', 'electronic', 'exchange_correlation',
         'ionic', 'kpoints_opt', 'ldau', 'linear_response', 'metadata/build',
         'metadata/performance', 'metadata/resources', 'phonons', 'vdw',
         'write_flags']
        >>> run_info["extensions"]["dos"]
        {'EFERMI_METHOD': 'LEGACY', 'ISMEAR': 0, 'SIGMA': 0.05}
        >>> run_info["extensions"]["exchange_correlation"]["functional_1/NAME"]
        'PBE'
        >>> run_info["extensions"]["metadata/build"]["variant"]
        'std'
        """
        with _hint_at_extensions(selection):
            return merge_default(
                self._source,
                self._quantity_name,
                selection,
                RunInfoHandler.from_data,
                RunInfoHandler.to_dict,
            )

    def to_dict(self, selection: str | None = None) -> dict:
        """Convenient alias for :py:meth:`read`."""
        return self.read(selection)

    def print(self, selection: str | None = None) -> None:
        """Print a string representation of this quantity.

        Parameters
        ----------
        selection : str | None
            Select which source of the quantity is printed. If you select multiple
            sources, py4vasp prints one block per source.
        """
        print(self.__str__(selection))

    def selections(self) -> dict:
        """Returns possible alternatives for this particular quantity VASP can produce.

        The returned dictionary contains a single item with the name of the quantity
        mapping to all possible selections. Each of these selections may be passed to
        the other methods of this quantity to choose which output of VASP is used.
        Not every calculation contains every source; use :py:meth:`is_available` to
        check which ones this calculation contains.

        Returns
        -------
        dict
            The key indicates this quantity and the value lists the possible choices
            for the selection argument of its other methods.
        """
        from py4vasp._raw import definition as raw_module

        return {self._quantity_name: list(raw_module.selections(self._quantity_name))}

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def __str__(self, selection=None) -> str:
        with _hint_at_extensions(selection):
            return merge_strings(
                self._source,
                self._quantity_name,
                selection,
                RunInfoHandler.from_data,
                RunInfoHandler.__str__,
            )

    def _to_database(self) -> dict:
        """Return {quantity[_selection]: handler_result} for database storage."""
        return merge_to_database(
            self._source,
            self._quantity_name,
            RunInfoHandler.from_data,
            RunInfoHandler.to_database,
        )

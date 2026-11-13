# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from contextlib import suppress

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

# core settings of raw.RunSetup; a key that is exactly an INCAR tag is reported in
# upper case, every other one in lower case, and the raw field is the lower-case key
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
            **self._dict_from_contcar(),
            **self._dict_from_phonon_dispersion(),
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
        system = convert.text_to_string(data["system_tag"] or "")
        header = f"run info for {system}:" if system else "run info:"
        lines = [header, *self._summary_lines(data)]
        return "\n".join(lines)

    def _summary_lines(self, data):
        """Report the quantities VASP wrote, skipping the ones it did not."""
        version = data["vasp_version"]
        if version is not None:
            yield f"    VASP version: {convert.text_to_string(version)}"
        if data["exit_status"] is not None:
            yield f"    exit status: {data['exit_status']}"
        if data["num_ionic_steps"] is not None:
            yield f"    ionic steps: {data['num_ionic_steps']}"
        spin = self._spin_string(data)
        if spin is not None:
            yield f"    spin: {spin}"
        qpoints = data["phonon_num_qpoints"]
        modes = data["phonon_num_modes"]
        if qpoints is not None and modes is not None:
            yield f"    phonon dispersion: {qpoints} q-points, {modes} modes"
        yield from self._setup_summary_lines(data)

    def _setup_summary_lines(self, data):
        coarse, fine = data["grid_coarse_shape"], data["grid_fine_shape"]
        if coarse is not None:
            grid = " x ".join(map(str, coarse))
            if fine is not None:
                grid += f" (fine {' x '.join(map(str, fine))})"
            yield f"    FFT grid: {grid}"
        ionic_update = self._tags(data, "IBRION", "ISIF", "NSW")
        if ionic_update:
            yield f"    ionic update: {ionic_update}"
        if data["XC"] is not None:
            yield f"    functional: {data['XC']}" + ("+U" if data["LDAU"] else "")
        settings = (
            ("electronic", "ENCUT"),
            ("dos", "ISMEAR"),
            ("dos", "SIGMA"),
            ("dos", "EFERMI"),
            ("electronic", "NELM"),
        )
        for section, tag in settings:
            value = data[tag]
            if value is not None:
                unit = self._setup_unit(section, tag)
                yield f"    {tag}: {value}" + (f" {unit}" if unit else "")
        if data["wall_time"] is not None:
            yield f"    wall time: {data['wall_time']:.1f} s"
        if data["peak_memory"] is not None:
            yield f"    peak memory: {data['peak_memory'] / 1024:.1f} MiB"
        parallelization = self._parallelization(data)
        if parallelization:
            yield f"    parallelization: {parallelization}"
        if data["error_messages"] is not None:
            yield f"    error messages: {len(data['error_messages'])}"
        if data["extensions"]:
            yield f"    extensions: {', '.join(sorted(data['extensions']))}"

    @staticmethod
    def _parallelization(data):
        parts = []
        if data["mpi_ranks"] is not None:
            parts.append(f"{data['mpi_ranks']} MPI ranks")
        if tags := RunInfoHandler._tags(data, "KPAR", "NCORE"):
            parts.append(tags)
        return ", ".join(parts)

    @staticmethod
    def _tags(data, *tags):
        "Join the INCAR tags VASP wrote as TAG=value."
        return ", ".join(f"{tag}={data[tag]}" for tag in tags if data[tag] is not None)

    def _setup_unit(self, section, tag):
        extension = getattr(self._raw_run_info, "extension", None)
        if check.is_none(extension) or section not in extension.sections:
            return None
        return extension.sections[section].units.get(tag)

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

    def _read_attr(self, *keys: str):
        data = self._raw_run_info
        for key in keys:
            data = getattr(data, key, None)
            if data is None:
                return None
        try:
            is_none = data.is_none() or data is None
        except AttributeError:
            is_none = False
        return data[:] if not is_none else None

    def _dict_additional_collection(self) -> dict:
        fermi_energy = None
        with suppress(exception.NoData):
            fermi_energy = self._raw_run_info.fermi_energy
        if isinstance(fermi_energy, VaspData):
            fermi_energy = fermi_energy._data

        return {
            **self._dict_from_setup(),
            "EFERMI": self._efermi(),
            "fermi_energy": fermi_energy,
        }

    def _efermi(self):
        """The EFERMI tag as set in the INCAR file: MIDGAP, LEGACY, or a number.

        VASP writes the number only if the user set one. A database column holds a
        single type, so the number is reported as text, too."""
        value = self._setup_value("efermi")
        if value is not None:
            return str(value)
        return self._setup_string("efermi_method")

    def _dict_from_setup(self) -> dict:
        settings = {
            **{key: self._setup_value(key.lower()) for key in _CORE_SETTINGS},
            **{key: self._setup_bool(key.lower()) for key in _CORE_LOGICALS},
            **{key: self._setup_string(key.lower()) for key in _CORE_STRINGS},
        }
        return {
            "grid_coarse_shape": self._setup_grid("ngx", "ngy", "ngz"),
            "grid_fine_shape": self._setup_grid("ngxf", "ngyf", "ngzf"),
            "exit_status": self._setup_string("exit_status"),
            **settings,
            "LDAU": self._is_ldau(settings),
        }

    def _is_ldau(self, settings):
        """VASP writes the ldau group, and so LDAUTYPE, only if LDA+U is active.

        Without any recorded INCAR setting, the absence of the group says nothing."""
        if all(value is None for key, value in settings.items() if key.isupper()):
            return None
        return self._setup_value("ldautype") is not None

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
        setup = self._raw_run_info.setup
        arrays = (setup.message_text, setup.message_severity, setup.message_count)
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
        setup = getattr(self._raw_run_info, "setup", None)
        if check.is_none(setup):
            return None
        value = getattr(setup, key)
        if check.is_none(value):
            return None
        value = _unwrap(value)
        return value.item() if hasattr(value, "item") else value

    def _setup_grid(self, *keys):
        shape = [self._setup_value(key) for key in keys]
        return None if None in shape else shape

    def _dict_from_system(self) -> dict:
        system_tag = None
        with suppress(exception.NoData):
            system_tag = self._read_attr("system", "system")
        return {
            "system_tag": system_tag,
        }

    def _dict_from_runtime(self) -> dict:
        vasp_version = None
        with suppress(exception.NoData):
            runtime_data = self._raw_run_info.runtime
            vasp_version = None if runtime_data is None else runtime_data.vasp_version
        return {
            "vasp_version": vasp_version,
        }

    def _dict_from_structure(self) -> dict:
        num_ion_steps = None
        with suppress(exception.NoData, AttributeError):
            positions = self._read_attr("structure", "positions")
            if not check.is_none(positions):
                num_ion_steps = 1 if positions.ndim == 2 else positions.shape[0]
        return {
            "num_ionic_steps": num_ion_steps,
        }

    def _dict_from_contcar(self) -> dict:
        has_selective_dynamics = None
        has_lattice_velocities = None
        has_ion_velocities = None
        with suppress(exception.NoData):
            has_selective_dynamics = not check.is_none(
                self._read_attr("contcar", "selective_dynamics")
            )
            has_lattice_velocities = not check.is_none(
                self._read_attr("contcar", "lattice_velocities")
            )
            has_ion_velocities = not check.is_none(
                self._read_attr("contcar", "ion_velocities")
            )
        return {
            "has_selective_dynamics": has_selective_dynamics,
            "has_lattice_velocities": has_lattice_velocities,
            "has_ion_velocities": has_ion_velocities,
        }

    def _dict_from_phonon_dispersion(self) -> dict:
        phonon_num_qpoints = None
        phonon_num_modes = None
        # a calculation without phonons has no dispersion at all, so guard against the
        # missing attribute the same way _dict_from_structure does
        with suppress(exception.NoData, AttributeError):
            eigenvalues = self._raw_run_info.phonon_dispersion.eigenvalues
            phonon_num_qpoints = eigenvalues.shape[0]
            phonon_num_modes = eigenvalues.shape[1]
        return {
            "phonon_num_qpoints": phonon_num_qpoints,
            "phonon_num_modes": phonon_num_modes,
        }


def _unwrap(data):
    return data[()] if isinstance(data, VaspData) else data


@quantity("run_info")
class RunInfo:
    """Contains information about the VASP run.

    The run info gathers a summary of the calculation in one place: the VASP version
    and the system, properties of the results such as the Fermi energy, the number
    of ionic steps, and the phonon dispersion.
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

        The dictionary summarizes the calculation: the VASP version, the system, and
        properties derived from the results like the Fermi energy
        ``"fermi_energy"`` in eV or the number of ionic steps. If VASP records how
        the run was set up, it also contains

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
          name such as "pbe", "hse06", or "mixture". ``"EFERMI"`` is the value set
          in the INCAR file, not the computed ``"fermi_energy"``: "LEGACY", "MIDGAP", or
          the Fermi energy in eV the user fixed, given as text, e.g. "5.5",
        - ``"LDAU"``, which is True if LDA+U is active; the U and J values are in
          ``read()["extensions"]["ldau"]``, which exists only if LDA+U is active,
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
          the HDF5 file under /messages.

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
         'ionic', 'kpoints_opt', 'linear_response', 'metadata/build',
         'metadata/performance', 'metadata/resources', 'vdw', 'write_flags']
        >>> run_info["extensions"]["dos"]
        {'EFERMI_METHOD': 'LEGACY', 'ISMEAR': 0, 'SIGMA': 0.05}
        >>> run_info["extensions"]["exchange_correlation"]["functional_1/NAME"]
        'PBE'
        >>> run_info["extensions"]["metadata/build"]["variant"]
        'std'
        """
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

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from contextlib import suppress

from py4vasp import raw
from py4vasp._calculation import bandgap as bandgap_module
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

_TO_DATABASE_SUPPRESSED_EXCEPTIONS = (
    exception.Py4VaspError,
    exception.OutdatedVaspVersion,
    exception.NoData,
    AttributeError,
    TypeError,
    ValueError,
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
        if data["num_ionic_steps"] is not None:
            yield f"    ionic steps: {data['num_ionic_steps']}"
        if data["fermi_energy"] is not None:
            yield f"    Fermi energy: {data['fermi_energy']:.3f}"
        spin = self._spin_string(data)
        if spin is not None:
            yield f"    spin: {spin}"
        if data["is_metallic"] is not None:
            yield f"    metallic: {'yes' if data['is_metallic'] else 'no'}"
        qpoints = data["phonon_num_qpoints"]
        modes = data["phonon_num_modes"]
        if qpoints is not None and modes is not None:
            yield f"    phonon dispersion: {qpoints} q-points, {modes} modes"

    @staticmethod
    def _spin_string(data):
        if data["is_collinear"]:
            return "collinear"
        if data["is_noncollinear"]:
            return "noncollinear"
        # the two flags fall back to different fields, so one may be known while the
        # other is not; a single False does not establish a nonpolarized calculation
        if data["is_collinear"] is None or data["is_noncollinear"] is None:
            return None
        return "nonpolarized"

    def to_database(self) -> dict:
        """Serialize run info for the database."""
        return RunInfoModel(**self.to_dict())

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

        is_success = None  # TODO implement

        is_collinear = self._is_collinear()
        is_noncollinear = self._is_noncollinear()
        is_metallic = self._is_metallic()
        is_magnetic = None  # TODO implement
        magnetic_order = None  # TODO implement

        grid_coarse_shape = (
            None  # TODO implement for FFT grid (currently not written to H5)
        )
        grid_fine_shape = (
            None  # TODO implement for FFT grid (currently not written to H5)
        )

        return {
            "grid_coarse_shape": grid_coarse_shape,
            "grid_fine_shape": grid_fine_shape,
            "is_success": is_success,
            "fermi_energy": fermi_energy,
            "is_collinear": is_collinear,
            "is_noncollinear": is_noncollinear,
            "is_metallic": is_metallic,
            "is_magnetic": is_magnetic,
            "magnetization_order": magnetic_order,
        }

    def _is_collinear(self):
        if not check.is_none(self._raw_run_info.len_dos):
            return self._raw_run_info.len_dos == 2
        else:
            if not check.is_none(self._raw_run_info.band_dispersion_eigenvalues):
                return len(self._raw_run_info.band_dispersion_eigenvalues) == 2
            else:
                return None

    def _is_noncollinear(self):
        if not check.is_none(self._raw_run_info.len_dos):
            return self._raw_run_info.len_dos == 4
        else:
            if not check.is_none(self._raw_run_info.band_projections):
                return len(self._raw_run_info.band_projections) == 4
            else:
                return None

    def _is_metallic(self):
        with suppress(*_TO_DATABASE_SUPPRESSED_EXCEPTIONS):
            if check.is_none(self._raw_run_info.bandgap):
                return None
            gap = bandgap_module.BandgapHandler.from_data(self._raw_run_info.bandgap)
            return all(gap._output_gap("fundamental", to_string=False) <= 0.0)
        return None

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


@quantity("run_info")
class RunInfo:
    "Contains information about the VASP run."

    def __init__(self, source, quantity_name: str = "run_info"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_run_info: raw.RunInfo) -> "RunInfo":
        """Create a RunInfo dispatcher from raw data (convenience for testing)."""
        return cls(source=DataSource(raw_run_info))

    def read(self) -> dict:
        "Convert the run information to a dictionary."
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            RunInfoHandler.from_data,
            RunInfoHandler.to_dict,
        )

    def to_dict(self, selection: str | None = None) -> dict:
        """Convenient alias for :py:meth:`read`."""
        return self.read()

    def print(self, selection: str | None = None) -> None:
        """Print a string representation of this quantity.

        Parameters
        ----------
        selection : str | None
            Select which source of the quantity is printed. If you select multiple
            sources, py4vasp prints one block per source.
        """
        print(self.__str__(selection))

    def selections(self):
        from py4vasp._raw import definition as raw_module

        return {self._quantity_name: list(raw_module.selections(self._quantity_name))}

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else "...")

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

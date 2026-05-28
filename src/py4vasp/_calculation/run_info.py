# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pathlib
from contextlib import suppress

from py4vasp import raw
from py4vasp._calculation import bandgap as bandgap_module, exception
from py4vasp._calculation.dispatch import (
    DataSource,
    FileSource,
    merge_default,
    quantity,
)
from py4vasp._raw.data_db import RunInfo_DB
from py4vasp._raw.data_wrapper import VaspData
from py4vasp._util import check

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

    def to_database(self) -> dict:
        """Serialize run info for the database."""
        return {
            "run_info": RunInfo_DB(**self.to_dict()),
        }

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
        with suppress(exception.NoData):
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

    @classmethod
    def from_path(cls, path=".") -> "RunInfo":
        """Create a RunInfo dispatcher that reads from HDF5 files at *path*."""
        return cls(source=FileSource(path))

    @classmethod
    def from_file(cls, file_name) -> "RunInfo":
        """Create a RunInfo dispatcher that reads from a specific HDF5 file."""
        resolved = pathlib.Path(file_name).expanduser().resolve()
        return cls(source=FileSource(resolved.parent, file=file_name))

    def read(self, selection: str | None = None) -> dict:
        "Convert the run information to a dictionary."
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            RunInfoHandler.from_data,
            RunInfoHandler.to_dict,
        )

    def to_dict(self, selection: str | None = None) -> dict:
        """Convenient alias for :py:meth:`read`."""
        return self.read(selection=selection)

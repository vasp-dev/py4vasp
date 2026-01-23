# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._calculation import base, exception
from py4vasp._calculation._dispersion import Dispersion


class RunInfo(base.Refinery):
    "Contains information about the VASP run."

    @base.data_access
    def to_dict(self):
        "Convert the run information to a dictionary."

        return {
            **self._dict_from_runtime(),
            **self._dict_from_system(),
            **self._dict_from_structure(),
            **self._dict_from_contcar(),
            **self._dict_from_phonon_dispersion(),
            # TODO add more run info
        }

    def _read(self, *keys: str) -> dict:
        data = self._raw_data
        for key in keys:
            data = getattr(data, key, None)
            if data is None:
                return None
        try:
            is_none = data.is_none() or data is None
        except AttributeError:
            is_none = False
        return data[:] if not is_none else None

    def _dict_from_system(self) -> dict:
        system_tag = None

        try:
            system_tag = self._read("system", "system")
        except exception.NoData:
            pass

        return {
            "system_tag": system_tag,
        }

    def _dict_from_runtime(self) -> dict:
        vasp_version = None

        try:
            runtime_data = self._raw_data.runtime
            vasp_version = None if runtime_data is None else runtime_data.vasp_version
        except exception.NoData:
            pass

        return {
            "vasp_version": vasp_version,
        }

    def _dict_from_structure(self) -> dict:
        num_ion_steps = None

        try:
            positions = self._read("structure", "positions")
            if positions is not None:
                num_ion_steps = 1 if positions.ndim == 2 else positions.shape[0]
        except exception.NoData:
            pass

        return {
            "num_ion_steps": num_ion_steps,
        }

    def _dict_from_contcar(self) -> dict:
        has_selective_dynamics = None
        has_lattice_velocities = None
        has_ion_velocities = None

        try:
            has_selective_dynamics = (
                self._read("contcar", "selective_dynamics") is not None
            )
            has_lattice_velocities = (
                self._read("contcar", "lattice_velocities") is not None
            )
            has_ion_velocities = self._read("contcar", "ion_velocities") is not None
        except exception.NoData:
            pass

        return {
            "has_selective_dynamics": has_selective_dynamics,
            "has_lattice_velocities": has_lattice_velocities,
            "has_ion_velocities": has_ion_velocities,
        }

    def _dict_from_phonon_dispersion(self) -> dict:
        phonon_num_qpoints = None
        phonon_num_modes = None

        try:
            eigenvalues = self._raw_data.phonon_dispersion.eigenvalues
            phonon_num_qpoints = eigenvalues.shape[0]
            phonon_num_modes = eigenvalues.shape[1]
        except exception.NoData:
            pass

        return {
            "phonon_num_qpoints": phonon_num_qpoints,
            "phonon_num_modes": phonon_num_modes,
        }

    @base.data_access
    def _to_database(self, *args, **kwargs):
        return {
            "run_info": self.to_dict(),
        }

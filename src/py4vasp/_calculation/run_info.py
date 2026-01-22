# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._calculation import base


class RunInfo(base.Refinery):
    "Contains information about the VASP run."

    @base.data_access
    def to_dict(self):
        "Convert the run information to a dictionary."

        # Structure
        positions = self._read("structure", "positions")
        number_steps = (
            None
            if positions is None
            else (1 if positions.ndim == 2 else positions.shape[0])
        )

        return {
            **self._dict_from_runtime(),
            **self._dict_from_structure(),
            **self._dict_from_contcar(),
            # TODO add more run info
        }

    def _read(self, *keys: str) -> dict:
        data = self._raw_data
        for key in keys:
            data = getattr(data, key, None)
            if data is None:
                return None
        return data[:] if not data.is_none() else None

    def _dict_from_runtime(self) -> dict:
        runtime_data = self._raw_data.runtime
        vasp_version = None if runtime_data is None else runtime_data.vasp_version

        return {
            "vasp_version": vasp_version,
        }

    def _dict_from_structure(self) -> dict:
        positions = self._read("structure", "positions")
        number_steps = (
            None
            if positions is None
            else (1 if positions.ndim == 2 else positions.shape[0])
        )

        return {
            "num_ion_steps": number_steps,
        }

    def _dict_from_contcar(self) -> dict:
        contcar = getattr(self._raw_data, "contcar", None)
        selective_dynamics = self._read("contcar", "selective_dynamics")
        lattice_velocities = self._read("contcar", "lattice_velocities")
        ion_velocities = self._read("contcar", "ion_velocities")
        return {
            "has_selective_dynamics": (
                None if contcar is None else selective_dynamics is not None
            ),
            "has_lattice_velocities": (
                None if contcar is None else lattice_velocities is not None
            ),
            "has_ion_velocities": (
                None if contcar is None else ion_velocities is not None
            ),
        }

    @base.data_access
    def _to_database(self, *args, **kwargs):
        return {
            "run_info": self.to_dict(),
        }

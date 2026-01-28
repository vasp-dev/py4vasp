# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._calculation import bandgap, base, exception
from py4vasp._calculation._dispersion import Dispersion
from py4vasp._util import check


class RunInfo(base.Refinery):
    "Contains information about the VASP run."

    @base.data_access
    def to_dict(self):
        "Convert the run information to a dictionary."

        return {
            **self._dict_from_runtime(),
            **self._dict_from_system(),
            **self._dict_from_structure(),
            **self._dict_additional_collection(),
            **self._dict_from_contcar(),
            **self._dict_from_phonon_dispersion(),
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

    def _dict_additional_collection(self) -> dict:
        fermi_energy = None
        try:
            fermi_energy = self._raw_data.fermi_energy
        except exception.NoData:
            pass

        is_success = None  # TODO implement

        is_collinear = self._is_collinear()
        is_noncollinear = self._is_noncollinear()
        is_metallic = self._is_metallic()
        is_magnetic = None  # TODO implement
        magnetic_order = None  # TODO implement, maybe as magnetic_space_group (ferromagnetic, antiferromagnetic, ...) via symmetry

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
            "magnetization_total": is_magnetic,
            "magnetization_order": magnetic_order,
        }

    def _is_collinear(self):
        try:
            return self._raw_data.len_dos == 2
        except exception.NoData:
            try:
                return len(self._raw_data.band_dispersion_eigenvalues) == 2
            except exception.NoData:
                return None

    def _is_noncollinear(self):
        try:
            return self._raw_data.len_dos == 4
        except exception.NoData:
            try:
                if check.is_none(self._raw_data.band_projections):
                    return None
                return len(self._raw_data.band_projections) == 4
            except exception.NoData:
                return None

    def _is_metallic(self):
        try:
            if check.is_none(self._raw_data.bandgap):
                return None
            gap = bandgap.Bandgap.from_data(self._raw_data.bandgap)
            return gap._output_gap("fundamental", to_string=False) <= 0.0
        except (exception.OutdatedVaspVersion, exception.NoData):
            return None
        except:
            return None

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
            if not check.is_none(positions):
                num_ion_steps = 1 if positions.ndim == 2 else positions.shape[0]
        except (exception.NoData, AttributeError):
            pass

        return {
            "num_ion_steps": num_ion_steps,
        }

    def _dict_from_contcar(self) -> dict:
        has_selective_dynamics = None
        has_lattice_velocities = None
        has_ion_velocities = None

        try:
            has_selective_dynamics = not check.is_none(
                self._read("contcar", "selective_dynamics")
            )
            has_lattice_velocities = not check.is_none(
                self._read("contcar", "lattice_velocities")
            )
            has_ion_velocities = not check.is_none(
                self._read("contcar", "ion_velocities")
            )
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

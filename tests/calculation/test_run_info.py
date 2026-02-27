# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp._calculation._CONTCAR import CONTCAR
from py4vasp._calculation._dispersion import Dispersion
from py4vasp._calculation.bandgap import Bandgap
from py4vasp._calculation.run_info import RunInfo
from py4vasp._calculation.structure import Structure
from py4vasp._util import check


@pytest.fixture(params=["Sr2TiO4"])
def run_info(request, raw_data):
    raw_run_info = raw_data.run_info(request.param)
    run_info = RunInfo.from_data(raw_run_info)
    run_info.ref = types.SimpleNamespace()
    run_info.ref.system_name = raw_run_info.system.system
    run_info.ref.runtime = raw_run_info.runtime
    run_info.ref.fermi_energy = raw_run_info.fermi_energy
    run_info.ref.bandgap = Bandgap.from_data(raw_run_info.bandgap)
    run_info.ref.len_dos = raw_run_info.len_dos
    run_info.ref.band_dispersion_eigenvalues = raw_run_info.band_dispersion_eigenvalues
    run_info.ref.band_projections = raw_run_info.band_projections
    run_info.ref.structure = Structure.from_data(raw_run_info.structure)
    run_info.ref.contcar = CONTCAR.from_data(raw_run_info.contcar)
    run_info.ref.phonon_dispersion = Dispersion.from_data(
        raw_run_info.phonon_dispersion
    )
    return run_info


def _check_dict(data_dict, runinfo_ref):
    # from runtime
    assert data_dict["vasp_version"] == runinfo_ref.runtime.vasp_version

    # from structure
    assert (
        data_dict["num_ion_steps"]
        == runinfo_ref.structure._raw_data.positions[:].shape[0]
    )

    # from system
    assert data_dict["system_tag"] == runinfo_ref.system_name

    # from contcar
    assert data_dict["has_selective_dynamics"] == (
        not check.is_none(runinfo_ref.contcar._raw_data.selective_dynamics)
    )
    assert data_dict["has_ion_velocities"] == (
        not check.is_none(runinfo_ref.contcar._raw_data.ion_velocities)
    )
    assert data_dict["has_lattice_velocities"] == (
        not check.is_none(runinfo_ref.contcar._raw_data.lattice_velocities)
    )

    # from phonon dispersion
    assert (
        data_dict["phonon_num_qpoints"]
        == runinfo_ref.phonon_dispersion._raw_data.eigenvalues[:].shape[0]
    )
    assert (
        data_dict["phonon_num_modes"]
        == runinfo_ref.phonon_dispersion._raw_data.eigenvalues[:].shape[1]
    )

    # extra collection
    assert data_dict["grid_coarse_shape"] is None
    assert data_dict["grid_fine_shape"] is None
    assert data_dict["is_success"] is None
    assert data_dict["fermi_energy"] == runinfo_ref.fermi_energy
    assert data_dict["is_collinear"] == (runinfo_ref.len_dos == 2)
    assert data_dict["is_noncollinear"] == (runinfo_ref.len_dos == 4)
    assert data_dict["is_metallic"] == all(
        runinfo_ref.bandgap._output_gap("fundamental", to_string=False) <= 0.0
    )
    assert data_dict["magnetization_total"] is None
    assert data_dict["magnetization_order"] is None


def test_read(run_info):
    _check_dict(run_info.read(), run_info.ref)


def test_to_database(run_info):
    _check_dict(run_info._read_to_database()["run_info:default"], run_info.ref)


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.run_info("Sr2TiO4")
    check_factory_methods(RunInfo, data)

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import numpy as np
import pytest

from py4vasp import raw
from py4vasp._calculation._CONTCAR import CONTCARHandler
from py4vasp._calculation._dispersion import DispersionHandler
from py4vasp._calculation.bandgap import Bandgap, BandgapHandler
from py4vasp._calculation.run_info import RunInfo, RunInfoHandler
from py4vasp._calculation.structure import StructureHandler
from py4vasp._raw.models import RunInfoModel
from py4vasp._util import check


@pytest.fixture(params=["Sr2TiO4"])
def run_info(request, raw_data):
    raw_run_info = raw_data.run_info(request.param)
    run_info = RunInfo.from_data(raw_run_info)
    run_info.ref = types.SimpleNamespace()
    run_info.ref.system_name = raw_run_info.system.system
    run_info.ref.runtime = raw_run_info.runtime
    run_info.ref.fermi_energy = raw_run_info.fermi_energy
    run_info.ref.setup = raw_run_info.setup
    run_info.ref.extension = raw_run_info.extension
    run_info.ref.bandgap = BandgapHandler.from_data(raw_run_info.bandgap)
    run_info.ref.len_dos = raw_run_info.len_dos
    run_info.ref.band_dispersion_eigenvalues = raw_run_info.band_dispersion_eigenvalues
    run_info.ref.band_projections = raw_run_info.band_projections
    run_info.ref.structure = StructureHandler.from_data(raw_run_info.structure)
    run_info.ref.contcar = CONTCARHandler.from_data(raw_run_info.contcar)
    run_info.ref.phonon_dispersion = DispersionHandler.from_data(
        raw_run_info.phonon_dispersion
    )
    return run_info


@pytest.fixture(params=["Sr2TiO4"])
def run_info_handler(request, raw_data):
    raw_run_info = raw_data.run_info(request.param)
    handler = RunInfoHandler.from_data(raw_run_info)
    handler.ref = types.SimpleNamespace()
    handler.ref.system_name = raw_run_info.system.system
    handler.ref.runtime = raw_run_info.runtime
    handler.ref.fermi_energy = raw_run_info.fermi_energy
    handler.ref.setup = raw_run_info.setup
    handler.ref.extension = raw_run_info.extension
    handler.ref.bandgap = BandgapHandler.from_data(raw_run_info.bandgap)
    handler.ref.len_dos = raw_run_info.len_dos
    handler.ref.band_dispersion_eigenvalues = raw_run_info.band_dispersion_eigenvalues
    handler.ref.band_projections = raw_run_info.band_projections
    handler.ref.structure = StructureHandler.from_data(raw_run_info.structure)
    handler.ref.contcar = CONTCARHandler.from_data(raw_run_info.contcar)
    handler.ref.phonon_dispersion = DispersionHandler.from_data(
        raw_run_info.phonon_dispersion
    )
    return handler


def _check_dict(models: RunInfoModel, runinfo_ref):
    # from runtime
    assert models.vasp_version == runinfo_ref.runtime.vasp_version

    # from structure
    assert (
        models.num_ionic_steps
        == runinfo_ref.structure._raw_structure.positions[:].shape[0]
    )

    # from system
    assert models.system_tag == runinfo_ref.system_name

    # from contcar
    assert models.has_selective_dynamics == (
        not check.is_none(runinfo_ref.contcar._raw_contcar.selective_dynamics)
    )
    assert models.has_ion_velocities == (
        not check.is_none(runinfo_ref.contcar._raw_contcar.ion_velocities)
    )
    assert models.has_lattice_velocities == (
        not check.is_none(runinfo_ref.contcar._raw_contcar.lattice_velocities)
    )

    # from phonon dispersion
    assert (
        models.phonon_num_qpoints
        == runinfo_ref.phonon_dispersion._raw_dispersion.eigenvalues[:].shape[0]
    )
    assert (
        models.phonon_num_modes
        == runinfo_ref.phonon_dispersion._raw_dispersion.eigenvalues[:].shape[1]
    )

    # extra collection
    setup = runinfo_ref.setup
    assert list(models.grid_coarse_shape) == [setup.ngx, setup.ngy, setup.ngz]
    assert list(models.grid_fine_shape) == [setup.ngxf, setup.ngyf, setup.ngzf]
    assert models.is_success == (setup.exit_status == "success")
    assert models.fermi_energy == runinfo_ref.fermi_energy
    assert models.is_collinear == (setup.ispin == 2)
    assert models.is_noncollinear == bool(setup.lnoncollinear)
    assert models.is_metallic == all(
        runinfo_ref.bandgap._output_gap("fundamental", to_string=False) <= 0.0
    )
    assert models.is_magnetic is None
    assert models.magnetization_order is None

    # core settings
    for field, expected in _expected_core_settings(setup).items():
        assert getattr(models, field) == expected, field


CORE_BOOLS = (
    "lnoncollinear",
    "lsorbit",
    "lkpoints_opt",
    "lasph",
    "lhfcalc",
    "ldipol",
    "lepsilon",
    "build_gpu",
    "build_mpi",
    "build_openmp",
)
CORE_STRINGS = (
    "lreal",
    "prec",
    "algo",
    "xc",
    "build_variant",
)
CORE_SCALARS = (
    "sigma",
    "ispin",
    "istart",
    "icharg",
    "nelm",
    "lmaxmix",
    "potim",
    "tebeg",
    "teend",
    "mdalgo",
    "isym",
    "lorbit",
    "images",
    "peak_memory",
    "ncore",
    "kpar",
    "mpi_ranks",
    "ibrion",
    "isif",
    "nsw",
    "ediffg",
    "encut",
    "ediff",
    "nbands",
    "nkpts",
    "nelect",
    "ismear",
    "aexx",
    "hfscreen",
    "ivdw",
    "nupdown",
    "idipol",
    "wall_time",
    "cpu_time",
    "omp_threads",
    "nodes",
    "gpus",
)


def _expected_core_settings(setup):
    return {
        **{field: getattr(setup, field) for field in CORE_SCALARS},
        **{field: bool(getattr(setup, field)) for field in CORE_BOOLS},
        **{field: _decode(getattr(setup, field)) for field in CORE_STRINGS},
    }


def _decode(text):
    return text.decode().strip() if isinstance(text, bytes) else text.strip()


def test_read(run_info):
    actual = run_info.read()
    expected = _expected_core_settings(run_info.ref.setup)
    assert {key: actual[key] for key in expected} == expected
    actual.pop("extensions")  # checked separately, not part of the database model
    _check_dict(RunInfoModel(**actual), run_info.ref)


def test_core_bools_are_bool(run_info):
    actual = run_info.read()
    for field in CORE_BOOLS:
        assert type(actual[field]) is bool, field


def test_lreal_of_demo_is_vasp_string(run_info):
    assert run_info.read()["lreal"] == "Auto"


@pytest.mark.parametrize("lreal", [b"Auto", b"On", b"F", b"T", b"Auto  ", "On"])
def test_lreal_keeps_vasp_spelling(raw_data, lreal):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup.lreal = lreal
    actual = RunInfo.from_data(raw_run_info).read()["lreal"]
    assert actual == _decode(lreal)
    assert (
        RunInfo.from_data(raw_run_info)._to_database()["run_info"]["default"].lreal
        == actual
    )


def test_core_settings_without_setup_are_none(raw_data):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup = None
    actual = RunInfo.from_data(raw_run_info).read()
    fields = CORE_BOOLS + CORE_STRINGS + CORE_SCALARS
    assert all(actual[field] is None for field in fields)
    assert actual["error_messages"] is None
    assert actual["error_message_counts"] is None


def test_algo_missing_in_file_is_none(raw_data):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup.algo = raw.VaspData(None)
    actual = RunInfo.from_data(raw_run_info).read()
    assert actual["algo"] is None
    assert actual["prec"] == "accurate"


def test_ldau_is_false_without_ldau_group(run_info):
    # VASP writes the ldau group only if LDA+U is active
    assert run_info.read()["ldau"] is False
    assert run_info._to_database()["run_info"]["default"].ldau is False


@pytest.mark.parametrize("ldautype", [1, 2, 4])
def test_ldau_is_true_if_vasp_wrote_ldautype(raw_data, ldautype):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup.ldautype = ldautype
    assert RunInfo.from_data(raw_run_info).read()["ldau"] is True


def test_ldau_without_setup_is_none(raw_data):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup = None
    assert RunInfo.from_data(raw_run_info).read()["ldau"] is None


def test_error_messages_with_counts(raw_data):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup.message_text = np.array(
        [b"first error", b"a warning", b"second error", b"a bug"]
    )
    raw_run_info.setup.message_severity = np.array(
        [b"error", b"warning", b"error", b"bug"]
    )
    raw_run_info.setup.message_count = np.array([2, 5, 1, 1])
    handler = RunInfoHandler.from_data(raw_run_info)
    actual = handler.to_dict()
    assert actual["error_messages"] == ["first error", "second error"]
    assert actual["error_message_counts"] == [2, 1]
    assert all(type(count) is int for count in actual["error_message_counts"])
    model = handler.to_database()
    assert model.error_messages == ["first error", "second error"]
    assert model.error_message_counts == [2, 1]


@pytest.mark.parametrize("wrap", [lambda x: x, np.array, raw.VaspData])
def test_single_message_as_scalars(raw_data, wrap):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup.message_text = wrap(b"the only error")
    raw_run_info.setup.message_severity = wrap(b"error")
    raw_run_info.setup.message_count = wrap(np.int64(3))
    actual = RunInfo.from_data(raw_run_info).read()
    assert actual["error_messages"] == ["the only error"]
    assert actual["error_message_counts"] == [3]


@pytest.mark.parametrize("missing", ["message_severity", "message_count"])
def test_partial_message_arrays_are_none(raw_data, missing):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    setattr(raw_run_info.setup, missing, raw.VaspData(None))
    handler = RunInfoHandler.from_data(raw_run_info)
    actual = handler.to_dict()
    assert actual["error_messages"] is None
    assert actual["error_message_counts"] is None
    handler.to_database()  # the rest of the record is still stored


def test_error_messages_with_padded_severity(raw_data):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup.message_text = np.array([b"padded error    ", b"a warning  "])
    raw_run_info.setup.message_severity = np.array([b"error  ", b"warning"])
    raw_run_info.setup.message_count = np.array([4, 1])
    actual = RunInfo.from_data(raw_run_info).read()
    assert actual["error_messages"] == ["padded error"]
    assert actual["error_message_counts"] == [4]


def test_read_contains_extensions(run_info):
    raw_extension = run_info.ref.extension
    expected = {
        name: section.values for name, section in raw_extension.sections.items()
    }
    assert expected  # the demo data should contain some sections
    assert run_info.read()["extensions"] == expected


def test_unknown_section_appears(raw_data):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.extension.sections["brand_new"] = raw.SetupSection(
        values={"LFOO": True, "functional_1/NAME": "PBE"}, units={"LFOO": "none"}
    )
    actual = RunInfo.from_data(raw_run_info).read()["extensions"]
    assert actual["brand_new"] == {"LFOO": True, "functional_1/NAME": "PBE"}


@pytest.mark.parametrize("missing_extension", [None, raw.VaspData(None)])
def test_read_without_extension_is_empty(raw_data, missing_extension):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.extension = missing_extension
    assert RunInfo.from_data(raw_run_info).read()["extensions"] == {}


def test_to_database_excludes_extensions(run_info_handler):
    model = run_info_handler.to_database()
    assert not hasattr(model, "extensions")
    assert "extensions" in run_info_handler.to_dict()


def test_no_messages_is_empty_list(raw_data):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup.message_text = raw.VaspData(None)
    raw_run_info.setup.message_severity = raw.VaspData(None)
    raw_run_info.setup.message_count = raw.VaspData(None)
    actual = RunInfo.from_data(raw_run_info).read()
    assert actual["error_messages"] == []
    assert actual["error_message_counts"] == []


def test_to_dict_matches_read(run_info_handler):
    assert run_info_handler.to_dict() == run_info_handler.to_dict()


def test_dispatcher_to_dict_matches_read(run_info):
    assert run_info.to_dict() == run_info.read()


def test_to_database(run_info_handler):
    _check_dict(run_info_handler.to_database(), run_info_handler.ref)


def test_dispatcher_to_database(run_info):
    """Dispatcher._to_database() returns {quantity: {selection: handler_result}}."""
    result = run_info._to_database()
    assert isinstance(result, dict)
    assert "run_info" in result
    assert isinstance(result["run_info"], dict)
    assert isinstance(result["run_info"]["default"], RunInfoModel)


def test_read_grid_shapes(raw_data):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup.ngx, raw_run_info.setup.ngy, raw_run_info.setup.ngz = 12, 15, 18
    raw_run_info.setup.ngxf = np.int64(24)
    raw_run_info.setup.ngyf, raw_run_info.setup.ngzf = 30, 36
    actual = RunInfo.from_data(raw_run_info).read()
    assert actual["grid_coarse_shape"] == [12, 15, 18]
    assert actual["grid_fine_shape"] == [24, 30, 36]
    assert all(type(n) is int for n in actual["grid_fine_shape"])


@pytest.mark.parametrize(
    "exit_status, expected",
    [("success", True), (b"success", True), ("error", False), ("killed", False)],
)
def test_is_success(raw_data, exit_status, expected):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup.exit_status = exit_status
    assert RunInfo.from_data(raw_run_info).read()["is_success"] is expected


@pytest.mark.parametrize("exit_status", ["success  ", b"success   "])
def test_is_success_with_padded_status(raw_data, exit_status):
    # Fortran pads fixed-length character variables with blanks
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup.exit_status = exit_status
    run_info = RunInfo.from_data(raw_run_info)
    assert run_info.read()["is_success"] is True
    assert "    exit status: success\n" in str(run_info)


@pytest.mark.parametrize(
    "ispin, lnoncollinear, collinear, noncollinear",
    [(2, 0, True, False), (1, 1, False, True), (1, 0, False, False)],
)
def test_spin_from_setup_preferred(
    raw_data, ispin, lnoncollinear, collinear, noncollinear
):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.len_dos = 4  # contradicts the setup, which takes precedence
    raw_run_info.setup.ispin = ispin
    raw_run_info.setup.lnoncollinear = lnoncollinear
    actual = RunInfo.from_data(raw_run_info).read()
    assert actual["is_collinear"] is collinear
    assert actual["is_noncollinear"] is noncollinear


@pytest.mark.parametrize("missing_setup", [None, raw.VaspData(None)])
def test_without_setup_falls_back(raw_data, missing_setup):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup = missing_setup
    raw_run_info.len_dos = 2
    actual = RunInfo.from_data(raw_run_info).read()
    assert actual["grid_coarse_shape"] is None
    assert actual["grid_fine_shape"] is None
    assert actual["is_success"] is None
    assert actual["is_collinear"] is True
    assert actual["is_noncollinear"] is False


def test_setup_with_missing_datasets_falls_back(raw_data):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup = raw.RunSetup()
    raw_run_info.len_dos = 4
    actual = RunInfo.from_data(raw_run_info).read()
    assert actual["grid_coarse_shape"] is None
    assert actual["is_success"] is None
    assert actual["is_noncollinear"] is True


def test_print(run_info, format_):
    actual, _ = format_(run_info)
    reference = """\
run info for Sr2TiO4:
    VASP version: 99.99.99
    exit status: success
    ionic steps: 4
    Fermi energy: 0.500
    spin: nonpolarized
    metallic: no
    phonon dispersion: 20 q-points, 21 modes
    FFT grid: 32 x 32 x 40 (fine 64 x 64 x 80)
    SIGMA: 0.05 eV
    NELM: 60
    peak memory: 256.0 MB
    parallelization: 64 MPI ranks, KPAR=4, NCORE=8
    error messages: 1
    extensions: dimensions, dipole, dos, electronic, exchange_correlation, ionic, kpoints_opt, linear_response, metadata/build, metadata/performance, metadata/resources, vdw, write_flags"""
    assert actual == {"text/plain": reference}


def test_print_lists_extension_sections(raw_data):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.extension = raw.RunInfoExtension(
        sections={"zeta": raw.SetupSection(), "alpha": raw.SetupSection()}
    )
    assert "    extensions: alpha, zeta" in str(RunInfo.from_data(raw_run_info))


def test_print_partial_parallelization(raw_data):
    # serial builds write no MPI resources, so only what VASP wrote is shown
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup.mpi_ranks = raw.VaspData(None)
    raw_run_info.setup.ncore = raw.VaspData(None)
    assert "    parallelization: KPAR=4\n" in str(RunInfo.from_data(raw_run_info))
    raw_run_info.setup.kpar = raw.VaspData(None)
    assert "parallelization" not in str(RunInfo.from_data(raw_run_info))


def test_print_omits_unknown_quantities(raw_data, format_):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.runtime = None
    raw_run_info.phonon_dispersion = None
    raw_run_info.setup = None
    raw_run_info.extension = None
    actual, _ = format_(RunInfo.from_data(raw_run_info))
    reference = """\
run info for Sr2TiO4:
    ionic steps: 4
    Fermi energy: 0.500
    spin: nonpolarized
    metallic: no"""
    assert actual == {"text/plain": reference}


@pytest.mark.parametrize(
    "len_dos, expected",
    [
        (2, "    spin: collinear"),
        (4, "    spin: noncollinear"),
        (1, "    spin: nonpolarized"),
    ],
)
def test_print_reports_the_spin_configuration(raw_data, len_dos, expected):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup = None  # the spin falls back to the DOS
    raw_run_info.len_dos = len_dos
    assert expected in str(RunInfo.from_data(raw_run_info))


@pytest.mark.parametrize(
    "known_field", ("band_dispersion_eigenvalues", "band_projections")
)
def test_print_omits_half_known_spin_configuration(raw_data, known_field):
    # without a DOS the two flags fall back to different fields, so one of them may be
    # False while the other is simply unknown; that is not enough to claim nonpolarized
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup = None
    raw_run_info.len_dos = None
    for field in ("band_dispersion_eigenvalues", "band_projections"):
        if field != known_field:
            setattr(raw_run_info, field, None)
    assert "spin:" not in str(RunInfo.from_data(raw_run_info))


def test_print_omits_unknown_spin_configuration(raw_data):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup = None
    raw_run_info.len_dos = None
    raw_run_info.band_dispersion_eigenvalues = None
    raw_run_info.band_projections = None
    assert "spin:" not in str(RunInfo.from_data(raw_run_info))


def test_print_writes_to_stdout(run_info, capsys):
    assert run_info.print() is None
    assert capsys.readouterr().out == str(run_info) + "\n"


def test_selections(run_info):
    assert run_info.selections() == {"run_info": ["default"]}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.run_info("Sr2TiO4")
    check_factory_methods(RunInfo, data, skip_methods=["selections"])

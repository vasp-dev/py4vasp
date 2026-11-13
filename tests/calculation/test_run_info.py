# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import json
import types

import numpy as np
import pytest

from py4vasp import exception, raw
from py4vasp._calculation._CONTCAR import CONTCARHandler
from py4vasp._calculation._dispersion import DispersionHandler
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
    assert models.exit_status == setup.exit_status
    assert models.EFERMI == _decode(setup.efermi_method)
    assert models.fermi_energy == runinfo_ref.fermi_energy

    # core settings
    for field, expected in _expected_core_settings(setup).items():
        assert getattr(models, field) == expected, field


CORE_BOOLS = (
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
CORE_STRINGS = (
    "LREAL",
    "PREC",
    "ALGO",
    "XC",
    "build_variant",
)
CORE_SCALARS = (
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
    "mpi_ranks",
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
    "wall_time",
    "cpu_time",
    "omp_threads",
    "nodes",
    "gpus",
)


def _expected_core_settings(setup):
    return {
        **{field: getattr(setup, field.lower()) for field in CORE_SCALARS},
        **{field: bool(getattr(setup, field.lower())) for field in CORE_BOOLS},
        **{field: _decode(getattr(setup, field.lower())) for field in CORE_STRINGS},
    }


def _decode(text):
    return text.decode().strip() if isinstance(text, bytes) else text.strip()


def test_read(run_info):
    actual = run_info.read()
    expected = _expected_core_settings(run_info.ref.setup)
    assert {key: actual[key] for key in expected} == expected
    actual.pop("extensions")  # checked separately, not part of the database model
    _check_dict(RunInfoModel(**actual), run_info.ref)


DROPPED_KEYS = (
    "is_magnetic",
    "magnetization_order",
    "build_mpi",
    "is_collinear",
    "is_noncollinear",
    "is_success",
    "is_metallic",
)


@pytest.mark.parametrize("key", DROPPED_KEYS)
def test_dropped_keys_are_absent(run_info, key):
    # redundant with the core settings or never implemented
    assert key not in run_info.read()
    assert not hasattr(RunInfoModel(), key)


def test_core_bools_are_bool(run_info):
    actual = run_info.read()
    for field in CORE_BOOLS:
        assert type(actual[field]) is bool, field


def test_lreal_of_demo_is_vasp_string(run_info):
    assert run_info.read()["LREAL"] == "Auto"


@pytest.mark.parametrize("lreal", [b"Auto", b"On", b"F", b"T", b"Auto  ", "On"])
def test_lreal_keeps_vasp_spelling(raw_data, lreal):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup.lreal = lreal
    actual = RunInfo.from_data(raw_run_info).read()["LREAL"]
    assert actual == _decode(lreal)
    assert (
        RunInfo.from_data(raw_run_info)._to_database()["run_info"]["default"].LREAL
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
    assert actual["ALGO"] is None
    assert actual["PREC"] == "accurate"


def test_ldau_is_false_without_ldau_group(run_info):
    # VASP writes the ldau group only if LDA+U is active
    assert run_info.read()["LDAU"] is False
    assert run_info._to_database()["run_info"]["default"].LDAU is False


@pytest.mark.parametrize("ldautype", [1, 2, 4])
def test_ldau_is_true_if_vasp_wrote_ldautype(raw_data, ldautype):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup.ldautype = ldautype
    assert RunInfo.from_data(raw_run_info).read()["LDAU"] is True


def test_ldau_without_setup_is_none(raw_data):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup = None
    assert RunInfo.from_data(raw_run_info).read()["LDAU"] is None


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
    # a bug in VASP is the most severe message, so it counts as an error
    assert actual["error_messages"] == ["first error", "second error", "a bug"]
    assert actual["error_message_counts"] == [2, 1, 1]
    assert all(type(count) is int for count in actual["error_message_counts"])
    model = handler.to_database()
    assert model.error_messages == ["first error", "second error", "a bug"]
    assert model.error_message_counts == [2, 1, 1]


@pytest.mark.parametrize("wrap", [lambda x: x, np.array, raw.VaspData])
def test_single_message_as_scalars(raw_data, wrap):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup.message_text = wrap(b"the only error")
    raw_run_info.setup.message_severity = wrap(b"error")
    raw_run_info.setup.message_count = wrap(np.int64(3))
    actual = RunInfo.from_data(raw_run_info).read()
    assert actual["error_messages"] == ["the only error"]
    assert actual["error_message_counts"] == [3]


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
    # VASP writes no message arrays if it issued no message
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup.number_messages = 0
    raw_run_info.setup.message_text = raw.VaspData(None)
    raw_run_info.setup.message_severity = raw.VaspData(None)
    raw_run_info.setup.message_count = raw.VaspData(None)
    actual = RunInfo.from_data(raw_run_info).read()
    assert actual["error_messages"] == []
    assert actual["error_message_counts"] == []


def test_messages_of_killed_run_are_unknown(raw_data):
    # a killed run only wrote the exit status at startup, not the messages
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup.exit_status = b"killed "
    raw_run_info.setup.number_messages = raw.VaspData(None)
    raw_run_info.setup.message_text = raw.VaspData(None)
    raw_run_info.setup.message_severity = raw.VaspData(None)
    raw_run_info.setup.message_count = raw.VaspData(None)
    handler = RunInfoHandler.from_data(raw_run_info)
    actual = handler.to_dict()
    assert actual["exit_status"] == "killed"
    assert actual["error_messages"] is None
    assert actual["error_message_counts"] is None
    handler.to_database()  # the rest of the record is still stored


@pytest.mark.parametrize("system", [b"Sr2TiO4 relaxed   ", "Sr2TiO4 relaxed"])
def test_system_tag_is_text(raw_data, system):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.system = raw.System(system)
    actual = RunInfo.from_data(raw_run_info).read()
    assert actual["system_tag"] == "Sr2TiO4 relaxed"
    json.dumps(actual["system_tag"])


@pytest.mark.parametrize("method", ["read", "to_dict", "print"])
def test_section_name_as_selection_points_to_extensions(run_info, method):
    # section names look like sources, but they are keys of the extensions
    with pytest.raises(exception.IncorrectUsage) as error:
        getattr(run_info, method)("electronic")
    assert 'read()["extensions"]["electronic"]' in str(error.value)


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
    [
        ("success", "success"),
        (b"error", "error"),
        ("killed    ", "killed"),
        (b"dry-run   ", "dry-run"),
    ],
)
def test_exit_status_is_trimmed_string(raw_data, exit_status, expected):
    # Fortran pads fixed-length character variables with blanks
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup.exit_status = exit_status
    run_info = RunInfo.from_data(raw_run_info)
    assert run_info.read()["exit_status"] == expected
    assert run_info._to_database()["run_info"]["default"].exit_status == expected
    assert f"    exit status: {expected}\n" in str(run_info)


@pytest.mark.parametrize("missing_setup", [None, raw.VaspData(None), raw.RunSetup()])
def test_without_setup_derived_values_are_none(raw_data, missing_setup):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup = missing_setup
    actual = RunInfo.from_data(raw_run_info).read()
    assert actual["grid_coarse_shape"] is None
    assert actual["grid_fine_shape"] is None
    assert actual["exit_status"] is None
    assert actual["EFERMI"] is None
    # nothing recorded means unknown, not "LDA+U off" or "no errors"
    assert actual["LDAU"] is None
    assert actual["error_messages"] is None
    assert actual["error_message_counts"] is None
    assert "error messages" not in str(RunInfo.from_data(raw_run_info))


def test_efermi_is_the_method_vasp_used(run_info):
    assert run_info.read()["EFERMI"] == "LEGACY"


@pytest.mark.parametrize("method", [b"MIDGAP  ", "LEGACY"])
def test_efermi_trims_the_method(raw_data, method):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup.efermi_method = method
    assert RunInfo.from_data(raw_run_info).read()["EFERMI"] == _decode(method)


def test_efermi_set_to_a_number_is_reported_as_text(raw_data):
    # https://vasp.at/wiki/EFERMI: EFERMI = MIDGAP | LEGACY | [real]; for a number
    # VASP writes the method FORCE together with the value. A database column holds
    # a single type, so the number is stored as the text of the value.
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup.efermi_method = b"FORCE  "
    raw_run_info.setup.efermi = np.float64(5.5)
    run_info = RunInfo.from_data(raw_run_info)
    assert run_info.read()["EFERMI"] == "5.5"
    assert run_info._to_database()["run_info"]["default"].EFERMI == "5.5"


def test_print(run_info, format_):
    actual, _ = format_(run_info)
    reference = """\
run info for Sr2TiO4:
    VASP version: 99.99.99
    exit status: success
    ionic steps: 4
    spin: nonpolarized
    phonon dispersion: 20 q-points, 21 modes
    FFT grid: 32 x 32 x 40 (fine 64 x 64 x 80)
    ionic update: IBRION=2, ISIF=3, NSW=40
    functional: pbe
    ENCUT: 520.0 eV
    ISMEAR: 0
    SIGMA: 0.05 eV
    EFERMI: LEGACY
    NELM: 60
    wall time: 1834.5 s
    peak memory: 256.0 MiB
    parallelization: 64 MPI ranks, KPAR=4, NCORE=8
    error messages:
        - The basis set is incomplete. (1×)
    extensions: dimensions, dipole, dos, electronic, exchange_correlation, ionic, kpoints_opt, linear_response, metadata/build, metadata/performance, metadata/resources, vdw, write_flags"""
    assert actual == {"text/plain": reference}


def test_print_lists_every_error_message_with_its_count(raw_data):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup.message_text = np.array([b"first error", b"a bug"])
    raw_run_info.setup.message_severity = np.array([b"error", b"bug"])
    raw_run_info.setup.message_count = np.array([2, 1])
    expected = """\
    error messages:
        - first error (2×)
        - a bug (1×)
"""
    assert expected in str(RunInfo.from_data(raw_run_info))


def test_print_without_error_messages(raw_data):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup.number_messages = 0
    assert "    error messages: none\n" in str(RunInfo.from_data(raw_run_info))


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


def test_print_partial_ionic_update(raw_data):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup.isif = raw.VaspData(None)
    assert "    ionic update: IBRION=2, NSW=40\n" in str(
        RunInfo.from_data(raw_run_info)
    )
    raw_run_info.setup.ibrion = raw.VaspData(None)
    raw_run_info.setup.nsw = raw.VaspData(None)
    assert "ionic update" not in str(RunInfo.from_data(raw_run_info))


def test_print_functional_with_lda_plus_u(raw_data):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup.ldautype = 2
    assert "    functional: pbe+U\n" in str(RunInfo.from_data(raw_run_info))


def test_print_of_vasp_without_setup(raw_data, format_):
    # VASP writes the exit status at startup, so without it the file predates the
    # setup writers; print says so instead of silently leaving the settings out
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.runtime = None
    raw_run_info.phonon_dispersion = None
    raw_run_info.setup = None
    raw_run_info.extension = None
    actual, _ = format_(RunInfo.from_data(raw_run_info))
    reference = """\
run info for Sr2TiO4:
    exit status: unknown (requires a newer VASP)
    ionic steps: 4
    settings: unknown (requires a newer VASP)"""
    assert actual == {"text/plain": reference}


@pytest.mark.parametrize(
    "ispin, lnoncollinear, expected",
    [
        (2, 0, "collinear"),
        (1, 1, "noncollinear"),
        (None, 1, "noncollinear"),
        (1, 0, "nonpolarized"),
    ],
)
def test_print_reports_the_spin_configuration(raw_data, ispin, lnoncollinear, expected):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup.ispin = raw.VaspData(ispin)
    raw_run_info.setup.lnoncollinear = lnoncollinear
    assert f"    spin: {expected}\n" in str(RunInfo.from_data(raw_run_info))


@pytest.mark.parametrize("ispin, lnoncollinear", [(None, 0), (2, None), (None, None)])
def test_print_omits_unknown_spin_configuration(raw_data, ispin, lnoncollinear):
    raw_run_info = raw_data.run_info("Sr2TiO4")
    raw_run_info.setup.ispin = raw.VaspData(ispin)
    raw_run_info.setup.lnoncollinear = raw.VaspData(lnoncollinear)
    assert "spin:" not in str(RunInfo.from_data(raw_run_info))


def test_print_writes_to_stdout(run_info, capsys):
    assert run_info.print() is None
    assert capsys.readouterr().out == str(run_info) + "\n"


def test_selections(run_info):
    assert run_info.selections() == {"run_info": ["default"]}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.run_info("Sr2TiO4")
    check_factory_methods(RunInfo, data, skip_methods=["selections"])

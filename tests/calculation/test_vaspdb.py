import os
import pathlib
import shutil
from functools import wraps

import pytest

from py4vasp import Calculation, exception
from py4vasp._calculation import DEFAULT_VASP_DB_NAME
from py4vasp._util import import_

vaspdb = import_.optional("vaspdb")


# Make sure tests are skipped if vaspdb is not installed
def dep_vaspdb(func):
    @wraps(func)
    @pytest.mark.skipif(
        import_.is_imported(vaspdb) is False, reason="vaspdb is not installed"
    )
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def _make_tmp_dir(path, tmp_path_factory):
    out_path = (
        tmp_path_factory.getbasetemp() / path
        if path is not None and (path != ".")
        else tmp_path_factory.getbasetemp()
    )
    if not (out_path.name.endswith(".db")):
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def _check_db_exists(expected_db_name, expected_db_path, tmp_path_factory):
    if expected_db_path is None:
        expected_out_path = tmp_path_factory.getbasetemp()
    else:
        expected_out_path = tmp_path_factory.getbasetemp() / expected_db_path
    if expected_db_name is None:
        expected_db_file_path = expected_out_path / DEFAULT_VASP_DB_NAME
    else:
        expected_db_file_path = expected_out_path / expected_db_name

    assert os.path.exists(expected_out_path)
    assert os.path.exists(expected_db_file_path)


@pytest.fixture
def VaspDB(tmp_path_factory):
    """Fixture to instantiate a vaspdb.VaspDB database object."""

    return vaspdb.VaspDB(
        db_path=str(_make_tmp_dir("test_database_path", tmp_path_factory)),
        db_name="test_database.db",
    )


@pytest.mark.skip
class TestToDatabase:
    """Test the to_database method of Calculation class."""

    calculation = Calculation.from_path(".")

    @dep_vaspdb
    @pytest.mark.parametrize(
        "db_name, db_path, expected_db_name, expected_db_path",
        [
            (None, None, DEFAULT_VASP_DB_NAME, None),
            ("test_database.db", None, "test_database.db", None),
            (None, "test_database_path", DEFAULT_VASP_DB_NAME, "test_database_path"),
            (None, "test_database_path.db", "test_database_path.db", None),
            (None, "test/test_database_path.db", "test_database_path.db", "test"),
            (
                "test_database",
                "test_database_path",
                "test_database.db",
                "test_database_path",
            ),
            ("test_database", "test_database.db", "test_database.db", None),
            (
                "test_database.db",
                "test_database_path",
                "test_database.db",
                "test_database_path",
            ),
        ],
    )
    def test_to_database(
        self, db_name, db_path, expected_db_name, expected_db_path, tmp_path_factory
    ):
        """Test valid to_database call with several valid specifications."""
        self.calculation.to_database(_make_tmp_dir(db_path, tmp_path_factory), db_name)
        _check_db_exists(expected_db_name, expected_db_path, tmp_path_factory)

        # TODO add more checks to verify content of the database

    @dep_vaspdb
    @pytest.mark.parametrize(
        "db_name, db_path",
        [
            ("test_database.db", "test_database_path.db"),
            ("test_database", "test_database_path.db"),
        ],
    )
    def test_to_database_duplicate_specification(
        self, db_name, db_path, tmp_path_factory
    ):
        """Test invalid to_database call with conflicting db_name and db_path."""
        with pytest.raises(exception.IncorrectUsage):
            self.calculation.to_database(
                _make_tmp_dir(db_path, tmp_path_factory), db_name
            )

    @dep_vaspdb
    def test_to_database_with_vaspdb_object(self, VaspDB, tmp_path_factory):
        """Test to_database call with a provided VaspDB class object."""

        self.calculation.to_database(VaspDB=VaspDB)
        _check_db_exists("test_database.db", "test_database_path", tmp_path_factory)

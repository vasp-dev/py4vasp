# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from py4vasp import Batch


def test_error_when_using_constructor():
    with pytest.raises(Exception):
        Batch()


def test_creation_from_paths():
    # Test creation from absolute paths
    absolute_path_1 = Path(__file__) / "path_1"
    absolute_path_2 = Path(__file__) / "path_2"
    batch = Batch.from_paths(path_name_1=absolute_path_1, path_name_2=absolute_path_2)
    output_paths = batch.paths()
    assert output_paths["path_name_1"] == [absolute_path_1]
    assert output_paths["path_name_2"] == [absolute_path_2]
    output_number_of_calculations = batch.number_of_calculations()
    assert output_number_of_calculations["path_name_1"] == 1
    assert output_number_of_calculations["path_name_2"] == 1
    # Test creation from relative paths
    relative_path_1 = os.path.relpath(absolute_path_1, Path.cwd())
    relative_path_2 = os.path.relpath(absolute_path_2, Path.cwd())
    batch = Batch.from_paths(path_name_1=relative_path_1, path_name_2=relative_path_2)
    output_paths = batch.paths()
    assert output_paths["path_name_1"] == [absolute_path_1]
    assert output_paths["path_name_2"] == [absolute_path_2]
    output_number_of_calculations = batch.number_of_calculations()
    assert output_number_of_calculations["path_name_1"] == 1
    assert output_number_of_calculations["path_name_2"] == 1
    # Test creation with string paths
    batch = Batch.from_paths(
        path_name_1=absolute_path_1.as_posix(), path_name_2=absolute_path_2.as_posix()
    )
    output_paths = batch.paths()
    assert output_paths["path_name_1"] == [absolute_path_1]
    assert output_paths["path_name_2"] == [absolute_path_2]
    output_number_of_calculations = batch.number_of_calculations()
    assert output_number_of_calculations["path_name_1"] == 1
    assert output_number_of_calculations["path_name_2"] == 1


def test_creation_from_paths_with_incorrect_input():
    with pytest.raises(Exception):
        Batch.from_paths(path_name_1=1, path_name_2=2)


def test_creation_from_paths_with_wildcards(tmp_path):
    paths_1 = [tmp_path / "path1_a", tmp_path / "path1_b"]
    absolute_paths_1 = [path.resolve() for path in paths_1]
    paths_2 = [tmp_path / "path2_a", tmp_path / "path2_b"]
    absolute_paths_2 = [path.resolve() for path in paths_2]
    create_paths = lambda paths: [path.mkdir() for path in paths]
    create_paths(paths_1)
    create_paths(paths_2)
    batch = Batch.from_paths(
        path_name_1=tmp_path / "path1_*", path_name_2=tmp_path / "path2_*"
    )
    output_paths = batch.paths()
    assert all(
        [
            output_paths["path_name_1"][i] == absolute_paths_1[i]
            for i in range(len(absolute_paths_1))
        ]
    )
    assert all(
        [
            output_paths["path_name_2"][i] == absolute_paths_2[i]
            for i in range(len(absolute_paths_2))
        ]
    )
    output_number_of_calculations = batch.number_of_calculations()
    assert output_number_of_calculations["path_name_1"] == 2
    assert output_number_of_calculations["path_name_2"] == 2


def test_creation_from_file():
    absolute_path_1 = Path(__file__) / "example_1.h5"
    absolute_path_2 = Path(__file__) / "example_2.h5"
    batch = Batch.from_files(path_name_1=absolute_path_1, path_name_2=absolute_path_2)
    output_paths = batch.paths()
    assert output_paths["path_name_1"] == [absolute_path_1.parent]
    assert output_paths["path_name_2"] == [absolute_path_2.parent]
    output_number_of_calculations = batch.number_of_calculations()
    assert output_number_of_calculations["path_name_1"] == 1
    assert output_number_of_calculations["path_name_2"] == 1


def test_create_from_files_with_wildcards(tmp_path):
    paths_1 = [tmp_path / "example1_a.h5", tmp_path / "example1_b.h5"]
    absolute_paths_1 = [path.resolve() for path in paths_1]
    paths_2 = [tmp_path / "example2_a.h5", tmp_path / "example2_b.h5"]
    absolute_paths_2 = [path.resolve() for path in paths_2]
    create_files = lambda paths: [path.touch() for path in paths]
    create_files(paths_1)
    create_files(paths_2)
    batch = Batch.from_files(
        file_1=tmp_path / "example1_*.h5",
        file_2=tmp_path / "example2_*.h5",
    )
    output_paths = batch.paths()
    assert all(
        [
            output_paths["file_1"][i] == absolute_paths_1[i].parent
            for i in range(len(absolute_paths_1))
        ]
    )
    assert all(
        [
            output_paths["file_2"][i] == absolute_paths_2[i].parent
            for i in range(len(absolute_paths_2))
        ]
    )
    output_number_of_calculations = batch.number_of_calculations()
    assert output_number_of_calculations["file_1"] == 2
    assert output_number_of_calculations["file_2"] == 2


@patch("py4vasp._calculation.base.Refinery.from_path", autospec=True)
@patch("py4vasp.raw.access", autospec=True)
def test_has_attributes(mock_access, mock_from_path):
    batch = Batch.from_paths(path_name_1="path_1", path_name_2="path_2")
    assert hasattr(batch, "energies")
    assert hasattr(batch.energies, "read")
    output_read = batch.energies.read()
    assert isinstance(output_read, dict)
    assert output_read.keys() == {"path_name_1", "path_name_2"}
    assert isinstance(output_read["path_name_1"], list)
    assert isinstance(output_read["path_name_2"], list)
    assert hasattr(batch, "forces")
    assert hasattr(batch.forces, "read")
    output_read = batch.forces.read()
    assert isinstance(output_read, dict)
    assert output_read.keys() == {"path_name_1", "path_name_2"}
    assert isinstance(output_read["path_name_1"], list)
    assert isinstance(output_read["path_name_2"], list)
    assert hasattr(batch, "stresses")
    assert hasattr(batch.stresses, "read")
    output_read = batch.stresses.read()
    assert isinstance(output_read, dict)
    assert output_read.keys() == {"path_name_1", "path_name_2"}
    assert isinstance(output_read["path_name_1"], list)
    assert isinstance(output_read["path_name_2"], list)

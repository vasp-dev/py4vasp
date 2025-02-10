# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest
from click.testing import CliRunner

from py4vasp.cli import cli


@pytest.mark.parametrize("lammps", ("LAMMPS", "Lammps", "lammps"))
def test_convert_lammps(lammps):
    runner = CliRunner()
    result = runner.invoke(cli, ["convert", lammps])
    assert result.exit_code == 0

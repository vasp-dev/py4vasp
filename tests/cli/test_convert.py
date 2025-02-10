# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from click.testing import CliRunner
from py4vasp.cli import cli


def test_convert():
    runner = CliRunner()
    result = runner.invoke(cli)  # , ["convert"])
    assert result.exit_code == 0

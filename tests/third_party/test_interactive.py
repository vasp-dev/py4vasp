# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from unittest.mock import patch

import py4vasp._third_party.interactive as interactive


def test_no_error_handling_outside_ipython():
    with patch("IPython.get_ipython", return_value=None) as mock:
        interactive.set_error_handling("Minimal")


def test_set_error_handling(capsys):
    with patch("IPython.get_ipython") as mock:
        interactive.set_error_handling("Minimal")
        mock.return_value.magic.assert_called_once_with("xmode Minimal")
        output, _ = capsys.readouterr()
        assert output == ""

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from py4vasp._util import import_

pretty = import_.optional("IPython.lib.pretty")


class AbstractTest:
    def test_from_string(self):
        text = "! comment line"
        with patch("py4vasp._control.base.open", mock_open()) as mock:
            instance = self.tested_class.from_string(text)
            mock.assert_not_called()
            assert str(instance) == instance.read() == text

    def test_from_string_to_file(self):
        text = "! comment line"
        path = "file_path"
        with patch("py4vasp._control.base.open", mock_open(read_data=text)) as mock:
            instance = self.tested_class.from_string(text, path)
            filename = Path(f"{path}/{self.tested_class.__name__}")
            mock.assert_called_once_with(filename, "w")
            mock().write.assert_called_once_with(text)
            mock.reset_mock()
            assert str(instance) == text
            mock.assert_called_once_with(filename, "r")
            mock().read.assert_called_once_with()

    def test_from_path(self):
        text = "! comment line"
        path = "file_path"
        with patch("py4vasp._control.base.open", mock_open(read_data=text)) as mock:
            instance = self.tested_class(path)
            assert instance.read() == text
            filename = Path(f"{path}/{self.tested_class.__name__}")
            mock.assert_called_once_with(filename, "r")
            mock().read.assert_called_once_with()
            mock.reset_mock()
            instance.write(text)
            mock.assert_called_once_with(filename, "w")
            mock().write.assert_called_once_with(text)

    def test_read_instance(self):
        text = "! comment line"
        instance = self.tested_class.from_string(text)
        assert instance.read() == text

    def test_print_instance(self):
        text = "! comment line"
        instance = self.tested_class.from_string(text)
        with redirect_stdout(StringIO()) as buffer:
            instance.print()
            assert buffer.getvalue().strip() == text

    @pytest.mark.skipif(
        not import_.is_imported(pretty),
        reason="This test requires pretty from IPython.",
    )
    def test_pretty_instance(self):
        text = "! comment line"
        instance = self.tested_class.from_string(text)
        assert pretty.pretty(instance) == text

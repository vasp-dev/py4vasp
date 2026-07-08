import os
import shutil

import pytest


def test_error_analysis():
    if shutil.which("error-analysis") is None:
        pytest.skip("error-analysis entry point not installed (py4vasp-core)")
    errcode = os.system("error-analysis --help")
    assert errcode == 0

import os


def test_error_analysis(not_core):
    errcode = os.system("error-analysis --help")
    assert errcode == 0

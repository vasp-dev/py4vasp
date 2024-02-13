import os


def test_error_analysis():
    errcode = os.system("error-analysis --help")
    assert errcode == 0

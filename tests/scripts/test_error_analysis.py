import subprocess


def test_error_analysis():
    proc = subprocess.check_output(["error-analysis", "--help"])
    assert proc.decode("utf-8").startswith("usage: error-analysis")

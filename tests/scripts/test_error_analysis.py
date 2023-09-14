import subprocess


def test_error_analysis():
    proc = subprocess.run(["error-analysis", "--help"])
    assert proc.returncode == 0

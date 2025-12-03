# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest

application = pytest.importorskip("sphinx.application")


@pytest.fixture(scope="module")
def sphinx_app_trial(not_core):
    import shutil
    from os import mkdir, path

    tmp_path = path.abspath(
        path.join(path.dirname(path.abspath(__file__)), "_trial_build")
    )
    if path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    if not path.exists(tmp_path):
        mkdir(tmp_path)
    srcdir = "src/py4vasp"
    confdir = "tests/sphinx/trial_doc"
    outdir = path.join(tmp_path, "_build")
    doctreedir = path.join(tmp_path, "_doctree")
    app = application.Sphinx(
        srcdir=srcdir,
        confdir=confdir,
        outdir=outdir,
        doctreedir=doctreedir,
        buildername="hugo",
        status=None,
        warning=None,
        freshenv=True,
    )
    app.build(force_all=True)
    return app


def test_build_analysis(sphinx_app_trial, not_core):
    """Test that the analysis module builds without errors."""
    assert sphinx_app_trial.statuscode == 0

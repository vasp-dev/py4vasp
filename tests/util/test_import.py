# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import subprocess
import sys

import pytest

from py4vasp import exception
from py4vasp._util import import_


def test_import_not_available():
    module = import_.optional("_name_which_does_not_exist_")
    assert not import_.is_imported(module)
    with pytest.raises(exception.ModuleNotInstalled):
        module.attribute


def test_import_for_existing_module():
    module = import_.optional("py4vasp")
    assert import_.is_imported(module)


def test_optional_defers_import_until_first_use():
    # colorsys is a tiny stdlib module py4vasp never imports; use it as a probe.
    sys.modules.pop("colorsys", None)
    module = import_.optional("colorsys")
    assert "colorsys" not in sys.modules  # merely creating the proxy imports nothing
    result = module.rgb_to_hls(0.0, 0.0, 0.0)  # attribute access triggers the import
    assert "colorsys" in sys.modules
    import colorsys

    assert module.rgb_to_hls is colorsys.rgb_to_hls
    assert result == colorsys.rgb_to_hls(0.0, 0.0, 0.0)


def test_is_imported_resolves_lazy_module():
    sys.modules.pop("colorsys", None)
    module = import_.optional("colorsys")
    # is_imported must answer "is it available?" even though nothing imported yet.
    assert import_.is_imported(module)


def test_import_py4vasp_defers_heavy_dependencies():
    # These libraries are only needed for specific features (plotting, viewing,
    # symmetry, ...); `import py4vasp` must not pull any of them in eagerly.
    deferred = (
        "plotly",
        "nglview",
        "ase",
        "scipy",
        "mdtraj",
        "spglib",
        "pandas",
        "IPython",
    )
    code = (
        "import sys, py4vasp;"
        f"loaded = sorted(m for m in {deferred} if m in sys.modules);"
        "assert not loaded, loaded"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Set the version of every distribution in the workspace.

py4vasp-core reads its version from ``src/py4vasp/__init__.py``, but the two wrapper
distributions have to repeat it: a metadata-only package has no module to read it from,
and both pin ``py4vasp-core`` exactly. This script keeps the four places in step;
``tests/test_version.py`` fails if they ever drift apart.

Usage: python scripts/set_version.py 0.12.0
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).parent.parent
MODULES = (
    ROOT / "src" / "py4vasp" / "__init__.py",
    ROOT / "packages" / "backend" / "src" / "vasp" / "backend" / "__init__.py",
)
PYPROJECTS = (
    ROOT / "packages" / "py4vasp" / "pyproject.toml",
    ROOT / "packages" / "backend" / "pyproject.toml",
)


def main(version):
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise SystemExit(f"'{version}' is not a <major>.<minor>.<patch> version.")
    for path in MODULES:
        substitute(path, r'^__version__ = ".*"$', f'__version__ = "{version}"')
    for path in PYPROJECTS:
        substitute(path, r'^version = ".*"$', f'version = "{version}"', optional=True)
        # The pins appear both as their own line in `dependencies` and inline in an
        # extra, so match the requirement itself rather than the whole line.
        substitute(
            path,
            r"py4vasp-core(\[[a-z,]*\])?==\d+\.\d+\.\d+",
            rf"py4vasp-core\g<1>=={version}",
        )


def substitute(path, pattern, replacement, optional=False):
    text = path.read_text()
    new_text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count == 0 and not optional:
        raise SystemExit(f"Found nothing matching {pattern!r} in {path}.")
    if new_text != text:
        path.write_text(new_text)
        print(f"updated {count} line(s) in {path.relative_to(ROOT)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])

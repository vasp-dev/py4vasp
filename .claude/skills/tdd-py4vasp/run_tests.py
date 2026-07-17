#!/usr/bin/env python
"""Run pytest against THIS checkout's src, then pass through any pytest args.

py4vasp is installed editable via a plain `.pth` file that points at the *main*
repo's `src`. Inside a `.claude/worktrees/<name>` worktree that means a bare
`pytest` silently imports the main repo and ignores your edits. This wrapper
finds the nearest `src/py4vasp` above the current directory, puts it first on
PYTHONPATH so it wins over the `.pth` entry, prints which one it picked, and
hands the rest of argv to pytest.

Usage (from anywhere inside the checkout):
    python .claude/skills/tdd-py4vasp/run_tests.py tests/calculation/test_symmetry.py::test_read
    python .claude/skills/tdd-py4vasp/run_tests.py tests/calculation/test_symmetry.py -q
    python .claude/skills/tdd-py4vasp/run_tests.py            # whole suite
"""
import os
import pathlib
import subprocess
import sys


def find_src(start: pathlib.Path) -> pathlib.Path:
    for parent in [start, *start.parents]:
        candidate = parent / "src" / "py4vasp"
        if candidate.is_dir():
            return parent / "src"
    sys.exit("error: could not find a src/py4vasp above " + str(start))


def main() -> int:
    src = find_src(pathlib.Path.cwd())
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src) + (os.pathsep + existing if existing else "")
    print(f"[run_tests] importing py4vasp from: {src}", flush=True)
    cmd = [sys.executable, "-m", "pytest", *sys.argv[1:]]
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())

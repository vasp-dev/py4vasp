#!/usr/bin/env python
"""Gather review evidence for the py4vasp changes on this branch.

Run this with the project's *venv* interpreter (it needs pytest-cov), from
anywhere inside the checkout:

    .venv/Scripts/python.exe .claude/skills/review-py4vasp/review.py            # vs master
    .venv/Scripts/python.exe .claude/skills/review-py4vasp/review.py main       # vs another base

What it prints, for the diff between <base> (default: master) and the working tree:
  1. Changed files, split into src modules / test files / other.
  2. Coverage of each changed src module (whole-package run over the mapped
     test files, filtered to the changed files) + the uncovered line numbers.
  3. Doctest pass/fail for the changed modules.
  4. A raise-convention scan: `raise X` in changed src files where X is not a
     py4vasp `exception.*` (candidates only — the reviewer judges them).
  5. Selection-API hints: where changed files mention `selection`, `select.Tree`,
     or `index.Selector`.

Everything here is *evidence*, not verdicts. The reviewer reads it, reads the
diff, and files findings via ReportFindings. See SKILL.md.

Why the odd pytest flags: py4vasp is installed editable via a `.pth` pointing at
the *main* repo src, so PYTHONPATH is forced to this checkout's src (else a
worktree silently measures the main repo). And coverage is requested as
`--cov=py4vasp` (whole package) NOT `--cov=py4vasp._calculation.foo`: the dotted
form trips a numpy-2 "cannot load module more than once" crash under Python 3.14.
"""
import os
import pathlib
import re
import subprocess
import sys


def sh(*args, **kwargs):
    return subprocess.run(args, text=True, capture_output=True, **kwargs)


def find_root(start):
    for parent in [start, *start.parents]:
        if (parent / "src" / "py4vasp").is_dir() and (parent / "tests").is_dir():
            return parent
    sys.exit("error: could not find a py4vasp checkout (src/py4vasp + tests) above " + str(start))


def changed_files(root, base):
    # committed changes base..HEAD plus anything dirty in the working tree
    out = sh("git", "-C", str(root), "diff", "--name-only", base, "HEAD").stdout
    out += sh("git", "-C", str(root), "diff", "--name-only").stdout
    out += sh("git", "-C", str(root), "diff", "--name-only", "--cached").stdout
    files = sorted({line.strip() for line in out.splitlines() if line.strip()})
    return [f for f in files if (root / f).exists()]


def map_src_to_tests(root, src_rel):
    """Best-effort: src/py4vasp/_calculation/symmetry.py -> tests/**/test_symmetry.py"""
    stem = pathlib.Path(src_rel).stem.lstrip("_")
    return sorted(str(p.relative_to(root)) for p in root.glob(f"tests/**/test_{stem}.py"))


def run_pytest(root, env, extra):
    cmd = [sys.executable, "-m", "pytest", *extra, "-q"]
    return subprocess.run(cmd, cwd=str(root), env=env, text=True, capture_output=True)


def coverage_rows(stdout, src_files):
    wanted = {f.replace("/", os.sep) for f in src_files}
    rows = []
    for line in stdout.splitlines():
        for w in wanted:
            if w in line.replace("/", os.sep) and "%" in line:
                rows.append(line.strip())
    return rows


def scan_raises(root, src_files):
    hits = []
    pattern = re.compile(r"^\s*raise\s+(\S+)")
    for f in src_files:
        for n, line in enumerate((root / f).read_text(encoding="utf-8").splitlines(), 1):
            m = pattern.match(line)
            if not m:
                continue
            token = m.group(1)
            if token.startswith("exception."):
                continue
            hits.append(f"{f}:{n}: {line.strip()}")
    return hits


def scan_selection(root, src_files):
    hits = []
    needles = ("selection", "select.Tree", "index.Selector")
    for f in src_files:
        for n, line in enumerate((root / f).read_text(encoding="utf-8").splitlines(), 1):
            if any(needle in line for needle in needles):
                hits.append(f"{f}:{n}: {line.strip()}")
    return hits


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "master"
    root = find_root(pathlib.Path.cwd())
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + (os.pathsep + env.get("PYTHONPATH", ""))
    print(f"[review] checkout : {root}")
    print(f"[review] base ref : {base}")
    print(f"[review] python   : {sys.executable}")

    files = changed_files(root, base)
    src = [f for f in files if f.startswith("src/py4vasp/") and f.endswith(".py")]
    tests = [f for f in files if f.startswith("tests/") and f.endswith(".py")]
    other = [f for f in files if f not in src and f not in tests]

    section("1. CHANGED FILES")
    print("src modules:", *(["\n  " + f for f in src] or [" (none)"]))
    print("test files :", *(["\n  " + f for f in tests] or [" (none)"]))
    print("other      :", *(["\n  " + f for f in other] or [" (none)"]))

    if not src:
        print("\nNo changed py4vasp src modules — coverage/doctest/style scans skipped.")
        return 0

    # collect test files: those in the diff + mapped from changed src
    test_targets = list(tests)
    unmapped = []
    for s in src:
        mapped = map_src_to_tests(root, s)
        if mapped:
            test_targets.extend(mapped)
        else:
            unmapped.append(s)
    test_targets = sorted(set(test_targets))

    section("2. COVERAGE OF CHANGED SRC (whole-package run, filtered)")
    if test_targets:
        print("running:", " ".join(["pytest", *test_targets, "--cov=py4vasp", "--cov-report=term-missing"]))
        result = run_pytest(root, env, [*test_targets, "--cov=py4vasp", "--cov-report=term-missing"])
        rows = coverage_rows(result.stdout, src)
        print("\nName                                          Stmts   Miss   Cover   Missing")
        for row in rows or ["  (no coverage rows matched the changed src files)"]:
            print("  " + row)
        tail = [l for l in result.stdout.splitlines() if " passed" in l or " failed" in l or " error" in l]
        print("\npytest:", tail[-1].strip() if tail else "(see full output)")
        if result.returncode != 0:
            print("!! tests did not all pass — coverage numbers may be unreliable")
    if unmapped:
        print("\n!! no test file auto-mapped for (locate & run manually):", *("\n  " + u for u in unmapped))

    section("3. DOCTESTS FOR CHANGED MODULES")
    stems = " or ".join(sorted({pathlib.Path(s).stem for s in src}))
    print(f"running: pytest tests/test_doctest.py -k \"{stems}\"")
    result = run_pytest(root, env, ["tests/test_doctest.py", "-k", stems])
    tail = [l for l in result.stdout.splitlines() if "passed" in l or "failed" in l or "error" in l or "no tests ran" in l]
    print(tail[-1].strip() if tail else result.stdout[-400:])

    section("4. RAISE-CONVENTION SCAN (candidates - reviewer judges)")
    print("Rule: user-facing raises should be py4vasp `exception.*`. Legit exceptions:")
    print("re-raising a caught error, plus framework code (_sphinx/ docutils, cli.py click).")
    raises = scan_raises(root, src)
    for h in raises or ["  (no non-exception.* raises in changed src)"]:
        print("  " + h)

    section("5. SELECTION-API HINTS")
    print("Rule: user option args are named `selection`, parsed via select.Tree,")
    print("and indexed via index.Selector when they map to array indices.")
    hits = scan_selection(root, src)
    for h in hits or ["  (no selection/Tree/Selector mentions in changed src)"]:
        print("  " + h)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

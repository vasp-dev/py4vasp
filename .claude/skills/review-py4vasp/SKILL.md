---
name: review-py4vasp
description: >-
  Code review for py4vasp changes. Use when asked to review, code-review, or
  critique py4vasp code, a branch, or a diff. Does the usual bug / edge-case
  review AND enforces four py4vasp-specific rules: exceptions come from
  py4vasp.exception, user selections use select.Tree / index.Selector, public
  methods have numpy-style docstrings with runnable doctest examples, and tests
  give >95% coverage. Runs doctests and coverage to substantiate findings, then
  reports via the ReportFindings tool.
---

# Reviewing py4vasp code

Standalone reviewer for the changes on the current branch (working tree +
commits) **relative to `master`**. It combines a normal correctness review with
four py4vasp house rules, and it *runs* the doctests and coverage rather than
guessing. Evidence is gathered by `.claude/skills/review-py4vasp/review.py`;
you read that evidence, read the diff, and file findings with **ReportFindings**
(most-severe first). Paths below are relative to the repo/worktree root.

## Step 1 — gather evidence (run the driver)

Run it with the **venv** interpreter (it needs `pytest-cov`, which the system
python lacks):

```bash
.venv/Scripts/python.exe .claude/skills/review-py4vasp/review.py
```

Pass a different base ref as the first arg if needed (`... review.py main`).
The driver prints five sections: changed files; coverage of each changed src
module with uncovered line numbers; doctest pass/fail for those modules; a
raise-convention scan; and selection-API hints. It takes ~15 s (a coverage run
over the touched test files). If a changed src module has no auto-mapped test
file, the driver says so — locate and run its tests manually (Step 3).

## Step 2 — review against the four py4vasp rules

Read the diff alongside the driver output and judge each rule. The driver gives
you leads; you decide what is a finding.

1. **Exceptions come from `py4vasp.exception`.** The contract (see the
   `exception.py` module docstring): *a py4vasp exception means the user made a
   mistake; any other exception is a bug to report.* So user-facing code must
   raise a `Py4VaspError` subclass (`IncorrectUsage`, `NoData`, `DataMismatch`,
   `NotImplemented`, …), not a bare `ValueError`/`KeyError`/`RuntimeError`.
   Section 4 lists every `raise X` in changed src where `X` is not
   `exception.*`. **Legit exceptions** (not findings): re-raising a caught error
   (`raise err`), and framework code — `src/py4vasp/_sphinx/` (docutils
   `SkipNode`) and `cli.py` (`click` errors). Flag the rest.

2. **User selections use `select.Tree` / `index.Selector`.** If a method lets
   the user pick among options, the argument is named `selection` and is parsed
   with `py4vasp._util.select.Tree.from_selection(...)` so every quantity
   behaves consistently. When those selections index into an array, extraction
   goes through `py4vasp._util.index.Selector`. Section 5 shows where the changed
   files mention `selection` / `select.Tree` / `index.Selector`. Flag a new
   user-facing option arg that is named something else, hand-rolls its own
   parsing, or indexes arrays manually instead of via `Selector`.

3. **Public methods have numpy-style docstrings with runnable examples.** Every
   public method (no leading underscore) needs a numpy-style docstring, and
   user-facing quantity methods carry a `>>>` example that the doctest suite
   executes. Section 3 shows the doctest result for the changed modules — a
   failure or "no tests ran" for a module that gained a public method is a
   finding. A public method with no docstring, or a docstring with no example
   where siblings have one, is a finding the driver can't see — read the diff.

4. **Tests cover the important behavior (>95%).** Section 2 gives each changed
   module's coverage % and the exact uncovered lines. Below ~95% is a finding
   unless the only misses are trivial (e.g. an `else` that raises
   `exception.NotImplemented`). Point findings at the specific uncovered lines
   and the behavior they represent.

Also do the **normal review**: correctness bugs, missing edge cases, wrong
results, resource/logic errors — the same scrutiny as a general code review.

## Step 3 — manual commands (when the driver can't map a test)

Same invocations the driver uses, for a module it couldn't auto-map. Set
`PYTHONPATH` to this checkout's `src` so a worktree measures its own edits, and
use `--cov=py4vasp` (whole package), never a dotted submodule (see Gotchas):

```bash
PYTHONPATH="$PWD/src" .venv/Scripts/python.exe -m pytest tests/calculation/test_symmetry.py --cov=py4vasp --cov-report=term-missing -q
```

```bash
PYTHONPATH="$PWD/src" .venv/Scripts/python.exe -m pytest tests/test_doctest.py -k symmetry -q
```

## Step 4 — report

Call **ReportFindings** with the verified findings, most-severe first (empty
array if the change is clean). Give each a concrete failure scenario and a
`file:line`. Prefer confirmed issues over speculation; the coverage/doctest
sections let you confirm rather than guess.

## Gotchas

- **Use the venv python, not `uv run`, not system python.** `uv run pytest
  --cov` and the venv python *both* crash on numpy under coverage **only** when
  you scope with a dotted module (`--cov=py4vasp._calculation.symmetry`):
  `ImportError: cannot load module more than once per process` (numpy 2 +
  Python 3.14). `--cov=py4vasp` (whole package) is fine — the driver always uses
  that and filters the report to the changed files. System python has no
  `pytest-cov` at all.
- **Worktree measures the wrong src without `PYTHONPATH`.** py4vasp is installed
  editable via a `.pth` pointing at the *main* repo `src`, so inside a
  `.claude/worktrees/<name>` worktree a bare run silently reviews the main repo.
  The driver forces `PYTHONPATH`; do the same in manual runs. Verify with
  `python -c "import py4vasp; print(py4vasp.__file__)"`.
- **Doctests need the project's runner.** They rely on globs injected by
  `tests/test_doctest.py` (`py4vasp`, `path`, `np`), so `pytest --doctest-modules`
  won't work — filter `tests/test_doctest.py` with `-k <module-stem>` instead.
- **The raise scan over-reports.** It is a plain grep; always apply the carve-outs
  in rule 1 before turning a hit into a finding.

## Human path

There is no app to launch — this is a review workflow. A developer reviews by
reading the diff and running the same `pytest --cov` / doctest commands above.
```

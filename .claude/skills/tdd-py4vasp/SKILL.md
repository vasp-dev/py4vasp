---
name: tdd-py4vasp
description: >-
  Test-driven development workflow for py4vasp. Use when adding or changing a
  feature, quantity, class, or method in py4vasp and you want tests written
  first. Triggers: "TDD", "test-driven", "write the tests first", "add a method
  with tests", "implement <quantity>", "red-green-refactor". Splits work into
  one-method chunks, watches each test fail, implements to green, refactors,
  and commits one chunk at a time.
---

# Test-driven development for py4vasp

Build py4vasp features test-first, one small chunk at a time. A **chunk** is
normally **one method of one class** (or one narrow behavior). You write its
tests, watch them fail for the right reason, implement until they pass, refactor
away duplication, and commit — then and only then move to the next chunk.

Tests are run with the wrapper `.claude/skills/tdd-py4vasp/run_tests.py`, which
imports py4vasp from *this* checkout's `src` (see [Gotchas](#gotchas)). All paths
below are relative to the repo/worktree root.

## The loop (follow in order)

### 0. Plan, then STOP for approval
Break the request into an ordered list of chunks, each ≈ one method/behavior.
Record it in the todo list, present it to the user, and **wait for their
sign-off before writing any code.** Do not start chunk 1 until they approve.
Keep chunks small — the project caps a PR at ~200 lines and asks for splits.

Then, for **each** chunk in turn:

### 1. RED — write the test(s) and watch them fail
Write the test(s) for this chunk only, following the conventions below. Run
just those tests and confirm they **fail for the right reason** — a real
`AssertionError` or the missing-method `AttributeError`, *not* a typo, bad
import, or wrong fixture. A test that errors before reaching its assertion is
not a valid RED; fix the test first.

```bash
python .claude/skills/tdd-py4vasp/run_tests.py tests/calculation/test_symmetry.py::test_read -q
```

Run only the tests you expect to fail. If your change may have side effects
elsewhere, add those specific untouched tests to the same run to watch them
stay green — but do **not** run the full suite here.

### 2. GREEN — implement the minimum to pass
Write the simplest implementation that makes the RED tests pass. Re-run the same
selection until green.

```bash
python .claude/skills/tdd-py4vasp/run_tests.py tests/calculation/test_symmetry.py -q
```

### 3. REFACTOR — remove duplication (code AND tests)
With tests green, remove duplication introduced by this chunk **and** against
existing code the new code now overlaps with. Apply the same to the tests: pull
shared setup into fixtures and use `@pytest.mark.parametrize` — this codebase is
fixture-heavy, mirror it. Re-run the chunk's tests to confirm still green.

### 4. COMMIT — one commit per passing chunk
Commit the tests + implementation + refactor together, locally, on the current
branch. Match the repo's message style (`Feat:`, `Fix:`, `Refactor:` prefix):

```bash
git add -A && git commit -m "Feat: add Symmetry.multiplicity"
```

Do **not** push or open a PR — that is a separate, later step. For a multi-line
message use `git commit -F <file>` or a heredoc, never `git commit -m @'...'@`
in the Bash tool (it wraps the message in literal `@`).

Only after the commit lands do you start the next chunk at step 1.

### 5. After ALL chunks — full suite, then report
When every chunk is committed, run the whole suite once. Only report the work as
done to the user if it is green.

```bash
python .claude/skills/tdd-py4vasp/run_tests.py -q
```

## py4vasp test & implementation conventions

Read a sibling test before writing yours — `tests/calculation/test_symmetry.py`
is a good template. Key patterns:

- **Fixtures from `tests/conftest.py`:** `raw_data` (a `RawDataFactory`),
  `Assert` (use `Assert.allclose(actual, desired)` for arrays/dataclasses),
  `format_`, `check_factory_methods`.
- **Construction:** calculation classes are built with
  `SomeClass.from_data(raw_data.<quantity>("Sr2TiO4"))`. Tests attach a
  reference namespace (`obj.ref = types.SimpleNamespace(); obj.ref.raw = ...`)
  and assert against it.
- **Standard method tests:** `read`/`to_dict` (with a
  `test_to_dict_is_alias_of_read`), `print`/`_repr_`, and
  `test_factory_methods(raw_data, check_factory_methods)`.
- **New raw quantity?** You usually must extend **two** places besides the
  calculation class: add a producer under `src/py4vasp/_demo/` and register a
  method on `RawDataFactory` in `tests/conftest.py`, so `raw_data.<quantity>()`
  exists for your test.
- **Optional deps:** guard tests needing extras with
  `pytest.importorskip("spglib")` as existing tests do.

## Gotchas

- **Worktree imports the wrong src.** py4vasp is installed editable via a `.pth`
  file pointing at the *main* repo's `src`. Inside a `.claude/worktrees/<name>`
  worktree, a bare `pytest` silently tests the main repo and ignores your edits.
  Always run tests through `run_tests.py` (it prepends the correct `src` to
  `PYTHONPATH` and prints which one it used) — or set `PYTHONPATH` to the
  worktree `src` yourself. Verify with:
  ```bash
  python -c "import py4vasp; print(py4vasp.__file__)"
  ```
- **Don't run the full suite mid-loop.** It is slow and dilutes the RED signal.
  Full suite runs once, at the end (step 5).
- **A weak RED is a bug.** If the test passes on the first run, or errors before
  its assertion, it isn't testing what you think — fix the test before implementing.

## Human path

There is no app to launch — this is a workflow skill. Developers run the same
tests directly with the project's tooling:

```bash
uv run --active pytest tests/calculation/test_symmetry.py
```

Use `run_tests.py` (or set `PYTHONPATH`) instead when working inside a worktree,
so the tests exercise your edits rather than the main checkout.

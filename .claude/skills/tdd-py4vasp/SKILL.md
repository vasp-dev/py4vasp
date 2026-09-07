---
name: tdd-py4vasp
description: >-
  Carry out ONE chunk of a py4vasp change test-first: RED (watch the test fail
  for the right reason) → GREEN → refactor → one local commit. Use it for each
  chunk of an approved chunk list, and for any change small enough to be a
  single chunk on its own — one method, one narrow behavior, a focused bug fix.
  If the work turns out bigger than one method or behavior, or there is no chunk
  list yet, it stops and chunks the work with the plan-py4vasp skill first
  rather than coding ahead. Triggers: "TDD", "test-driven", "write the tests
  first", "red-green-refactor", "next chunk", "chunk N", "add a method to
  <class>", "fix <bug>".
---

# Test-driven development for py4vasp — one chunk

Carry **one chunk** from failing test to commit. A **chunk** is normally **one
method of one class** (or one narrow behavior): you write its tests, watch them
fail for the right reason, implement until they pass, refactor away
duplication, and commit. Then you stop — the next chunk is a fresh pass.

Tests are run with the wrapper `.claude/skills/tdd-py4vasp/run_tests.py`, which
imports py4vasp from *this* checkout's `src` (see [Gotchas](#gotchas)). All paths
below are relative to the repo/worktree root.

## 0. Scope check — what is this chunk? (do this first)

Decide which of three situations you are in **before writing anything**:

- **An approved chunk list exists.** Take the next unstarted chunk and only
  that one. Its test names are already written down; honor them.
- **The request is itself one chunk** — one method, one narrow behavior, a
  focused bug fix with a single reproducing test. Treat the request as chunk 1
  of 1 and continue to step 1.
- **The work is bigger than one chunk** — several methods, a new quantity
  end-to-end, more than one class, error handling with real logic of its own,
  or you cannot name the chunk's test in one line. **Stop. Do not start
  coding.** Chunk the work with the **plan-py4vasp** skill, get the chunk list
  signed off, then come back here for chunk 1.

That last branch is the fallback that matters: a chunk you cannot describe in
one sentence is a plan you have not written yet. Reaching for it is cheap;
discovering it four files into an implementation is not.

Take it **even when you could plausibly implement the whole thing in one
pass** — the value of the chunk list is that the user sees the shape of the
work while changing it is still free. If you genuinely cannot pause for
sign-off (a non-interactive or delegated run), you still produce the chunk list
first and state it up front, then work it one chunk at a time with a commit
each — never a single sweeping commit.

Then, for the chunk you settled on:

## 1. RED — write the test(s) and watch them fail

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

## 2. GREEN — implement the minimum to pass

Write the simplest implementation that makes the RED tests pass. Re-run the same
selection until green. Resist implementing the *next* chunk's behavior because
it is obvious and nearby — it will arrive without a failing test to justify it.

```bash
python .claude/skills/tdd-py4vasp/run_tests.py tests/calculation/test_symmetry.py -q
```

## 3. REFACTOR — remove duplication (code AND tests)

With tests green, remove duplication introduced by this chunk **and** against
existing code the new code now overlaps with. Apply the same to the tests: pull
shared setup into fixtures and use `@pytest.mark.parametrize` — this codebase is
fixture-heavy, mirror it. Re-run the chunk's tests to confirm still green.

## 4. COMMIT — one commit for this chunk

Commit the tests + implementation + refactor together, locally, on the current
branch. Match the repo's message style (`Feat:`, `Fix:`, `Refactor:` prefix):

```bash
git add -A && git commit -m "Feat: add Symmetry.multiplicity"
```

Do **not** push or open a PR — that is a separate, later step
(`push-py4vasp`). For a multi-line message use `git commit -F <file>` or a
heredoc, never `git commit -m @'...'@` in the Bash tool (it wraps the message
in literal `@`).

## 5. Stop, and report where the chunk list stands

After the commit lands, say which chunk is done and what the next one is. Start
the next chunk as a fresh pass through this skill — do not roll straight on.

**Only after the last chunk of the list** run the whole suite once, and only
report the work as done to the user if it is green:

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
- **Public methods** get a numpy-style docstring with a **runnable** doctest —
  `tests/test_doctest.py` executes examples from `_calculation` with `path` and
  `py4vasp` injected as globals, so the example must build its own data.
- **Exceptions** come from `py4vasp.exception`; user selections go through
  `select.Tree` / `index.Selector`.
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
  Full suite runs once, after the final chunk (step 5).
- **A weak RED is a bug.** If the test passes on the first run, or errors before
  its assertion, it isn't testing what you think — fix the test before implementing.
- **Scope creep is the common failure.** Finishing two chunks in one commit
  looks efficient and costs you the ability to revert either. One chunk, one
  commit.

## Human path

There is no app to launch — this is a workflow skill. Developers run the same
tests directly with the project's tooling:

```bash
uv run --active pytest tests/calculation/test_symmetry.py
```

Use `run_tests.py` (or set `PYTHONPATH`) instead when working inside a worktree,
so the tests exercise your edits rather than the main checkout.

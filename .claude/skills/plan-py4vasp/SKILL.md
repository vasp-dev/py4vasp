---
name: plan-py4vasp
description: >-
  Plan a py4vasp change as an ordered list of test-first chunks — that chunk
  list *is* the plan. Use it BEFORE writing any implementation plan for
  py4vasp: a new feature, quantity, class or method, a refactor, or a bug fix.
  Load it as soon as a py4vasp code change is under discussion — before
  answering, not after — including when the user only asks design questions
  ("are there other things to consider?", "how would you implement X?", "what
  would it take to add X?"). Answering those IS the first half of planning, and
  the answer has to close with the chunk list or with what must be settled
  before the work can be chunked. Correct to use in plan mode:
  it only reads code and writes the plan, it never edits source and never
  commits. Each chunk is one method or behavior with its tests named up front;
  carrying out a chunk is then handed to the tdd-py4vasp skill, one chunk at a
  time.
---

# Planning a py4vasp change

A py4vasp plan is **an ordered list of chunks** — not a description of the
finished code with the tests bolted on at the end. A **chunk** is normally
**one method of one class** (or one narrow behavior): small enough that its
tests, its implementation and its commit stay reviewable together.

This skill produces that list and **stops**. Each chunk is then carried out by
the **tdd-py4vasp** skill (RED → GREEN → refactor → commit), one at a time.

All paths below are relative to the repo/worktree root.

## 1. Read before you plan

Never plan from the request alone. Establish four things first:

- **Does it exist already?** Grep for the method/quantity name across
  `src/py4vasp` and `tests`. Partial implementations and near-duplicates are
  common and change the chunking completely.
- **The sibling to copy.** Find the closest existing quantity or method and
  read *both* its implementation and its test.
  `src/py4vasp/_calculation/symmetry.py` + `tests/calculation/test_symmetry.py`
  is a good default pair.
- **The seams.** How does data actually reach the class — `raw.access`,
  a `FileSource`, a dispatcher, `@quantity` injection? Name the exact file and
  line the new code hangs off.
- **What the request leaves open.** Collect the genuine design questions and
  put your recommended default next to each, so the user can answer with
  "defaults, except 3".

## Questions are half of planning

A request phrased as a question — "are there other things to consider?", "what
would it take?", "query me on the open questions" — is still a planning turn.
Answer the questions, with your recommended default beside each, and then
**close with the chunking anyway**:

- If the open questions do not change the shape of the work, give the chunk
  list outright and mark which chunks the answers would affect.
- If they *do* change the shape (they decide whether a helper module exists at
  all, say), give the chunk list for the part that is already settled and name
  the specific answers you need before the rest can be chunked.

What you must not do is answer the questions and stop there. That is how a
change ends up planned implementation-first later, in a turn where nobody
remembers to chunk it.

## 2. Cut the work into chunks

- One method or one narrow behavior each. If a chunk needs the word "and" to
  describe it, split it.
- Each chunk must be **independently committable and revertible** — green
  suite at every chunk boundary.
- **Order pure logic before wiring.** Chunks that touch only a helper module
  are testable without any public API change; put them first so feedback is
  fast and early commits are safe.
- Error and edge-case behavior is its own chunk when the errors carry real
  logic (ambiguity, validation, rejection), not an afterthought inside another
  chunk.
- The docstring + runnable doctest is part of the chunk that adds the public
  method, never a separate "docs at the end" chunk.
- Aim for 3–7 chunks. More than that and the request probably needs splitting
  into two rounds; fewer than two and you likely do not need a plan at all —
  go straight to **tdd-py4vasp**.

## 3. Write each chunk in this shape

Numbered, and for every chunk state all four:

```
N. <behavior, one line>
   Test:  tests/<path>::<test name(s)>  — what the assertion actually checks
   Code:  src/py4vasp/<path>  — the method/class added or changed
   Also:  any second place that must change for the test to be writable
```

That `Also:` line is where py4vasp plans usually go wrong — see the facts
below. If a chunk has no `Test:` line you have not finished planning it.

## 4. The shape to avoid

Do **not** produce a plan that describes the implementation in full and then
carries a trailing `## Tests` section listing the test files to add. That is an
implement-then-test plan; it reads fine and it silently discards TDD, because
by the time anyone reaches the test section the code is already written. The
tests belong *inside* each chunk, named before its implementation.

Equally: no chunk whose description is "write the tests" or "add test
coverage". Coverage is a property of every chunk, not a phase.

## 5. Present the plan, then stop

Put the chunk list in the todo list, present it, and **wait for sign-off before
any code is written.**

If **plan mode** is active, this chunk list *is* the plan you hand to
`ExitPlanMode` — ordered chunks with tests named, plus the design decisions and
open questions. Approval of the plan is the sign-off. Do not describe the
tdd-py4vasp loop as a final step of the plan; it is how each chunk gets done.

## 6. Hand the chunks over

Once approved, execute **one chunk at a time** with the **tdd-py4vasp** skill,
starting at chunk 1 and returning to it for each subsequent chunk. Do not batch
several chunks into one pass, and do not run the full suite between chunks —
that happens once, after the last chunk lands.

## Planning-time py4vasp facts that change the plan

These are the ones that alter the chunk list, so check them while planning
rather than discovering them mid-implementation:

- **A new raw quantity costs two extra places.** Besides the calculation class
  you must add a producer under `src/py4vasp/_demo/` and register a method on
  `RawDataFactory` in `tests/conftest.py`, or `raw_data.<quantity>()` does not
  exist and the chunk's test cannot even be written. That belongs on the
  chunk's `Also:` line.
- **Construction and fixtures.** Calculation classes are built with
  `SomeClass.from_data(raw_data.<quantity>("Sr2TiO4"))`. `tests/conftest.py`
  supplies `raw_data`, `Assert` (`Assert.allclose` for arrays/dataclasses),
  `format_` and `check_factory_methods`.
- **The standard method set.** A new quantity is expected to bring `read`/
  `to_dict` (plus `test_to_dict_is_alias_of_read`), `print`/`_repr_`, and
  `test_factory_methods(raw_data, check_factory_methods)`. Plan a chunk for
  each rather than one "add the quantity" chunk.
- **Public methods need a numpy-style docstring with a runnable doctest.**
  `tests/test_doctest.py` collects and *executes* docstring examples from
  `_calculation`, injecting `path` and `py4vasp` as globals — so the example
  must be able to build its own data. A doctest that cannot run is a failing
  test, not a comment.
- **Exceptions come from `py4vasp.exception`**, and user-facing selections go
  through `select.Tree` / `index.Selector`. Decide which exception type each
  error chunk raises while planning; `review-py4vasp` enforces this later.
- **Optional dependencies** get guarded with `pytest.importorskip("spglib")`
  in the test, as existing tests do. Note it on the chunk.
- **Docs.** New public methods usually need adding to the relevant file under
  `docs/`; fold it into the chunk that introduces the method.

## Out of scope for this skill

No edits to `src/` or `tests/`, no commits, no running the test suite. The
verification commands belong *in* the plan as text, to be run by tdd-py4vasp
when the chunk is executed. If you find yourself wanting to write code to
answer a design question, write a throwaway probe in the scratchpad instead and
say so in the plan.

## Human path

There is no app to launch — this is a workflow skill. The plan is a message (or
a plan file); nothing is installed or run to produce it.

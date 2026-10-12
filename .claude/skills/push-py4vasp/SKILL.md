---
name: push-py4vasp
description: >-
  Push py4vasp changes to origin and open a PR. Use when asked to push, publish
  the branch, ship the changes, or open a pull request for py4vasp. Reviews code
  changes first (dispatches /review-py4vasp to a subagent unless already done),
  checks the backlog, applies black + isort, pushes a fresh descriptive branch
  to origin, and opens the PR with gh (or hands back a compare link and the PR
  message where gh is missing).
---

# Push py4vasp to origin

Orchestrates the last mile: review → backlog → format → push → PR. The
deterministic prep (classify changes, decide if review is needed, list backlog
changes, suggest a branch name, check for `gh`, draft the PR message) is done by
the read-only driver
`.claude/skills/push-py4vasp/push_prep.py`; the side-effecting steps (dispatch
review, apply fixes, format, push) are orchestrated here. `origin` is the **main**
repo `vasp-dev/py4vasp`, so pushing publishes to upstream. Paths are relative to
the repo/worktree root.

## Step 0 — prep (run the driver)

```bash
python .claude/skills/push-py4vasp/push_prep.py
```

It prints: changed files split into **CODE** (`*.py` under `src/` or `tests/`) vs
non-code (docs/CI/tooling); whether a review is required; whether a **user
simulation** is required (and why); the backlog files the branch adds or deletes;
a suggested branch name; whether `gh` can open the PR (else the compare URL); and
a draft PR message. Read this first — it drives every decision below.

## Step 1 — review gate (only if CODE changed)

If the driver reports **no** code changes (only docs/CI/tooling), skip to Step 2 —
a docs change still faces the user simulation gate.

If code changed, a py4vasp review must exist for this diff:
- If you already ran `/review-py4vasp` on this diff earlier in the session and
  its findings were addressed, skip re-running it.
- If you **cannot tell** whether a review was done, ask the user before
  dispatching.
- Otherwise dispatch a subagent to review: use the **Agent** tool
  (`subagent_type: general-purpose`) and instruct it to follow the
  `/review-py4vasp` skill on the current diff and return the findings.

Then **address the findings** — fix the code, or record why a finding is a
non-issue. Re-run the affected tests (see `/review-py4vasp` for the invocation).

## Step 2 — user simulation gate (only if the user interface changed)

If the driver reports `USER SIMULATION REQUIRED: NO`, skip to Step 3.

Otherwise somebody who may read only the documentation must have tried this diff.
The internal gates cannot see a feature that works perfectly and cannot be found,
or one whose defaults quietly produce the wrong physics:

- If you already ran `/simulate-user-py4vasp` for this diff earlier in the session
  and its blockers were addressed, skip re-running it.
- If you **cannot tell** whether a simulation was done, ask the user before
  dispatching.
- Otherwise follow the `/simulate-user-py4vasp` skill — it writes the premise and
  dispatches the subagent itself. Do not hand-roll the dispatch, and **never run it
  from plan mode** (the subagent inherits plan mode and cannot write an input file,
  which silently removes most of the surface a user touches).

Then act on the report: blockers and documentation gaps are fixed before the push;
a documentation gap larger than this change is recorded in `backlog/` as a short
markdown file; annoyances go into the reviewer comment (Step 7) so the human
reviewer inherits them.

## Step 3 — backlog check

Bring the backlog in line with the branch before it is published:

- **Resolved items are gone.** If the work came from a file under `backlog/`, the
  branch deletes it, or trims it to the points still open. Check the driver's
  backlog section and fetch `origin/master`: a change merged meanwhile may have
  resolved more of the item. Skim the other `backlog/` files too — this change
  may have resolved one nobody named.
- **Follow-ups are recorded.** Decide whether the work left anything for later:
  a limitation you chose not to lift, a review finding deferred as out of scope,
  a missing VASP output that forced a workaround. Each one becomes a short
  markdown file in `backlog/`, written so someone without this session's context
  can pick it up. Do not record what is already fixed or trivially obvious.

Commit the backlog changes, and list them to the user at the confirm step.

## Step 4 — confirm (hard stop before publishing)

Show the user the PR title and message (Step 7) and the backlog changes, then
STOP and ask for confirmation before formatting, pushing and opening the PR.
Pushing publishes the branch to `origin` (`vasp-dev/py4vasp`) — do not proceed
without an explicit go-ahead.

## Step 5 — format

Apply the project's formatters to `src` and `tests` (no-op if pre-commit already
ran):

```bash
uv run --active black src tests
```

```bash
uv run --active isort src tests
```

Then look at the reformatted diff: black sometimes introduces awkward line breaks
(a long call or expression wrapped across several lines). Where that hurts
readability, extract a well-named helper variable so the line fits naturally, and
re-run black to confirm it stays put.

If either step changed files, commit that:

```bash
git commit -am "Apply black and isort"
```

## Step 6 — branch + push

Create a **fresh descriptive** branch (start from the driver's suggestion, refine
if it misses the point), then push it and set upstream:

```bash
git switch -c feat/<descriptive-name>
```

```bash
git push -u origin feat/<descriptive-name>
```

## Step 7 — open the PR

The PR is squash-merged, so **the PR title and body become the commit message on
`master`**. Write them for someone reading the history a year from now, starting
from the driver's draft:

- Title: `Feat:` / `Fix:` / `Refactor:` / `Docs:` and one line saying what changed.
- `## Summary`: what the change does and why, the new or changed public API (a
  short usage example helps), behavior that changed for existing users, and the
  backlog files added or removed. `Fixes #123` if there is an issue.
- Leave out how it was verified (tests run, review and simulation outcomes,
  coverage, pre-existing failures) and notes for the reviewer. They describe the
  PR, not the code, and are noise in the history.

If `gh` is available (the driver says so), write the body to a file in the
scratchpad and open the PR, then give the user its URL:

```bash
gh pr create --repo vasp-dev/py4vasp --base master --head <branch> --title "<title>" --body-file <file>
```

Anything the reviewer should know — verification, known annoyances, deliberate
deviations from a rule — goes into a PR comment instead, so it never reaches the
commit:

```bash
gh pr comment <number> --repo vasp-dev/py4vasp --body-file <file>
```

If `gh` is missing or not logged in, hand the user the compare URL from the
driver (swap the branch segment if you renamed the branch) and the PR message in
a copy-paste block, with the reviewer notes in a second block for them to post
as a comment.

## Gotchas

- **`origin` is upstream, not a fork.** `git@github.com:vasp-dev/py4vasp.git`.
  Pushing goes straight to the main repo (the user is a maintainer). A fork
  remote `private` (`martin-schlipf/py4vasp`) also exists — only use it if the
  user asks.
- **`gh` may be missing on another developer's machine.** The driver checks
  `gh auth status`; only then use `gh pr create`, otherwise fall back to the
  compare URL it builds from `remote.origin.url`.
- **Non-code = review-skipped, but not necessarily gate-free.** A CI/tooling
  change skips Steps 1–2 but still gets the backlog check. A **docs** change skips the code
  review yet still requires the user simulation — the documentation is precisely
  what that gate reads.
- **The driver is read-only.** `push_prep.py` never pushes, commits, or edits;
  re-run it freely. The only publishing actions are `git push` in Step 6 and
  `gh pr create` / `gh pr comment` in Step 7.
- **Worktree note:** run inside the worktree you actually changed; the driver
  reports the checkout path so you can confirm.

## Human path

A developer does the same by hand: review the diff, tidy `backlog/`, `uv run
--active black src tests && uv run --active isort src tests`, `git push -u origin
<branch>`, then `gh pr create` or open the compare URL in a browser.

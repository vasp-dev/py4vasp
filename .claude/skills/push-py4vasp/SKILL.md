---
name: push-py4vasp
description: >-
  Push py4vasp changes to origin and open a PR. Use when asked to push, publish
  the branch, ship the changes, or open a pull request for py4vasp. Reviews code
  changes first (dispatches /review-py4vasp to a subagent unless already done),
  applies black + isort, pushes a fresh descriptive branch to origin, and hands
  back a PR compare link plus a draft PR message.
---

# Push py4vasp to origin

Orchestrates the last mile: review → format → push → PR link. The deterministic
prep (classify changes, decide if review is needed, suggest a branch name, build
the PR URL and draft message) is done by the read-only driver
`.claude/skills/push-py4vasp/push_prep.py`; the side-effecting steps (dispatch
review, apply fixes, format, push) are orchestrated here. `origin` is the **main**
repo `vasp-dev/py4vasp`, so pushing publishes to upstream. Paths are relative to
the repo/worktree root.

## Step 0 — prep (run the driver)

```bash
python .claude/skills/push-py4vasp/push_prep.py
```

It prints: changed files split into **CODE** (`*.py` under `src/` or `tests/`) vs
non-code (docs/CI/tooling); whether a review is required; a suggested branch
name; the PR compare URL; and a draft PR message. Read this first — it drives
every decision below.

## Step 1 — review gate (only if CODE changed)

If the driver reports **no** code changes (only docs/CI/tooling), skip to Step 3.

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

## Step 2 — confirm (hard stop before publishing)

STOP and ask the user to confirm before formatting and pushing. Pushing
publishes the branch to `origin` (`vasp-dev/py4vasp`) — do not proceed without
an explicit go-ahead.

## Step 3 — format

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

## Step 4 — branch + push

Create a **fresh descriptive** branch (start from the driver's suggestion, refine
if it misses the point), then push it and set upstream:

```bash
git switch -c feat/<descriptive-name>
```

```bash
git push -u origin feat/<descriptive-name>
```

## Step 5 — PR link + message

Give the user the compare URL and the draft message from Step 0 (the driver
already built both, using the real branch name if you kept its suggestion — else
swap the branch segment). The URL opens GitHub's "Open a pull request" page:

```
https://github.com/vasp-dev/py4vasp/compare/master...<branch>?expand=1
```

Present the draft PR message in a copy-paste block.

## Gotchas

- **`origin` is upstream, not a fork.** `git@github.com:vasp-dev/py4vasp.git`.
  Pushing goes straight to the main repo (the user is a maintainer). A fork
  remote `private` (`martin-schlipf/py4vasp`) also exists — only use it if the
  user asks.
- **`gh` is not installed** here (only `glab`). Do not reach for `gh pr create`;
  hand over the constructed compare URL instead — the driver builds it from
  `remote.origin.url`.
- **Non-code = review-skipped, but still formatted + pushed.** A docs/CI/tooling
  change goes straight from Step 0 to Step 3.
- **The driver is read-only.** `push_prep.py` never pushes, commits, or edits;
  re-run it freely. The only publishing action is `git push` in Step 4.
- **Worktree note:** run inside the worktree you actually changed; the driver
  reports the checkout path so you can confirm.

## Human path

A developer does the same by hand: review the diff, `uv run --active black src
tests && uv run --active isort src tests`, `git push -u origin <branch>`, then
open the compare URL in a browser.

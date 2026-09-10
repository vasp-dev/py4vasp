---
name: simulate-user-py4vasp
description: >-
  Validate a py4vasp change from the outside by dispatching a subagent to
  role-play a user who may read only the documentation, never the source. Use
  before pushing or opening a MR for anything that touches the user interface —
  a public method of a quantity class, a CLI command, or the docs — and whenever
  asked to "simulate a user", "test this like a user would", or "check the
  usability" of a py4vasp feature. Returns ranked findings, a verdict on whether
  a documentation-only user can succeed, and the agent's confessions.
---

# Simulating a py4vasp user

`plan-py4vasp`, `tdd-py4vasp` and `review-py4vasp` all judge a change from the
**inside**: code, tests, coverage, house rules. A change can pass every one of
them and still be unusable, because none of them asks the only question a user
asks — *can I work out how to do this from the documentation?*

This skill answers that by dispatching a subagent that role-plays a user with
**no access to the source**. It is a validation gate, not an exploration: you
write the premise, dispatch it, verify it stayed inside the rules, and act on
what it reports. All paths are relative to the repo/worktree root.

## When it is required

Whenever the diff touches the **user interface**:

- a public method (no leading underscore) added to or changed on a quantity class
  under `src/py4vasp/_calculation/`
- any change to `src/py4vasp/cli.py`
- any change under `docs/`

A change confined to private helpers, the raw schema, the demo data or the tests
does not need it. `push-py4vasp` treats this as a blocking gate and its driver
prints `USER SIMULATION REQUIRED: YES/NO`, so the decision is a fact rather than
a judgement.

## Step 0 — do NOT run this from plan mode

**Check first.** A subagent inherits plan mode from the session that dispatches
it, and a simulated user in plan mode cannot write a POSCAR, cannot use
`-o/--output`, and cannot read an input file from disk — i.e. it cannot exercise
the surface a real user touches most. In the trial run that produced this skill
the agent tried to work around the restriction with a process substitution, which
is not something any user would do, and the resulting traceback leaked source
code to it.

If plan mode is on, say so and stop. Run the simulation from normal mode.

## Step 1 — write the premise (without naming the API)

The agent must *discover* the feature, because discoverability is the thing under
test. Give it exactly what a colleague would have said in one sentence — the
capability, never the command or method names:

> A colleague mentioned that the latest py4vasp can **create KPOINTS files for
> you** — both for a band structure along the high-symmetry path and for a regular
> k-mesh — and that it works from the command line as well as from Python.

Derive that sentence from the MR description, not from the diff. If you catch
yourself writing `generate_kpath`, delete it.

## Step 2 — dispatch

Use the **Agent** tool with `subagent_type: general-purpose` and the default
model. Never `fork`: a fork inherits your context, which contains the
implementation, and the whole exercise collapses. Capability is not what makes a
simulation unrealistic — *access* is — so do not reach for a weaker model either;
a capable agent constrained to the documentation writes a far more useful report.

The prompt must contain all six blocks below. Copy them; the wording matters.

**1. The persona.** A computational materials scientist who runs VASP, comfortable
with the shell and Python, has never seen py4vasp's source and never will. Then
the premise from Step 1.

**2. The isolation contract.** Allowed, because it is what a real user has:
`py4vasp --help` and every subcommand's `--help`; `help(obj)` and `obj?` at the
Python prompt (docstrings *are* user-facing documentation); the sources under
`docs/`; `README.md`. Forbidden: any file under `src/` or `tests/`; git history,
diffs, commit messages, branch names; `.claude/`; any plan, issue or notes file.
And the clause that matters, because this is how leaks actually happen: *if a tool
result shows you source code — a traceback will — stop reading it and write that
down in your report.*

**3. Behave like a person, not an agent.** A handful of actions in a natural
order. No enumerating the package, no reverse-engineering intent from the
repository. **When the documentation does not answer a question, that is a finding
to report — not a puzzle to solve by digging.**

**4. The environment.** py4vasp is usually a development checkout here, so spell
out the invocation and tell the agent to treat it as the real command:

```bash
export PYTHONPATH=<checkout>/src
PY=<path to the venv>/bin/python
$PY -m py4vasp --help          # the command line interface
$PY -c "import py4vasp; ..."   # the Python interface
```

Give it a working directory of its own (`/tmp/.../user-trial-<feature>`),
**explicitly grant it permission to write files there**, and forbid modifying
anything in the checkout.

**5. The task.** Work out from the documentation how to use the feature and note
what had to be guessed; write your own input file and use the feature on it; do
whatever you would normally do to convince yourself the result is right, and say
whether you *could*; make one or two mistakes a real user plausibly makes and see
whether the message tells you what to fix; try both interfaces if both exist.

**6. The deliverable.** Five sections, in this order:

- **A. Narrative** — what you did, in order, with the exact commands, including
  the dead ends.
- **B. Findings** — each with the command, what you expected, what happened, and
  what would have helped; ranked *would have stopped me* versus *annoying*;
  documentation gaps are first-class findings; and for each, **would an ordinary
  user have noticed this at all?**
- **C. Verdict** — can a user who only reads the documentation succeed? yes /
  partly / no, and why.
- **D. Documentation trail** — every source consulted, in order, and whether it
  helped.
- **E. Confessions** — anything you did that a real user could not; any rule you
  broke; **any false alarm you raised and then retracted**; and every moment you
  wanted to read the source, and what question drove the urge.

Close with: *do not soften the report to be agreeable; blunt is useful.*

Sections B-"ordinary user", D and E are what turn a report into evidence. The
trial run's most valuable line was the agent volunteering that an ordinary user
would *not* have caught its own worst finding, because the only symptom was a
space-group label nobody checks by eye. D localises the documentation gap to the
exact channel. E is where the leaked traceback got reported.

## Step 3 — verify the isolation held

Trust the agent, then check. The Agent tool returns an `output_file` — the full
JSONL transcript. **Do not read it**; it runs to hundreds of KB. Grep the tool
*inputs* only:

```bash
grep -o -E '"name":"(Bash|Read|Grep|Glob)","input":\{"[a-z_]+":"[^"]{0,120}' "$TRANSCRIPT" \
  | grep -o -E '(src/py4vasp/[A-Za-z_./]+|tests/[A-Za-z_./]+|git (log|diff|show|blame))' \
  | sort | uniq -c
```

Two traps, both of which caught me while building this:

- The pattern must include the `py4vasp/` segment. A bare `src` also matches the
  `PYTHONPATH` you supplied yourself, which is legitimate and appears in every
  command the agent runs.
- A source path in a tool **result** is a leak, not a violation. Only inputs
  count. `docs/` hits are expected and fine.

A non-empty result invalidates the run: the findings may come from reading the
implementation rather than the documentation. Re-dispatch a fresh agent.

## Step 4 — act on the findings

Present the report to the user and classify with them:

- **Blockers** — anything that stopped the simulated user, and anything that
  produced a silently wrong result. These are fixed before the MR.
- **Documentation gaps** — fixed before the MR, or, if the gap is bigger than the
  change (a missing reference page for a whole interface, say), recorded in
  `backlog/` as a short markdown file.
- **Annoyances** — fixed if cheap, otherwise recorded in the PR message so the
  human reviewer inherits them rather than rediscovering them.

Never fix a finding by editing only the report's wording. If the answer is "the
user should have known", the documentation is the thing that failed.

## Gotchas

- **Plan mode cripples it.** See Step 0. This is the single most likely way to
  waste a run.
- **`fork` defeats it.** The fork knows the implementation. Always
  `general-purpose`.
- **Naming the API in the premise** removes the discoverability test, which is
  usually where the real findings are.
- **A stale installed py4vasp hides the feature.** If the environment has an old
  release installed, the agent tests *that* and reports the feature as missing.
  Always pass an explicit `PYTHONPATH` and, if in doubt, have it print
  `py4vasp.__file__` first.
- **Docstrings count as documentation.** Do not forbid `help()` — it is exactly
  what a notebook user reads. What is forbidden is opening the file that contains
  the docstring.
- **`dir()` / tab-completion is a grey zone.** The trial agent used it to find a
  method the docs never mention, and flagged that as borderline itself. Allow it,
  and treat "I only found this by tab-completion" as a documentation finding.
- **Budget.** Expect roughly 25-30 tool calls, ~80 k tokens and a few minutes for
  a two-command feature. Cheap next to the MR it validates.

## Human path

There is no app to launch — this is a workflow skill. The human equivalent is
handing the branch to a colleague who has not read the code, with the MR summary
and nothing else, and asking them to use the feature and say where they got
stuck.

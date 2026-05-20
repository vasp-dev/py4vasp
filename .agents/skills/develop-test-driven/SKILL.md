---
name: develop-test-driven
description: Enforce real red-green-refactor implementation instead of code-first changes. Use this whenever a task will change executable behavior and the repo has a practical automated test seam, whether directly from the user or as a delegated slice inside `execute-plan`.
compatibility: opencode
---

# Develop test driven

Use this skill when implementation should be led by tests instead of by coding first and adding coverage afterward. The goal is to change behavior with evidence, not to write code that merely seems right.

This skill is for active implementation work. It is most useful for features, bug fixes, and refactors where the correct behavior can be expressed in automated tests. It works well as a focused sub-skill inside `execute-plan`: `execute-plan` owns sequencing, delegation, work card creation, and integrated verification handoff; this skill owns the red-green-refactor loop for the implementation slice.

When the slice is mainly about an interface boundary such as web UI, CLI, TUI, API contracts, emails, or notifications, pair this skill with `implement-interface` rather than forcing TDD to also carry all of the interface-fidelity guidance.

If you were invoked from `execute-plan`, treat the feature card and current task as your boundary. Implement only that slice, return evidence, and let `execute-plan` decide broader sequencing and downstream handoffs.

## Owns

- One behavior-changing implementation slice at a time
- The red-green-refactor loop for that slice
- Producing concrete evidence: the failing test or automated check, the minimal code change, and the passing verification

## Does Not Own

- moving feature cards between lanes
- deciding overall task sequencing across a full feature plan
- product discovery, feature design, or implementation planning
- broad final review or documentation closure for the feature

## When To Use

Use this skill when the user or coordinating skill:

- asks for a feature or bug fix and the repo has a test harness
- wants true TDD or says to write tests first
- needs a regression test for a reported failure before changing code
- is working in an area where behavior drift would be expensive
- is executing a task from `execute-plan` that changes executable behavior

Do not use this skill when:

- the change is purely docs, comments, planning artifacts, or static configuration with no meaningful behavior to test
- the repo genuinely has no practical automated test seam for the requested change
- the task is still discovery or planning and implementation should not start yet
- the main work is lane movement or feature-card maintenance rather than behavior change

If there is no good automated seam yet, first find the smallest useful seam. Do not use that as an excuse to skip testing entirely.

## Core Rule

Do not write or keep production code for a new behavior change until a test or other automated check demonstrates the missing behavior first.

That means:

- write the test before the implementation change
- run it and confirm it fails for the expected reason
- make the smallest implementation change that can make it pass
- rerun the focused check and confirm the behavior changed
- refactor only after the tests are green

If code was already written for the requested behavior before the failing test exists, do not quietly continue from there. Either delete the speculative code or ignore it and re-derive the implementation from the test.

Speed is not an exception. When the user wants a quick fix, make the cycle smaller and more targeted, not looser.

## Workflow

### 1. Start From The Intended Behavior And The Current Task Boundary

Before touching code, state what behavior is changing.

Capture:

- the user-visible or externally observable behavior
- the seam where that behavior can be tested
- the smallest first test that would prove the behavior is currently missing
- any task constraints supplied by `execute-plan` or the feature card

Prefer a test that expresses the desired behavior through the real public interface. Default away from mocks when real interactions are practical. A mock-heavy test that only proves the mock was called is weaker than a small real test at the correct boundary.

If a coordinating skill supplied task-specific verification, satisfy that verification without expanding into adjacent plan work.

### 2. Write The Smallest Failing Test

Add one focused test for one behavior.

Good tests usually:

- name the behavior clearly
- assert one meaningful outcome
- use real code paths when practical
- stay small enough that the reason for failure is obvious

Avoid combining multiple behaviors into one broad test unless the repo's existing patterns clearly prefer that shape.

### 3. Verify Red Before Implementing

Run the most targeted command that proves the new test fails.

Confirm all of these:

- the test actually runs
- it fails, not passes
- it fails for the expected reason
- the failure points to missing or incorrect behavior, not a typo or broken test setup

If the test passes immediately, you have not proven the change. Tighten the test or choose a better seam.

If the test fails for the wrong reason, fix the test first. Do not start implementing against a broken signal.

### 4. Make The Smallest Green Change

Only after the failing signal is real, change the implementation.

Aim for the minimum code needed to satisfy the current test. Prefer a direct, boring change over early abstraction. Do not bundle unrelated cleanup, extra options, adjacent feature work, or feature-card edits into the same step.

When fixing a bug, the regression test is the first deliverable. The code change is second.

### 5. Verify Green Immediately

Rerun the focused test or check that failed in the red step.

Then confirm nearby regression safety with whatever additional verification fits the repo and change risk, such as:

- the relevant test file or package
- related integration tests
- lint, typecheck, or build when they are meaningful safeguards

Do not claim success because the code looks correct. Claim success when the targeted failing signal turned green and the surrounding checks still hold.

### 6. Refactor Without Changing Behavior

Once green, improve the code only if the cleanup helps and the tests still guard the behavior.

Good refactors here include:

- removing duplication introduced during the green step
- clarifying names or structure
- extracting a helper that the passing tests now protect

Do not sneak in extra behavior during refactor. If behavior must change again, start another red-green cycle.

### 7. Return Evidence, Then Repeat If Needed

For multi-part work, repeat with the next smallest missing behavior instead of writing a large block of implementation and backfilling tests later.

Each cycle should leave behind:

- one clearer expression of expected behavior
- one passing proof that the behavior now exists
- no ambiguity about what changed

If you are working under `execute-plan`, report back in a way the coordinator can use directly:

- the behavior covered
- the new or updated test
- the command that proved red
- the minimal implementation change for green
- the command that proved green
- any remaining risk or next likely cycle

Do not move the feature card between lanes from this skill unless the coordinating instructions explicitly made that the current task, which is unusual.

## TDD Heuristics

Prefer:

- user-facing or contract-facing tests over implementation-detail tests
- focused commands that validate one new behavior quickly
- minimal implementation before abstraction
- regression tests for every confirmed bug fix
- small repeated cycles over one giant test file rewrite

Flag:

- implementation code appearing before any new failing test
- a new test that passes on the first run without proving anything
- tests that mostly assert mocks, stubs, or internals instead of behavior
- broad code changes justified by one narrow failing test
- claims that manual checking is enough for a behavior that can be automated

## When The Test Seam Is Hard To Reach

Treat a hard-to-test change as design feedback.

First ask:

- is there a smaller public seam to test?
- can the behavior be moved behind a simpler interface?
- can an integration test cover this more honestly than deep mocking?

If the only path forward is limited, be explicit about the compromise and still preserve the red-green order. A weaker but honest automated check is usually better than code-first work with no durable proof at all.

If you are operating under `execute-plan` and no practical automated seam exists, return that constraint clearly so the coordinator can decide whether the task needs replanning, a documented exception, or a different verification approach.

## Resist Common Drift

Push back on these failure patterns:

- "fix it first and add tests after"
- "just patch the function quickly"
- "manual verification is enough for this one"
- "the change is too small to bother testing"

Do not argue abstractly. Translate the pushback into a smaller TDD step: name the narrow failing test, run it, make it pass, then continue.

## Checkpoint Reporting

At meaningful points, report progress in the language of the cycle:

- the behavior under test
- the current feature-task boundary if one was supplied
- the new failing test or check
- the command that proved red
- the implementation change made for green
- the command that proved green
- any remaining behaviors that still need another cycle

Keep these updates short. The point is to show evidence, not to narrate every keystroke.

## Quality Bar

Before finishing, check that:

- every intended behavior change in the current slice is covered by a new or updated automated check
- each important new test was observed failing before the corresponding implementation change
- the final implementation is no broader than the tested scope requires
- bug fixes include a regression test that would have caught the original failure
- the relevant verification commands were actually run and their outcomes are known
- any coordination context from `execute-plan` was honored without drifting into unrelated work

If those are not true, the work is not done yet. Keep cycling until the evidence matches the claim.

#!/usr/bin/env python
"""Prepare a py4vasp branch for pushing to origin. READ-ONLY: no push, no edits.

Run from anywhere inside the checkout (plain system python is fine):

    python .claude/skills/push-py4vasp/push_prep.py            # base = master
    python .claude/skills/push-py4vasp/push_prep.py main       # different base

It prints the decisions the push workflow needs:
  1. Changed files, classified into CODE (*.py under src/ or tests/) vs
     non-code (docs / CI / tooling).
  2. Whether a code review is required (any code changes present).
  3. A suggested descriptive branch name derived from the commit subjects.
  4. The GitHub PR "compare" URL for origin.
  5. A draft PR message (title + summary bullets) to copy/paste.

This tool decides nothing side-effectful. The agent orchestrates review,
formatting, and the actual push per SKILL.md.
"""
import pathlib
import re
import subprocess
import sys


def git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], text=True, capture_output=True
    ).stdout.strip()


def find_root(start):
    for parent in [start, *start.parents]:
        if (parent / ".git").exists() and (parent / "src" / "py4vasp").is_dir():
            return parent
        # worktrees keep a .git file, not dir; also accept src/py4vasp presence
        if (parent / "src" / "py4vasp").is_dir():
            return parent
    sys.exit("error: could not find a py4vasp checkout above " + str(start))


def changed_files(root, base):
    out = git(root, "diff", "--name-only", base, "HEAD")
    out += "\n" + git(root, "diff", "--name-only")
    out += "\n" + git(root, "diff", "--name-only", "--cached")
    return sorted({l.strip() for l in out.splitlines() if l.strip()})


def is_code(path):
    return path.endswith(".py") and (
        path.startswith("src/") or path.startswith("tests/")
    )


def classify(files):
    code, noncode = [], []
    for f in files:
        (code if is_code(f) else noncode).append(f)
    return code, noncode


def commit_subjects(root, base):
    log = git(root, "log", "--format=%s", f"{base}..HEAD")
    return [l for l in log.splitlines() if l.strip()]


def slugify(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return re.sub(r"-+", "-", text)[:50]


def suggest_branch(subjects, code, noncode):
    prefix_map = {"feat": "feat", "fix": "fix", "refactor": "refactor",
                  "docs": "docs", "test": "test", "chore": "chore"}
    if subjects:
        first = subjects[0]
        m = re.match(r"\s*(\w+)\s*:\s*(.*)", first)
        if m and m.group(1).lower() in prefix_map:
            return f"{prefix_map[m.group(1).lower()]}/{slugify(m.group(2))}"
        return slugify(first)
    # no commits ahead — name from the touched area
    sample = (code or noncode or ["changes"])[0]
    return "update/" + slugify(pathlib.Path(sample).stem)


def origin_https(url):
    url = url.strip()
    if url.endswith(".git"):
        url = url[:-4]
    m = re.match(r"git@([^:]+):(.+)", url)          # scp-like: git@host:owner/repo
    if m:
        return f"https://{m.group(1)}/{m.group(2)}"
    m = re.match(r"ssh://git@([^/:]+)(?::\d+)?/(.+)", url)  # ssh://git@host[:port]/owner/repo
    if m:
        return f"https://{m.group(1)}/{m.group(2)}"
    if url.startswith("https://"):
        return url
    return url


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "master"
    root = find_root(pathlib.Path.cwd())
    files = changed_files(root, base)
    code, noncode = classify(files)
    subjects = commit_subjects(root, base)
    branch = suggest_branch(subjects, code, noncode)
    https = origin_https(git(root, "config", "--get", "remote.origin.url"))
    compare_url = f"{https}/compare/{base}...{branch}?expand=1"

    print(f"[push_prep] checkout: {root}")
    print(f"[push_prep] base    : {base}")

    print("\n" + "=" * 60 + "\n1. CHANGED FILES\n" + "=" * 60)
    print("CODE (*.py in src/ or tests/):", *(["\n  " + f for f in code] or ["  (none)"]))
    print("non-code (docs/CI/tooling)   :", *(["\n  " + f for f in noncode] or ["  (none)"]))

    print("\n" + "=" * 60 + "\n2. REVIEW REQUIRED?\n" + "=" * 60)
    if code:
        print("YES - code changed. Dispatch /review-py4vasp to a subagent unless a")
        print("review was already conducted this session (ask the user if unsure).")
    else:
        print("NO - only docs/CI/tooling changed. Skip the review; go straight to push.")

    print("\n" + "=" * 60 + "\n3. SUGGESTED BRANCH NAME\n" + "=" * 60)
    print(f"  {branch}")
    print("  (Derive a fresh descriptive name; refine this if it misses the point.)")

    print("\n" + "=" * 60 + "\n4. PR COMPARE URL (origin)\n" + "=" * 60)
    print(f"  {compare_url}")
    print("  (Replace the branch segment if you push under a different name.)")

    print("\n" + "=" * 60 + "\n5. DRAFT PR MESSAGE\n" + "=" * 60)
    title = subjects[0] if subjects else branch
    print(f"Title: {title}\n")
    print("## Summary")
    for s in subjects or ["(no commits ahead of base — describe the change)"]:
        print(f"- {s}")
    print("\n## Notes")
    print("- Mention the related issue (e.g. \"Fixes #123\").")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

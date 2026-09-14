#!/usr/bin/env python
"""Prepare a py4vasp branch for pushing to origin. READ-ONLY: no push, no edits.

Run from anywhere inside the checkout (plain system python is fine):

    python .claude/skills/push-py4vasp/push_prep.py            # base = master
    python .claude/skills/push-py4vasp/push_prep.py main       # different base

It prints the decisions the push workflow needs:
  1. Changed files, classified into CODE (*.py under src/ or tests/) vs
     non-code (docs / CI / tooling).
  2. Whether a code review is required (any code changes present).
  3. Whether a user simulation is required (the diff touches the user
     interface: a public method of a quantity class, the CLI, or docs/).
  4. A suggested descriptive branch name derived from the commit subjects.
  5. The GitHub PR "compare" URL for origin.
  6. A draft PR message (title + summary bullets) to copy/paste.

This tool decides nothing side-effectful. The agent orchestrates review,
formatting, and the actual push per SKILL.md.
"""

import ast
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


def public_api_changes(root, base):
    """Public classes and methods under _calculation whose interface changed.

    A quantity's docstring is the documentation its users read, so rewriting one
    changes the interface even when the signature does not move. That is why this
    compares the parsed sources rather than the diff: with no context lines a rewritten
    docstring is an anonymous pair of +/- lines, so matching `def` lines alone reported
    no interface change at all for a branch that rewrote thirty docstrings and added
    runnable examples to six quantity classes.
    """
    changes = {}
    for path in changed_files(root, base):
        if not path.startswith("src/py4vasp/_calculation/") or not path.endswith(".py"):
            continue
        before = public_api(git(root, "show", f"{base}:{path}"))
        after = public_api(read_worktree(root, path))
        names = sorted(
            name
            for name in set(before) | set(after)
            if before.get(name) != after.get(name)
        )
        if names:
            changes[path] = names
    return changes


def read_worktree(root, path):
    """The file as it stands now, including changes that are not committed yet."""
    file = pathlib.Path(root) / path
    return file.read_text() if file.is_file() else ""


def public_api(source):
    """Map every public class and method to what a user of it can see.

    The value is the header and the docstring. The header runs from the first decorator
    to the line the body starts on, so a decorator counts as part of it -- py4vasp has
    one that substitutes text into the docstring, which makes it user-facing too.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    lines = source.splitlines()

    def visible(node):
        start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
        end = max(node.body[0].lineno - 1, node.lineno) if node.body else node.lineno
        return "\n".join(lines[start - 1 : end]), ast.get_docstring(node, clean=False)

    api = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
            continue
        api[node.name] = visible(node)
        for member in node.body:
            is_method = isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            if is_method and not member.name.startswith("_"):
                api[f"{node.name}.{member.name}"] = visible(member)
    return api


def user_interface_reasons(root, base, files):
    """Why a user simulation is required; an empty list means it is not."""
    reasons = []
    if "src/py4vasp/cli.py" in files:
        reasons.append("src/py4vasp/cli.py changed - the command line interface")
    if documentation := [f for f in files if f.startswith("docs/")]:
        reasons.append(f"documentation changed - {', '.join(documentation)}")
    for path, names in public_api_changes(root, base).items():
        reasons.append(f"public interface in {path} - {', '.join(names)}")
    return reasons


def commit_subjects(root, base):
    log = git(root, "log", "--format=%s", f"{base}..HEAD")
    return [l for l in log.splitlines() if l.strip()]


def slugify(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return re.sub(r"-+", "-", text)[:50]


def suggest_branch(subjects, code, noncode):
    prefix_map = {
        "feat": "feat",
        "fix": "fix",
        "refactor": "refactor",
        "docs": "docs",
        "test": "test",
        "chore": "chore",
    }
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
    m = re.match(r"git@([^:]+):(.+)", url)  # scp-like: git@host:owner/repo
    if m:
        return f"https://{m.group(1)}/{m.group(2)}"
    m = re.match(
        r"ssh://git@([^/:]+)(?::\d+)?/(.+)", url
    )  # ssh://git@host[:port]/owner/repo
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
    print(
        "CODE (*.py in src/ or tests/):", *(["\n  " + f for f in code] or ["  (none)"])
    )
    print(
        "non-code (docs/CI/tooling)   :",
        *(["\n  " + f for f in noncode] or ["  (none)"]),
    )

    print("\n" + "=" * 60 + "\n2. REVIEW REQUIRED?\n" + "=" * 60)
    if code:
        print("YES - code changed. Dispatch /review-py4vasp to a subagent unless a")
        print("review was already conducted this session (ask the user if unsure).")
    else:
        print(
            "NO - only docs/CI/tooling changed. Skip the review; go straight to push."
        )

    print("\n" + "=" * 60 + "\n3. USER SIMULATION REQUIRED?\n" + "=" * 60)
    reasons = user_interface_reasons(root, base, files)
    if reasons:
        print("USER SIMULATION REQUIRED: YES - the diff touches the user interface:")
        print(*("\n  " + reason for reason in reasons))
        print("\nDispatch /simulate-user-py4vasp to a subagent unless a simulation was")
        print("already run for this diff (ask the user if unsure). NOT from plan mode.")
    else:
        print("USER SIMULATION REQUIRED: NO - no public method, CLI, or docs change.")

    print("\n" + "=" * 60 + "\n4. SUGGESTED BRANCH NAME\n" + "=" * 60)
    print(f"  {branch}")
    print("  (Derive a fresh descriptive name; refine this if it misses the point.)")

    print("\n" + "=" * 60 + "\n5. PR COMPARE URL (origin)\n" + "=" * 60)
    print(f"  {compare_url}")
    print("  (Replace the branch segment if you push under a different name.)")

    print("\n" + "=" * 60 + "\n6. DRAFT PR MESSAGE\n" + "=" * 60)
    title = subjects[0] if subjects else branch
    print(f"Title: {title}\n")
    print("## Summary")
    for s in subjects or ["(no commits ahead of base — describe the change)"]:
        print(f"- {s}")
    print("\n## Notes")
    print('- Mention the related issue (e.g. "Fixes #123").')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

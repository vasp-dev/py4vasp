# The documentation site does not mention the command line interface

`docs/` has no reference page for the CLI. `py4vasp convert`, `symmetrize` and
`generate` exist only as `--help` output; the sole trace in the prose is one row of
the optional-dependency table saying the `cli` extra provides `python -m py4vasp`.

The Python side is as thin: `generate_kpath` and `generate_kmesh` appear only as two
words in that table, and `Structure.from_POSCAR` is pointed at from nowhere.

A user simulation found the feature from `--help` in two minutes and from the
documentation not at all. Needs a CLI reference page and a how-to.

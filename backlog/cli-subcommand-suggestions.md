# Mistyped subcommands get no suggestion

A mistyped option is handled well: `py4vasp generate kpath --npoints 20` answers
*"Did you mean --number-points?"*. A mistyped subcommand is not:
`py4vasp generate kmes` answers only *"No such command 'kmes'."*

click can suggest close matches for commands too, via a `Group` subclass that
overrides `resolve_command` (or `get_command`) and compares the token against
`list_commands` with `difflib.get_close_matches`. Applying it to the `cli` group and
to `generate` would make both levels behave the same.

Cosmetic, but the inconsistency is the sort of thing that makes a tool feel
unfinished.

The same holds one level down: `py4vasp convert structure lamps` answers "Converting
structure to lamps is not implemented." without listing `lammps` as the format that is
(from the reviewer notes of #353).

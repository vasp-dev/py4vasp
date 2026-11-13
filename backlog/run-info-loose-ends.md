# run_info: loose ends after reading the VASP setup writers

`run_info` reads the core settings from `/setup`, `/metadata` and `/messages` and
exposes every other recorded setting as `read()["extensions"]`. These points were
left open.

**Found by simulated users who only had the documentation.**
- Nothing in `docs/` names `run_info`. The users found it only by listing
  `calculation.selections(only_available=True)` or `dir(calculation)`. A how-to
  such as "check how a run was set up and whether it succeeded" would make it
  discoverable.
- Storing the summary in a database is not documented for users. The only mention
  is the developer page `docs/architecture/calculation.rst`, so a user cannot
  confirm that `run_info` ends up in a database or how to get it there.
- `run_info.read("electronic")` raises the generic "not a source" error. Because
  section names look like selections, the message could point to
  `read()["extensions"]["electronic"]`.
- For files that predate the setup writers, `print` silently omits exit status and
  settings. Writing "unknown (requires a newer VASP)" would distinguish not
  recorded from omitted.
- `print` reports the number of error messages but not their text, and leaves out
  `PREC`, `LDAU`, `nodes`, `gpus` and `build_variant`.
- `read()` drops the units VASP attaches to the datasets (the `units` attribute);
  only the docstring states them.
- `system_tag` is returned as bytes while the other values are plain Python
  types, so `json.dumps(run_info.read())` fails. This predates the setup work.
- The demo data reports `LKPOINTS_OPT: False` although it contains kpoints_opt
  results, and a phonon dispersion next to a relaxation.

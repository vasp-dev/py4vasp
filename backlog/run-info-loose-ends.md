# run_info: loose ends after reading the VASP setup writers

`run_info` reads the core settings from `/setup`, `/metadata` and `/messages` and
exposes every other recorded setting as `read()["extensions"]`. These points were
left open.

**Waiting for VASP.**
- The message fields assume VASP writes `messages/{text,severity,count}` together.
  If one is missing, py4vasp reports both message fields as None instead of
  failing.
- `metadata/resources` exists only in MPI builds, so `IMAGES`, `NCORE`, `KPAR`,
  `mpi_ranks`, `omp_threads`, `nodes` and `gpus` are None for serial builds.
- VASP writes only the numeric `IALGO`, so `ALGO` (read from
  `setup/electronic/ALGO`) is None until VASP writes the ALGO string there.
- `LDAU` relies on VASP writing the `setup/ldau` group only if LDA+U is active. If
  VASP ever writes it unconditionally, `LDAU` must read the LDAU flag instead.

**Found by a simulated user who only had the documentation.**
- Nothing in `docs/` names `run_info`. The user found it only by listing
  `calculation.selections(only_available=True)`. A how-to such as "check how a run
  was set up and whether it succeeded" would make it discoverable.
- `run_info.read("electronic")` raises the generic "not a source" error. Because
  section names look like selections, the message could point to
  `read()["extensions"]["electronic"]`.
- For files that predate the setup writers, `print` silently omits exit status and
  settings. Writing "unknown (requires a newer VASP)" would distinguish not recorded from
  omitted.
- `print` reports the number of error messages but not their text.
- `system_tag` is returned as bytes and `fermi_energy` as `np.float64`, while the
  other values are plain Python types. This predates the setup work.
- The demo data reports `LKPOINTS_OPT: False` although it contains kpoints_opt
  results.

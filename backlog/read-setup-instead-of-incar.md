# Read settings from /setup instead of /input/incar

`input/incar` holds only the tags the user wrote into the INCAR file. `/setup`, which VASP
writes since 6.7, holds the value VASP actually used, including defaults and values it
derived. The schema's `Fallback` reads the first of several datasets present in the
file, so a field can prefer `/setup` and keep `input/incar` for older files, as
`run_info` already does for `selective_dynamics`.

Candidates in `src/py4vasp/_raw/definition.py`:

- **`idipol` and `ldipol`** of every `raw.Cell` and `raw.Structure` source (default,
  final, phonon, exciton, ...) read `input/incar/IDIPOL` and `input/incar/LDIPOL`.
  `setup/dipole/IDIPOL` and `setup/dipole/LDIPOL` exist, so these could become
  `Fallback("setup/dipole/IDIPOL", "input/incar/IDIPOL")`. Check first whether the
  quantities treat a missing tag as "not set": with `/setup`, the default value is
  present instead, e.g. `LDIPOL = F`.
- **`SYSTEM`** (`raw.System` and `run_info.system_tag`) has no counterpart in `/setup`,
  so it stays on `input/incar`.
- The commented-out electron-phonon fields `input/incar/elph_selfen_carrier_*` should
  be checked against `/setup` once they are read.

Worth checking the other direction as well: the core settings of `run_info` read only
`/setup`, so files of VASP versions before 6.7 report them as None. A `Fallback` to
`input/incar` would recover the tags the user set, but a tag left at its default would
still be None, which `read()` would then have to document as "not set" rather than
"unknown".

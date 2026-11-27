# A POSCAR that cannot be read still aborts the availability listing

`Calculation.selections(only_available=True)` and `Calculation.is_available()` check
every source of every quantity. A source counts as unavailable when accessing it
raises one of `_UNAVAILABLE_EXCEPTIONS` in `_calculation/dispatch.py`. Under
`from_path`, the `poscar` source of `structure` reads `<path>/POSCAR`. Two cases
raise something else, so one bad file aborts the listing of every quantity:

- a POSCAR that does not parse raises `IncorrectUsage` from `_util/parse.py`;
- a POSCAR without read permission raises the builtin `PermissionError` from
  `_raw/read.py::CONTCAR`.

A file that is not text at all, e.g. the HDF5 file a calculation created with
`from_file` passes to this source, is already handled: `raw.access` turns the
`UnicodeDecodeError` into a `FileAccessError`.

The parse error helps when the user reads the source directly, so it should
probably not be converted where the file is accessed. Only the availability check
should treat it as unavailable. The `PermissionError` likely belongs next to the
`FileNotFoundError` branch of `_State.access` in `_raw/access.py`.

# Discovered follow-up issues

Issues uncovered while migrating quantities to the `Handler` + `@quantity` dispatcher
architecture. These are **out of scope for the migration itself** — the migration must
preserve behavior, not change it — so they are recorded here to be addressed separately.

## Handlers whose `to_database()` is an empty stub

These quantities return an empty `{}` from `to_database()`, so they carry **no** database
content. The new architecture correctly filters empty results out of the database
(`dispatch._result_has_data`), which is the desired behavior — but the underlying problem
is that the result *should not be empty*. Each should be taught to emit real database data.

- `CurrentDensityHandler.to_database()` — [current_density.py:67](src/py4vasp/_calculation/current_density.py#L67) returns `{}`.
- `PhononBandHandler.to_database()` — [phonon_band.py:49](src/py4vasp/_calculation/phonon_band.py#L49) returns `{}`.

Discovered via `tests/regression/test_manual.py`: the legacy reference for
`current_density:nmr` is itself an empty `{}` (emitted only as a presence marker), so the
new architecture drops it. The drop is correct; the emptiness is the real issue.

## `ionic()` dielectric-function demo cannot be written through the `"ion"` source

The `dielectric_function` `ionic()` demo ([dielectric_function.py:17](src/py4vasp/_demo/dielectric_function.py#L17))
populates `q_point` with real data, but the `"ion"` source schema
([definition.py:162](src/py4vasp/_raw/definition.py#L162)) defines **no** `q_point` path
(only the `"bse"` source does). `py4vasp._raw.write.write()` therefore tries to use the
unset `VaspData(None)` target as an HDF5 key and crashes
(`A name should be string or bytes, not VaspData`).

This is a pre-existing defect introduced by #271 (`Feat: q-point dependent dielectric
function`), which added `q_point` to the class and rewrote the `ionic()` demo to set it
(previously it set `current_current=VaspData(None)`, which was harmlessly skipped). It is
**not** caused by the architecture migration. Either the `ionic()` demo is mis-paired with
the `"ion"` source (q_point may belong to `"bse"`), or the `"ion"` source is missing a
`q_point` path. The corresponding regression case is commented out until this is resolved.

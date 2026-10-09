# Many quantities' `read()` take no selection

`born_effective_charge.read` and `elastic_modulus.read` now take a `selection` like
their `print`. A scan of the registry finds about twenty more quantities whose `read`
takes no `selection`: `current_density`, `density`, `dielectric_tensor`
(whose `to_INCAR` takes one), `effective_coulomb`, `electron_phonon` chemical potential
and self energy, `exciton.eigenvector`, `force`, `force_constant`, `internal_strain`,
`local_moment`, `nics`, `partial_density`, `piezoelectric_tensor`, `polarization`,
`potential`, `run_info`, `stress`, `structure`, `symmetry`, `system`, `velocity`,
`workfunction`. Some of them take other parameters instead (`structure` is sliced), so
each needs a look rather than a blanket change.

Calling one of them with a selection raises a bare `TypeError` from Python. The
cheapest fix that covers all of them at once is a check in the dispatch layer turning a
surplus positional argument into `IncorrectUsage` that lists the sources; the real fix
is the `selection` parameter on each `read`, as done for the two above.

Open question found alongside: `born_effective_charge.print()` says "cumulative output"
where OUTCAR says "cummulative output" (`linear_response.F:967`, `pead.F:2356,2368`), so
grepping one for the other fails. Copy VASP's spelling, or keep the correct one?

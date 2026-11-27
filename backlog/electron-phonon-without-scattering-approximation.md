# Electron-phonon runs without the scattering approximation

Some real runs (33 of the electron-phonon test runs we checked) write the
self energy and the band gap without `scattering_approximation`: the dataset of every
instance is missing. Reading works, but:

- `print`, `__str__`, and `selections()` of `electron_phonon.bandgap` and
  `electron_phonon.self_energy` raise NoData, because `selections()` takes
  `np.unique` of the approximations;
- the NoData message blames `selfen_carrier_den` and `selfen_mu`, which these runs
  deliberately lack (one doping suffices since the chemical potential fix);
- `len(bandgap)` counts only the SERTA instances, so it is 0 on these runs although
  `read()` returns data.

Until this is fixed, `is_available` reports these two quantities as available only
for `read` and `to_dict` on such runs, so a tool offers nothing that fails. Decide
what the approximation of an instance without one is (unknown? the run's default?),
make `selections()` and the printout cope with it, and count every instance in
`__len__` or document why only SERTA counts.

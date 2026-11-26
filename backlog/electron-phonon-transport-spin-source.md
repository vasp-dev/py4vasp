# The spin-resolved transport cannot be selected

The schema defines a second source `spin` for `electron_phonon.transport` (the
spin-resolved conductivity, mobility, Seebeck coefficient, ...). But
`ElectronPhononTransport.to_graph` and `select` read the default source directly
(`self._source.access(self._quantity_name)`) instead of going through the dispatcher,
so `transport.to_graph("spin(mobility)")` cannot reach it. Their docstrings now say
that the data always comes from the default source.

Route both through `merge_default` like the other routines, so that a source in the
selection is honored, and test with spin-resolved transport data.

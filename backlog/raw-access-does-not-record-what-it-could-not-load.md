# Raw access does not record which dataset it failed to load

When a field is absent, `_raw/access.py` hands out an anonymous `VaspData(None)`. Once
the raw dataclass has been built, nothing knows which HDF5 dataset was looked up, in
which file, or why it came back empty. The only error the user eventually sees is the
generic `NoData` raised lazily from `VaspData.data`:

    NoData: Could not find data in output, please make sure that the provided input
    should produce this data and that the VASP calculation already finished. Also check
    that VASP did not exit with an error.

Since #346 the dispatch layer improves this message by naming the missing source and any complete sources. To list the
missing datasets, it has to rebuild the paths by walking the schema against the raw
data afterwards. That can only be best effort, because the access layer resolves much
more than the schema shows:

- keys are templates expanded per index (`key.format(index)` for `Mapping` quantities
  with `valid_indices`), so the schema string is not the path that was read;
- `Length` keys turn into an `int` or `VaspData(None)`;
- a `Link` whose target is too old or whose file is missing silently becomes
  `VaspData(None)` in `_resolve_link`, so "missing" can mean "VASP version too old" or
  "file not found", not just "dataset absent";
- scalars are unwrapped to plain numpy values (`result[()]`), and sources built by a
  `data_factory` return whatever the factory makes, often plain numpy arrays, with no
  schema paths at all;
- a non-default `file` (`source.file` or the user's `file=`) changes which file was
  read, and the error does not name that file.

## Proposal

Let the access layer record what it did while it does it. `_State._get_dataset` and
`_resolve_link` know the resolved key, the file and the reason (dataset absent, link
target outdated, link file missing). Attach that to the empty value, e.g.
`VaspData(None, origin=MissingOrigin(file, key, reason))`, and have `VaspData.data` raise
`NoData` naming the dataset and the reason. Data factories and in-memory data
(`from_data` with numpy arrays) carry no origin and keep today's behavior.

The dispatch-level message can then use the recorded origins instead of walking the
schema, which removes its guesswork about template keys and links.

Open points: keep `VaspData` cheap (the origin is only created for missing data); check
that pickling, `dataclasses.replace` and the `__repr__` used in tests still behave; decide
whether an outdated-version link should raise `OutdatedVaspVersion` instead of `NoData`
when it is accessed.

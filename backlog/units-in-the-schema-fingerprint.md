# The schema fingerprint cannot see a unit change

`models.schema_fingerprint()` records the name and type of every database field, and
`tests/raw/test_schema_version.py` refuses a model change without a bump of
`__DB_SCHEMA__`. A unit is neither name nor type: when `StressModel` and the elastic
tensors moved from kBar to GPa, every row changed meaning by a factor of ten and the
fingerprint stayed identical. The check even rejected the bump that change needed.

The stopgap is `__DB_SEMANTIC_CHANGES__` in `_raw/models.py`: a bump with unchanged
fields is accepted when it records why. That relies on the developer noticing; nothing
fails if a unit changes and nobody writes the reason down.

The fix is to make the unit part of the field, so it lands in the fingerprint and a
change of unit is a model change like any other. Today the unit lives only in prose
("..., in GPa.") in the field docstrings, which the fingerprint deliberately ignores so
that wording edits do not force a bump. A structured unit, e.g. `Annotated[float,
Unit("GPa")]` or dataclass field metadata, would let the fingerprint record
`[name, type, unit]` and the docstrings stop repeating it. Every model needs one pass,
and a field without a unit (a ratio, a count) has to say so rather than be left blank.

`__DB_SEMANTIC_CHANGES__` stays for the meanings a unit does not capture, e.g. a
different reference energy or sign convention.

Depends on [energy-units-across-quantities]: decide which unit each quantity reports
before writing that unit into the schema, or the first round of fixes there forces a
second schema bump.

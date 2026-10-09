# Selection errors quote the parsed text, not what the user typed

`dispatch._parse_selections` splits the source off a selection and hands the handler the
rest, rebuilt with `select.selections_to_string`. A handler that rejects the rest can
only quote that rebuilt text. So `elastic_modulus.voigt("relaxed-ion")` answers
"The selection 'relaxed - ion' is not one of ...", which reads as if py4vasp computed a
difference, and `voigt("relaxed ion")` complains about `'relaxed'` alone. The message
still lists the valid choices, so the user recovers, but every quantity whose handler
validates its selection has the same echo.

Pass the user's original text along with the parsed remainder (e.g. a third field on
`SelectionContext`) so that error messages can quote it verbatim.

Related: `selections()` lists only the sources of a quantity. Choices a method accepts
on top of the source, such as `clamped_ion` and `relaxed_ion` of the elastic modulus or
the directions of the dielectric function, are discoverable only from the docstrings or
from an error message.

Found by the simulated user that validated `ElasticModulus.voigt` and the moduli.

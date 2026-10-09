# Missing-data advice does not name the INCAR tags, and other messages found alongside

Found by a simulated user who had only the documentation, while validating the change that
makes `NoData` name the missing source, the missing datasets and the complete sources.
That change got the user through; these are what was left.

**The INCAR advice is generic.** When no source of a quantity has data, the message ends

    No source of 'born_effective_charge' contains the required data. Please make sure that
    the INCAR tags of the calculation produce this data, that VASP finished, and that it
    did not exit with an error.

but never says *which* tags (here LEPSILON or IBRION = 7, 8), and neither does
`help(calc.born_effective_charge)`. The same holds for `dielectric_tensor` and most other
quantities. The dispatch layer cannot know the tags; each quantity would have to declare
them (e.g. a class attribute the message and the docstring both use). The user had to fall
back on their own VASP knowledge. An ordinary user would notice this.

`phonon.band` is a second example, found on a linear-response run that has modes but no
dispersion. The message should say that VASP writes a phonon band structure only when the
INCAR sets `LPHON_DISPERSION = T` and a `QPOINTS` file provides the q-point path, and that
the modes at Γ of a linear-response run are in `calc.phonon.mode`. The `dispersion` source
of `phonon.mode` needs the same tags. A hint like this may name an alternative quantity,
not only tags, so the declaration should allow free text.

**Smaller messages in the same area, all pre-existing:**

- A mistyped source (`mode.read("dispersoin")`) raises "The selection 'dispersoin' is not a
  source of the quantity 'phonon_mode' and the method takes no further selections. Use
  `selections` or `is_available` …". It could list the sources or suggest the closest one;
  "takes no further selections" is jargon. `_util/suggest.py::did_you_mean`, which the CLI
  uses for mistyped commands and formats, would give the suggestion.
- A misspelled projection selection (`band.plot("Sr(q)")`, or an element not in the
  structure such as `"Ba"`) lists the valid selections, which is enough, but then advises
  checking the INCAR file and the VASP version, which is a red herring for a typo.
- A wrong phonon mode number says "select a mode by the number print labels it with, from 1
  to 21". For the "dispersion" source `print` shows no labels at all; `print` is not
  formatted as code, so the sentence reads garbled.
- `to_image` says a relative filename is saved "relative to the internal path"; the user
  could not tell that this means the calculation directory and used an absolute path to
  avoid writing into their run.
- The advice "Use `selections` or `is_available`" leads nowhere when the quantity has no
  data: `phonon.band.selections()` on a run without a dispersion raises the very same
  `NoData` (from the reviewer notes of #349).
- An error raised while reading a linked quantity is reported under the link target, e.g.
  a missing structure under 'structure' rather than under the quantity the user asked
  for (from #346). Some users read the listed HDF5 paths as noise.

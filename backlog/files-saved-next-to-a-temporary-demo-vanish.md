# Files saved next to a temporary demo calculation vanish with it

Found while reviewing `feat/demo-calculation-without-path`, which lets
`demo.calculation()` write its data to a temporary directory that is removed once the
calculation and every quantity taken from it are garbage collected.

Every quantity's `to_image` and `to_csv` resolve a relative filename against the
directory of the calculation (`_third_party/graph/mixin.py:105-110` and `:159-165`), and
the docstrings say so. With a temporary demo that directory is the temporary one, so

    calculation = demo.calculation()
    calculation.dos.to_image(filename="dos.png")

succeeds, prints nothing, and writes `dos.png` into `/tmp/py4vasp-…`. The file is gone
as soon as `calculation` goes out of scope or the kernel restarts. The branch's own
`Graph.__add__` example does the same on purpose: it saves `comparison.png` to
`coarse.path() / "comparison.png"` and prints the path, which will not exist by the
time a reader goes to look.

Two related ways to hold on to a path that no longer exists:

- `pickle.loads(pickle.dumps(calculation))` gives a calculation that reads from the same
  directory but does not own it; once the original is collected, every read raises
  `FileAccessError`. This affects multiprocessing / joblib workers that receive a
  pickled demo calculation. `ArchiveSource` may have the same property; not checked.
- `Calculation.from_path(calculation.path())` likewise; this one is documented.

Options: write relative filenames of a temporary source to the current working
directory (and say so), or raise `IncorrectUsage` for a relative filename on a
temporary source; in the examples, save to an explicit directory the reader chose.
For pickling, either copy the data into a fresh `TemporarySource` on unpickling or
refuse to pickle a temporary source with a clear message.

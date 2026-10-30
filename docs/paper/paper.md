---
title: 'py4vasp: a Python interface to VASP calculation results'
tags:
  - Python
  - density functional theory
  - materials science
  - VASP
  - post-processing
authors:
  - name: Martin Schlipf
    orcid: 0000-0003-3198-0497
    corresponding: true
    affiliation: 1
  - name: Max Liebetreu
    orcid: 0000-0001-8374-8476
    affiliation: 1
  - name: Sudarshan Vijay
    orcid: 0000-0001-8242-0161
    affiliation: 2
  - name: Tomáš Bučko
    orcid: 0000-0002-5847-9478
    affiliation: 3
  - name: Orest Dubay
    affiliation: 4
  - name: Marie-Therese Huebsch
    orcid: 0000-0002-3541-2895
    affiliation: 1
  - name: Jonathan Lahnsteiner
    affiliation: 1
  - name: Henrique Miranda
    orcid: 0000-0002-2843-0876
    affiliation: 1
  - name: Christopher Sheldon
    orcid: 0000-0001-7160-2955
    affiliation: 1
  - name: Andreas Singraber
    orcid: 0000-0002-4330-1394
    affiliation: 1
  - name: Alexey Tal
    orcid: 0000-0002-2216-9283
    affiliation: 1
  - name: Michael Wolloch
    orcid: 0000-0002-3419-5526
    affiliation: 1
  - name: Georg Kresse
    orcid: 0000-0001-9102-4259
    affiliation: "1, 5"
affiliations:
  - name: VASP Software GmbH, Vienna, Austria
    index: 1
  - name: Indian Institute of Technology Bombay, Mumbai, India
    index: 2
  - name: Comenius University, Bratislava, Slovakia
    index: 3
  - name: Independent Researcher
    index: 4
  - name: Computational Materials Physics, University of Vienna, Vienna, Austria
    index: 5
date: 30 October 2026
bibliography: paper.bib
---

# Summary

py4vasp is a Python interface that exposes the results of a VASP
[@Kresse1996] calculation as `calculation.<quantity>.<method>` calls:
`calculation.dos.read()` returns the density of states as a Python
dictionary, `calculation.dos.plot()` returns an interactive figure, and the
same convention extends across the roughly forty physical quantities VASP
can compute, from electronic structure to lattice dynamics to structural
analysis. Every quantity is exposed the same way, in both a quick,
interactive form for a first look at the data and a script-friendly form
that hands off a Python dictionary or a `pandas` DataFrame for further
processing. py4vasp reads directly from VASP's HDF5 output rather than the
more brittle `OUTCAR` and XML formats, which matters most as VASP adds new
features. This paper accompanies version 1.0, after which this interface
will not change.

# Statement of need

VASP writes the results of a calculation into a single HDF5 file,
`vaspout.h5`, but turning that file into a band structure, a density of
states, or a set of phonon frequencies has traditionally been left to each
research group's own scripts or to general-purpose libraries such as ASE
[@Larsen2017] and pymatgen [@Ong2013]. Both typically parse the older
`OUTCAR` or XML output formats, whose free-form text changes as VASP evolves,
and the scripts get rewritten independently across groups and silently
diverge in correctness. `vaspout.h5`, by contrast,
is governed by an explicit, versioned schema rather than free-form text, so
it only needs to be parsed correctly once. We replace this pattern
with a single, shared interface that covers electronic structure, lattice
dynamics, response properties, and structural analysis alike, so that
learning to extract one quantity teaches the pattern for all of them.
Because this interface is designed to be shared across research groups
rather than reimplemented by each one, it is only useful if it keeps
working as VASP itself evolves: on every code change, continuous
integration checks almost every quantity and export method against
reference data on Linux, macOS, and Windows with Python 3.11 to 3.14,
including an installation that provides nothing beyond `numpy`
[@Harris2020] and `h5py` [@Collette2013].

# State of the field

Several tools provide programmatic access to first-principles simulation
output. General-purpose materials-informatics libraries operate across
multiple simulation codes: ASE [@Larsen2017] provides a common atomic
structure representation and calculator interface spanning a wide range of
codes including VASP; pymatgen [@Ong2013] similarly parses and analyzes
output from multiple codes; and AiiDA [@Pizzi2016] and pyiron
[@Janssen2019] manage and track the provenance of calculations across
codes as part of automated workflows.
VASPKIT [@Wang2021] and PyProcar [@Herath2020] are instead focused
specifically on VASP, covering high-throughput pre- and post-processing for
VASPKIT and band-structure and Fermi-surface analysis for PyProcar, each
through its own command-line tools and functions. Keeping a reader of
`vaspout.h5` consistent as VASP's output evolves requires close access to
VASP's own schema, which is why py4vasp is maintained by the same team that
develops VASP rather than as an extension of one of these tools.

# Software design

Every physical quantity is exposed through the same
`calculation.<quantity>.<method>` convention regardless of how
heterogeneous the underlying data is, so that learning to extract one
quantity is sufficient to extract any other. The trade-off is that any new
quantity must adhere to the same uniform shape rather than choosing an
optimal interface per quantity. py4vasp reads `vaspout.h5` directly,
which requires VASP to be linked against HDF5.

The rest of these design choices are easiest to see by following one task
through: comparing how a hybrid functional shifts the *p*-projected density
of states of silicon relative to PBE, then checking that a subsequent
relaxation of the hybrid-functional structure actually converged. A first
look at one calculation, for example in a Jupyter notebook inside its
directory, needs only

```python
import py4vasp
py4vasp.calculation.dos.plot("p")
```

which returns an interactive figure of the *p*-projected density of states;
`calculation.dos.read()` returns the same data as a Python dictionary
instead. Comparing two functionals means addressing both calculations
explicitly:

```python
pbe = py4vasp.Calculation.from_path("PBE")
pbe0 = py4vasp.Calculation.from_path("PBE0")
```

Creating these objects does not load any data: `Calculation.from_path(...)`
does not require the calculation to be finished, or even the path to exist
yet, and costs nothing until a `<method>` call actually reads from
`vaspout.h5`. A script can therefore set up as many calculation objects as
convenient and only pay the cost of reading a file for the ones it ends up
using. Even if the hybrid-functional calculation is not finished yet,
creating `pbe0` still succeeds; py4vasp only raises an error once something
actually tries to read from it.

Installation follows the same principle of paying only for what is used.
`pip install py4vasp-core`
installs the same code as `pip install py4vasp` with a dependency footprint
of only `numpy` and `h5py`; every further package is optional, and each
feature becomes available the moment the package it depends on is
installed, without reconfiguring py4vasp itself. These optional packages
cover interactive plotting (`pandas` [@McKinney2010], `plotly`
[@PlotlyTechnologies2015], `kaleido` [@KaleidoProject]), structural
analysis (`ase` [@Larsen2017], `spglib` [@Togo2018], `seekpath`
[@Hinuma2017]), and visualization in notebooks (`nglview` [@Nguyen2018],
`ipython` [@Perez2007]), among others; `pip install py4vasp` installs all
of them.

Using a feature whose package is missing raises an error that names the
package to install. Like an unfinished calculation or an unrecognized
selection, this is a situation py4vasp anticipates, and each of these
raises a `py4vasp.exception.Py4VaspError`. An error of any other type
indicates a bug or a system-level problem. A script can therefore use
`except exception.Py4VaspError:` to tell expected py4vasp errors apart from
unexpected failures.

Once both calculations are finished, comparing them needs only

```python
pbe.dos.plot("p").label("PBE") + pbe0.dos.plot("p").label("PBE0")
```

A plotting method returns a `Graph`, which holds its series as a plain
list, so any number of graphs can be combined into one figure with `+`,
not just two. The `"p"` selection used above is resolved by a parser
shared across every quantity that supports orbital-, atom-, or
spin-resolved output, so the same argument works identically for `band`
or any other quantity that accepts one.

Now suppose the hybrid-functional structure is relaxed and we want to
confirm the total energy actually converged along the way, not just
inspect the final structure. Every quantity that evolves over a trajectory
supports the same `[]` indexing: the final step is used by default, and
any other step or range can be selected explicitly.

```python
pbe0.energy[:].read()        # every ionic step of the relaxation
pbe0.structure[-1].to_ase()  # the relaxed structure, handed to ASE
```

Here, `to_ase` is one of several format-specific export methods
(`to_frame`, `to_csv`, `to_POSCAR`, `to_mdtraj` [@McGibbon2015], among
others) that follow the same `calculation.<quantity>.to_<format>()`
convention and, like the plotting methods, become available once the
package they hand the data to is installed.

A new user does not need to already know which quantities py4vasp
supports: tab-completing `calculation.` (or a `Calculation` instance) in an
interactive session lists every quantity py4vasp supports,
`selections(only_available=True)` on a `Calculation` narrows this to the
quantities computed for that particular run,
and `calculation.dos.selections()` — or the equivalent on any other
quantity — lists exactly which selections it accepts, such as the `"p"`
orbital used throughout this example. The [full
documentation](https://vasp.at/py4vasp/latest) covers every quantity and
method in more detail.

# Research impact statement

py4vasp is integrated into [VASP's official tutorial
curriculum](https://vasp.at/tutorials/latest/): 37 of its 40 parts,
spanning essentially every major method category VASP supports (electronic
structure, phonons, electron-phonon transport, GW, BSE, NMR, magnetism,
molecular dynamics, machine-learned force fields), execute py4vasp code
directly. It is also documented as a supported analysis tool on European
HPC infrastructure: an environment module on the [Tetralith
supercomputer](https://www.nsc.liu.se/software/catalogue/tetralith/modules/py4vasp.html)
(NSC, Sweden) and worked examples in [ENCCS's VASP Best Practices
guide](https://enccs.github.io/vasp-best-practices/tools/) for both
Tetralith and the EuroHPC LEONARDO system. Independent of the VASP
development team, a dataset from Oak Ridge National Laboratory published in
*Scientific Data* [@LupoPasini2024] explicitly recommends py4vasp for
analyzing VASP's HDF5 output, a recommendation repeated across [five
companion datasets](https://doi.org/10.13139/OLCF/2466323) from the same
group. py4vasp's [GitHub
repository](https://github.com/vasp-dev/py4vasp/issues) also receives
issues from users outside the development team, indicating use beyond
internal testing.

# AI usage disclosure

Two AI coding assistants were used during development, each with the most
recent model version available at the time. Since February 2026, GitHub
Copilot, using Claude Sonnet models from version 4.6 onward, has assisted
with implementing and testing individual code features. Since June 2026,
Claude Code, using Claude Opus models up to version 5.5, has assisted with
code development; the drafting of this manuscript used Claude Opus 5.5
exclusively. Every change, regardless of how it was written, passes the
same tests against reference data and continuous integration described in
the Statement of need before being merged, and the authors reviewed and take responsibility
for all code that is part of the release. On the manuscript side, Claude
Code drafted text from an outline, a tone-of-voice reference, and a
source-verified facts sheet prepared and directed by the authors, including
the code example above and the research-impact evidence above, which the
authors directed it to gather and verify from the cited sources. The
authors reviewed, edited, and checked all AI-assisted text against the
py4vasp source code and the cited literature, and take full responsibility
for the accuracy of the published paper.

# Conflicts of interest

Most of the authors are employed by VASP Software GmbH, which develops and
commercially licenses VASP; py4vasp is maintained by the same organization,
is free and open source, and improves the usability of VASP's output.

# Acknowledgements

This work was funded entirely by VASP Software GmbH, which employs most of
the authors and was therefore involved, through them, in the design and
development of py4vasp and in the preparation of this manuscript. We thank
Eisuke Kawashima, Shibu Meher, and Jan Priessnitz for their contributions
to the code, and the users of py4vasp for their feedback and bug reports.

# References

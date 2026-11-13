# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw

# Settings of a nonpolarized run in the layout of the /setup group VASP writes.
# Every entry is (value, units, description); units and description may be None.
_SETUP = {
    "dimensions": {
        "NGX": (32, None, "coarse FFT grid along the first lattice vector"),
        "NGY": (32, None, "coarse FFT grid along the second lattice vector"),
        "NGZ": (40, None, "coarse FFT grid along the third lattice vector"),
        "NGXF": (64, None, "fine FFT grid along the first lattice vector"),
        "NGYF": (64, None, "fine FFT grid along the second lattice vector"),
        "NGZF": (80, None, "fine FFT grid along the third lattice vector"),
        "NBANDS": (48, None, "number of bands"),
        "NKPTS": (20, None, "number of k points"),
    },
    "electronic": {
        "ISPIN": (1, None, "spin polarization"),
        "LNONCOLLINEAR": (False, None, "noncollinear magnetism"),
        "LSORBIT": (False, None, "spin-orbit coupling"),
        "ISTART": (0, None, "start from the WAVECAR file"),
        "ICHARG": (2, None, "initial charge density"),
        "NELM": (60, None, "maximum number of electronic steps"),
        "LREAL": ("Auto", None, "real-space projection"),
        "LMAXMIX": (4, None, "maximum l passed to the mixer"),
        "ENCUT": (520.0, "eV", "plane-wave cutoff"),
        "EDIFF": (1e-06, "eV", "stopping criterion of the electronic minimization"),
        "NELECT": (40.0, None, "number of electrons"),
        "NUPDOWN": (-1.0, None, "fixed magnetic moment"),
        "LASPH": (True, None, "nonspherical contributions inside the PAW spheres"),
        "PREC": ("accurate", None, "precision"),
        "ALGO": ("Normal", None, "electronic minimization algorithm"),
    },
    "ionic": {
        "POTIM": (0.5, "fs", "time step"),
        "TEBEG": (300.0, "K", "initial temperature"),
        "TEEND": (300.0, "K", "final temperature"),
        "MDALGO": (0, None, "molecular-dynamics algorithm"),
        "ISYM": (2, None, "symmetry treatment"),
        "selective_dynamics": (False, None, "selective dynamics"),
        "IBRION": (2, None, "ionic update"),
        "ISIF": (3, None, "stress and relaxation"),
        "NSW": (40, None, "number of ionic steps"),
        "EDIFFG": (-0.01, None, "stopping criterion of the ionic relaxation"),
    },
    "dos": {
        "ISMEAR": (0, None, "smearing method"),
        "SIGMA": (0.05, "eV", "smearing width"),
        "EFERMI_METHOD": ("LEGACY", None, "method to determine Fermi energy"),
    },
    "vdw": {
        "IVDW": (0, None, "van der Waals correction"),
    },
    "dipole": {
        "IDIPOL": (0, None, "dipole correction"),
        "LDIPOL": (False, None, "dipole correction of the potential"),
    },
    # the default demo contains bands and DOS on the KPOINTS_OPT mesh
    "kpoints_opt": {"LKPOINTS_OPT": (True, None, "evaluate KPOINTS_OPT")},
    "write_flags": {"LORBIT": (11, None, "projections onto local orbitals")},
    "exchange_correlation": {
        "functional_1/NAME": ("PBE", None, "name of the functional"),
        "AEXX": (0.0, None, "fraction of exact exchange"),
        "HFSCREEN": (0.0, None, "range-separation parameter"),
        "LHFCALC": (False, None, "exact exchange"),
        "XC": ("pbe", None, "functional components"),
    },
    "linear_response": {
        "LEPSILON": (False, None, "dielectric tensor from linear response"),
    },
    "ldau": {"LDAU": (False, None, "DFT+U")},
    # the same dispersion as the phonon band of the default demo calculation
    "phonons": {
        "method": ("finite differences", None, "method for the force constants"),
        "num_modes": (21, None, "number of modes at Gamma"),
        "num_qpoints": (164, None, "number of q points"),
        "num_branches": (21, None, "number of branches"),
    },
}
# The /metadata group in the same layout; resources exist only for MPI builds.
_METADATA = {
    "build": {
        "variant": ("std", None, "variant of the executable"),
        "mpi": (True, None, "built with MPI"),
        "openmp": (False, None, "built with OpenMP"),
        "configuration/fortran_compiler": ("demo-fortran", None, None),
        "gpu": (False, None, "built with GPU support"),
    },
    "resources": {
        "mpi_ranks": (64, None, "number of MPI ranks"),
        "images": (0, None, "number of images"),
        "KPAR": (4, None, None),
        "NCORE": (8, None, None),
        "NPAR": (2, None, None),
        "omp_threads": (1, None, "OpenMP threads per rank"),
        "nodes": (2, None, "number of nodes"),
        "gpus": (0, None, "number of GPUs, 0 for CPU builds"),
    },
    "performance": {
        "peak_memory": (262144.0, "kB", "peak memory"),
        "wall_time": (1834.5, "s", "Elapsed time (sec)"),
        "cpu_time": (1790.2, "s", "Total CPU time used (sec)"),
    },
}
_GROUPS = {"setup": _SETUP, "metadata": _METADATA}
_EXIT_STATUS = "success"
_MESSAGES = {
    "text": ["Use more k points.", "The basis set is incomplete."],
    "severity": ["warning", "error"],
    "count": [3, 1],
}


def Sr2TiO4():
    return raw.RunInfo(
        vasp_version=raw.Version(99, 99, 99),
        system_tag="Sr2TiO4",
        fermi_energy=0.5,
        number_ionic_steps=len(_demo.structure.Sr2TiO4().positions),
        poscar_selective_dynamics=0,
        phonon_frequencies=_demo.phonon.band.Sr2TiO4().dispersion.eigenvalues,
        **run_setup(),
        extension=extension(),
    )


def _value(section, key, table=_SETUP):
    "The value as VASP stores it, i.e., logicals as integers 0/1."
    value = table[section][key][0]
    return int(value) if isinstance(value, bool) else value


def _sections():
    "Every section the data factory reads, with the name it reports it under."
    for group, table in _GROUPS.items():
        prefix = "" if group == "setup" else f"{group}/"
        for name, entries in table.items():
            yield f"{group}/{name}", prefix + name, entries


def run_setup():
    "Fields of the core settings of a nonpolarized run that finished successfully."
    setup = {
        "NGX": ("dimensions", "NGX"),
        "NGY": ("dimensions", "NGY"),
        "NGZ": ("dimensions", "NGZ"),
        "NGXF": ("dimensions", "NGXF"),
        "NGYF": ("dimensions", "NGYF"),
        "NGZF": ("dimensions", "NGZF"),
        "NBANDS": ("dimensions", "NBANDS"),
        "nkpts": ("dimensions", "NKPTS"),
        "ISPIN": ("electronic", "ISPIN"),
        "LNONCOLLINEAR": ("electronic", "LNONCOLLINEAR"),
        "LSORBIT": ("electronic", "LSORBIT"),
        "ISTART": ("electronic", "ISTART"),
        "ICHARG": ("electronic", "ICHARG"),
        "NELM": ("electronic", "NELM"),
        "LREAL": ("electronic", "LREAL"),
        "LMAXMIX": ("electronic", "LMAXMIX"),
        "ENCUT": ("electronic", "ENCUT"),
        "PREC": ("electronic", "PREC"),
        "EDIFF": ("electronic", "EDIFF"),
        "ALGO": ("electronic", "ALGO"),
        "NELECT": ("electronic", "NELECT"),
        "NUPDOWN": ("electronic", "NUPDOWN"),
        "LASPH": ("electronic", "LASPH"),
        "ISMEAR": ("dos", "ISMEAR"),
        "SIGMA": ("dos", "SIGMA"),
        "efermi_method": ("dos", "EFERMI_METHOD"),
        "IBRION": ("ionic", "IBRION"),
        "ISIF": ("ionic", "ISIF"),
        "NSW": ("ionic", "NSW"),
        "EDIFFG": ("ionic", "EDIFFG"),
        "POTIM": ("ionic", "POTIM"),
        "TEBEG": ("ionic", "TEBEG"),
        "TEEND": ("ionic", "TEEND"),
        "MDALGO": ("ionic", "MDALGO"),
        "ISYM": ("ionic", "ISYM"),
        "selective_dynamics": ("ionic", "selective_dynamics"),
        "phonon_method": ("phonons", "method"),
        "phonon_num_modes": ("phonons", "num_modes"),
        "phonon_num_qpoints": ("phonons", "num_qpoints"),
        "phonon_num_branches": ("phonons", "num_branches"),
        "XC": ("exchange_correlation", "XC"),
        "LDAU": ("ldau", "LDAU"),
        "LHFCALC": ("exchange_correlation", "LHFCALC"),
        "AEXX": ("exchange_correlation", "AEXX"),
        "HFSCREEN": ("exchange_correlation", "HFSCREEN"),
        "IVDW": ("vdw", "IVDW"),
        "LDIPOL": ("dipole", "LDIPOL"),
        "IDIPOL": ("dipole", "IDIPOL"),
        "LEPSILON": ("linear_response", "LEPSILON"),
        "LKPOINTS_OPT": ("kpoints_opt", "LKPOINTS_OPT"),
        "LORBIT": ("write_flags", "LORBIT"),
    }
    metadata = {
        "IMAGES": ("resources", "images"),
        "NCORE": ("resources", "NCORE"),
        "KPAR": ("resources", "KPAR"),
        "mpi_ranks": ("resources", "mpi_ranks"),
        "omp_threads": ("resources", "omp_threads"),
        "nodes": ("resources", "nodes"),
        "gpus": ("resources", "gpus"),
        "peak_memory": ("performance", "peak_memory"),
        "wall_time": ("performance", "wall_time"),
        "cpu_time": ("performance", "cpu_time"),
        "build_variant": ("build", "variant"),
        "build_gpu": ("build", "gpu"),
        "build_openmp": ("build", "openmp"),
    }
    return dict(
        **{field: _value(*location) for field, location in setup.items()},
        **{field: _value(*location, _METADATA) for field, location in metadata.items()},
        exit_status=_EXIT_STATUS,
        number_messages=len(_MESSAGES["text"]),
        message_text=np.array(_MESSAGES["text"]),
        message_severity=np.array(_MESSAGES["severity"]),
        message_count=np.array(_MESSAGES["count"]),
    )


def extension():
    "Every entry of the /setup and /metadata groups as the data factory reads it."
    sections = {}
    for _, name, entries in _sections():
        section = raw.SetupSection()
        for key, (value, units, description) in entries.items():
            section.values[key] = value
            if units is not None:
                section.units[key] = units
            if description is not None:
                section.descriptions[key] = description
        sections[name] = section
    return raw.RunInfoExtension(sections=sections)


def write_to_h5(h5f):
    """Write the settings, metadata, and messages the way VASP does.

    raw.write cannot handle the data factory of the extension, so the groups are
    written directly with h5py."""
    for path, _, entries in _sections():
        for key, (value, units, description) in entries.items():
            data = int(value) if isinstance(value, bool) else value
            dataset = h5f.create_dataset(f"{path}/{key}", data=data)
            if isinstance(value, bool):
                dataset.attrs["dtype"] = "bool"
            if units is not None:
                dataset.attrs["units"] = units
            if description is not None:
                dataset.attrs["description"] = description
    # VASP declares fixed-length character variables, which Fortran pads with blanks
    h5f["messages/exit_status"] = _EXIT_STATUS.ljust(10)
    h5f["messages/number_messages"] = len(_MESSAGES["text"])
    h5f["messages/text"] = [text.ljust(80) for text in _MESSAGES["text"]]
    h5f["messages/severity"] = [severity.ljust(7) for severity in _MESSAGES["severity"]]
    h5f["messages/count"] = _MESSAGES["count"]

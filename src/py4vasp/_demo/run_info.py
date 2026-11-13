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
    "kpoints_opt": {"LKPOINTS_OPT": (False, None, "evaluate KPOINTS_OPT")},
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
    runtime_data = _demo.runtime_data.runtime_data("Sr2TiO4")
    return raw.RunInfo(
        system=raw.System("Sr2TiO4"),
        runtime=runtime_data,
        fermi_energy=0.5,
        structure=_demo.structure.Sr2TiO4(),
        contcar=_demo.CONTCAR.Sr2TiO4(),
        phonon_dispersion=_demo.phonon.band.Sr2TiO4().dispersion,
        setup=run_setup(),
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
    "Core settings of a nonpolarized run that finished successfully."
    setup = {
        "ngx": ("dimensions", "NGX"),
        "ngy": ("dimensions", "NGY"),
        "ngz": ("dimensions", "NGZ"),
        "ngxf": ("dimensions", "NGXF"),
        "ngyf": ("dimensions", "NGYF"),
        "ngzf": ("dimensions", "NGZF"),
        "nbands": ("dimensions", "NBANDS"),
        "nkpts": ("dimensions", "NKPTS"),
        "ispin": ("electronic", "ISPIN"),
        "lnoncollinear": ("electronic", "LNONCOLLINEAR"),
        "lsorbit": ("electronic", "LSORBIT"),
        "istart": ("electronic", "ISTART"),
        "icharg": ("electronic", "ICHARG"),
        "nelm": ("electronic", "NELM"),
        "lreal": ("electronic", "LREAL"),
        "lmaxmix": ("electronic", "LMAXMIX"),
        "encut": ("electronic", "ENCUT"),
        "prec": ("electronic", "PREC"),
        "ediff": ("electronic", "EDIFF"),
        "algo": ("electronic", "ALGO"),
        "nelect": ("electronic", "NELECT"),
        "nupdown": ("electronic", "NUPDOWN"),
        "lasph": ("electronic", "LASPH"),
        "ismear": ("dos", "ISMEAR"),
        "sigma": ("dos", "SIGMA"),
        "efermi_method": ("dos", "EFERMI_METHOD"),
        "ibrion": ("ionic", "IBRION"),
        "isif": ("ionic", "ISIF"),
        "nsw": ("ionic", "NSW"),
        "ediffg": ("ionic", "EDIFFG"),
        "potim": ("ionic", "POTIM"),
        "tebeg": ("ionic", "TEBEG"),
        "teend": ("ionic", "TEEND"),
        "mdalgo": ("ionic", "MDALGO"),
        "isym": ("ionic", "ISYM"),
        "xc": ("exchange_correlation", "XC"),
        "lhfcalc": ("exchange_correlation", "LHFCALC"),
        "aexx": ("exchange_correlation", "AEXX"),
        "hfscreen": ("exchange_correlation", "HFSCREEN"),
        "ivdw": ("vdw", "IVDW"),
        "ldipol": ("dipole", "LDIPOL"),
        "idipol": ("dipole", "IDIPOL"),
        "lepsilon": ("linear_response", "LEPSILON"),
        "lkpoints_opt": ("kpoints_opt", "LKPOINTS_OPT"),
        "lorbit": ("write_flags", "LORBIT"),
    }
    metadata = {
        "images": ("resources", "images"),
        "ncore": ("resources", "NCORE"),
        "kpar": ("resources", "KPAR"),
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
    # LDA+U is off, so VASP writes no ldau group and ldautype stays unset
    return raw.RunSetup(
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

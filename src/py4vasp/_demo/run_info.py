# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw

# Settings of a nonpolarized run in the layout of the /setup group VASP 6.7 writes.
# Every entry is (value, units, description); units and description may be None.
_SETUP = {
    "dimensions": {
        "NGX": (32, None, "coarse FFT grid along the first lattice vector"),
        "NGY": (32, None, "coarse FFT grid along the second lattice vector"),
        "NGZ": (40, None, "coarse FFT grid along the third lattice vector"),
        "NGXF": (64, None, "fine FFT grid along the first lattice vector"),
        "NGYF": (64, None, "fine FFT grid along the second lattice vector"),
        "NGZF": (80, None, "fine FFT grid along the third lattice vector"),
    },
    "electronic": {
        "ISPIN": (1, None, "spin polarization"),
        "LNONCOLLINEAR": (False, None, "noncollinear magnetism"),
        "LSORBIT": (False, None, "spin-orbit coupling"),
        "ISTART": (0, None, "start from the WAVECAR file"),
        "ICHARG": (2, None, "initial charge density"),
        "NELM": (60, None, "maximum number of electronic steps"),
        "LREAL": (False, None, "projection in real space"),
        "LMAXMIX": (4, None, "maximum l passed to the mixer"),
    },
    "ionic": {
        "POTIM": (0.5, "fs", "time step"),
        "TEBEG": (300.0, "K", "initial temperature"),
        "TEEND": (300.0, "K", "final temperature"),
        "MDALGO": (0, None, "molecular-dynamics algorithm"),
        "ISYM": (2, None, "symmetry treatment"),
    },
    "dos": {
        "ISMEAR": (0, None, "smearing method"),
        "SIGMA": (0.05, "eV", "smearing width"),
    },
    "kpoints_opt": {"LKPOINTS_OPT": (False, None, "evaluate KPOINTS_OPT")},
    "write_flags": {"LORBIT": (11, None, "projections onto local orbitals")},
    "exchange_correlation": {
        "functional_1/NAME": ("PBE", None, "name of the functional"),
    },
}
# The /metadata group in the same layout; resources exist only for MPI builds.
_METADATA = {
    "resources": {"mpi_ranks": 64, "images": 0, "KPAR": 4, "NCORE": 8, "NPAR": 2},
    "performance": {"peak_memory": 262144.0},  # kB
}
_EXIT_STATUS = "success"
_MESSAGES = {
    "text": ["Use more k points.", "The basis set is incomplete."],
    "severity": ["warning", "error"],
    "count": [3, 1],
}


def Sr2TiO4():
    runtime_data = _demo.runtime_data.runtime_data("Sr2TiO4")
    dos = _demo.dos.Sr2TiO4("with_projectors")
    band = _demo.band.single_band()
    return raw.RunInfo(
        system=raw.System("Sr2TiO4"),
        runtime=runtime_data,
        fermi_energy=0.5,
        bandgap=_demo.bandgap.bandgap(None),
        len_dos=len(dos.dos),
        band_dispersion_eigenvalues=band.dispersion.eigenvalues,
        band_projections=band.projections,
        structure=_demo.structure.Sr2TiO4(),
        contcar=_demo.CONTCAR.Sr2TiO4(),
        phonon_dispersion=_demo.phonon.band.Sr2TiO4().dispersion,
        setup=run_setup(),
        extension=extension(),
    )


def _value(section, key):
    value = _SETUP[section][key][0]
    return int(value) if isinstance(value, bool) else value


def run_setup():
    "Core settings of a nonpolarized run that finished successfully."
    core = {
        "ngx": ("dimensions", "NGX"),
        "ngy": ("dimensions", "NGY"),
        "ngz": ("dimensions", "NGZ"),
        "ngxf": ("dimensions", "NGXF"),
        "ngyf": ("dimensions", "NGYF"),
        "ngzf": ("dimensions", "NGZF"),
        "ispin": ("electronic", "ISPIN"),
        "lnoncollinear": ("electronic", "LNONCOLLINEAR"),
        "lsorbit": ("electronic", "LSORBIT"),
        "istart": ("electronic", "ISTART"),
        "icharg": ("electronic", "ICHARG"),
        "nelm": ("electronic", "NELM"),
        "lreal": ("electronic", "LREAL"),
        "lmaxmix": ("electronic", "LMAXMIX"),
        "sigma": ("dos", "SIGMA"),
        "potim": ("ionic", "POTIM"),
        "tebeg": ("ionic", "TEBEG"),
        "teend": ("ionic", "TEEND"),
        "mdalgo": ("ionic", "MDALGO"),
        "isym": ("ionic", "ISYM"),
        "lkpoints_opt": ("kpoints_opt", "LKPOINTS_OPT"),
        "lorbit": ("write_flags", "LORBIT"),
    }
    return raw.RunSetup(
        **{field: _value(*location) for field, location in core.items()},
        images=_METADATA["resources"]["images"],
        ncore=_METADATA["resources"]["NCORE"],
        kpar=_METADATA["resources"]["KPAR"],
        mpi_ranks=_METADATA["resources"]["mpi_ranks"],
        peak_memory=_METADATA["performance"]["peak_memory"],
        exit_status=_EXIT_STATUS,
        message_text=np.array(_MESSAGES["text"]),
        message_severity=np.array(_MESSAGES["severity"]),
        message_count=np.array(_MESSAGES["count"]),
    )


def extension():
    "Every setting of the /setup group as the data factory reads it."
    sections = {}
    for name, entries in _SETUP.items():
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
    """Write the settings, metadata, and messages the way VASP 6.7 does.

    raw.write cannot handle the data factory of the extension, so the groups are
    written directly with h5py."""
    for name, entries in _SETUP.items():
        for key, (value, units, description) in entries.items():
            dataset = h5f.create_dataset(f"setup/{name}/{key}", data=_value(name, key))
            if isinstance(value, bool):
                dataset.attrs["dtype"] = "bool"
            if units is not None:
                dataset.attrs["units"] = units
            if description is not None:
                dataset.attrs["description"] = description
    for group, entries in _METADATA.items():
        for key, value in entries.items():
            h5f[f"metadata/{group}/{key}"] = value
    # VASP declares fixed-length character variables, which Fortran pads with blanks
    h5f["messages/exit_status"] = _EXIT_STATUS.ljust(10)
    h5f["messages/number_messages"] = len(_MESSAGES["text"])
    h5f["messages/text"] = [text.ljust(80) for text in _MESSAGES["text"]]
    h5f["messages/severity"] = [severity.ljust(7) for severity in _MESSAGES["severity"]]
    h5f["messages/count"] = _MESSAGES["count"]

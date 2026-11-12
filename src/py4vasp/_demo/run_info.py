# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


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
    )


def run_setup():
    "Core settings of a nonpolarized run that finished successfully."
    return raw.RunSetup(
        ngx=32,
        ngy=32,
        ngz=40,
        ngxf=64,
        ngyf=64,
        ngzf=80,
        ispin=1,
        lnoncollinear=0,
        lsorbit=0,
        istart=0,
        icharg=2,
        nelm=60,
        lreal=0,
        lmaxmix=4,
        sigma=0.05,
        potim=0.5,
        tebeg=300.0,
        teend=300.0,
        mdalgo=0,
        isym=2,
        lkpoints_opt=0,
        lorbit=11,
        images=0,
        peak_memory=262144.0,
        exit_status="success",
        message_text=np.array(["Use more k points.", "The basis set is incomplete."]),
        message_severity=np.array(["warning", "error"]),
        message_count=np.array([3, 1]),
    )

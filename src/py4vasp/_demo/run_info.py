# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
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
    )

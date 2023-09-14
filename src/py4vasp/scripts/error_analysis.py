# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import argparse
import sys
from argparse import RawTextHelpFormatter

import numpy as np

from py4vasp._analysis.mlff import MLFFErrorAnalysis
from py4vasp._third_party.graph import Graph, Series


def get_options(args=sys.argv[1:]):
    """
    parses input arguments
    Parameters
    ----------
    -dft/ --DFTfiles
          user has to supply a list of vaspout.h5 files generated during
          DFT calculations
    -ml/ --MLfiles
          user has to supply a list of vaspout.h5 files generated during
          machine learning calculations. The order of the files has to match
          the order of DFT files
    -plt/ --MakePlot
          whit this flag the user can decide if a plot of the errors will
          be shown
    -pdf/ --pdfplot
          whit this flag the user can decide if a plot of the errors will
          be stored to a pdf file
    -txt/ --XYtextFile
          with this file the user gets 3 error files in xy format
          the default is a csv file containing all three errors
    """
    parser = argparse.ArgumentParser(
        description="py4vasp error-analysis\n" + "----------------------\n"
        "This script will extract energies, forces and stress tensors from two sets of files. \n"
        "The first set of files are vaspout.h5 files computed with first-principles DFT calculations.\n"
        + "The second set of files belongs to the same POSCARs but this time the data is\n"
        + "computed with a pre-trained machine-learned force field.\n"
        + "Then the script computes the errors between the energies, forces and stress tensors.",
        formatter_class=RawTextHelpFormatter,
    )
    requiredNamed = parser.add_argument_group("required arguments")
    requiredNamed.add_argument(
        "-dft",
        "--DFTfiles",
        required=True,
        type=str,
        help="Your vaspout.h5 input files obtained from DFT calculations.",
    )
    requiredNamed.add_argument(
        "-ml",
        "--MLfiles",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-plt",
        "--MakePlot",
        action="store_true",
        help="supply flag (without keyword) if you would like to generate a plot",
    )
    parser.add_argument(
        "-pdf",
        "--pdfplot",
        action="store_true",
        help="supply flag (without keyword) if you would like to save plot to pdf",
    )
    parser.add_argument(
        "-txt",
        "--XYtextFile",
        action="store_true",
        help="Supply flag (without keyword) if you want to have XY txt files for the computed errors.\n"
        + "Default output will be a csv file (ErrorAnalysis.csv) containing all analysed errors",
    )
    options = parser.parse_args(args)

    return options


def write_energy_error_file(cls, fname="EnergyError.out"):
    files = cls._calculations.files()
    energy_error_per_atom = cls.get_energy_error_per_atom()
    dft_files = files["dft_data"]
    mlff_files = files["mlff_data"]
    writeout = np.array([dft_files, mlff_files, energy_error_per_atom]).T
    header = "file_path_dft, file_path_mlff, energy difference in eV/atom (value > 0 MLFF predicts too high value)"
    np.savetxt(fname, writeout, fmt="%s", delimiter=",", header=header)


def write_force_error_file(cls, fname="ForceError.out"):
    files = cls._calculations.files()
    force_error = cls.get_force_rmse()
    dft_files = files["dft_data"]
    mlff_files = files["mlff_data"]
    writeout = np.array([dft_files, mlff_files, force_error]).T
    header = "file_path_dft, file_path_mlff, force rmse in eV/Angstrom"
    np.savetxt(fname, writeout, fmt="%s", delimiter=",", header=header)


def write_stress_error_file(cls, fname="StressError.out"):
    files = cls._calculations.files()
    stress_error = cls.get_stress_rmse()
    dft_files = files["dft_data"]
    mlff_files = files["mlff_data"]
    writeout = np.array([dft_files, mlff_files, stress_error]).T
    header = "file_path_dft, file_path_mlff, stress rmse in kbar"
    np.savetxt(fname, writeout, fmt="%s", delimiter=",", header=header)


def make_plot(cls, show=False, pdf=False, graph_name="ErrorAnalysis.pdf"):
    energy_error = cls.get_energy_error_per_atom()
    force_error = cls.get_force_rmse()
    stress_error = cls.get_stress_rmse()
    xaxis = np.arange(1, len(energy_error) + 1)
    energy = Series(xaxis, energy_error[:], subplot=1)
    force = Series(xaxis, force_error[:], subplot=2)
    stress = Series(xaxis, stress_error[:], subplot=3)
    graph = Graph((energy, force, stress))
    graph.xlabel = ("configuration",) * 3
    graph.ylabel = (
        "error energy [eV/atom]",
        "rmse force [eV/Å]",
        "rmse stress [kbar]",
    )
    figure = graph.to_plotly()
    if pdf:
        figure.write_image(graph_name)
    if show:
        figure.show()
    return figure


def main():
    options = get_options(sys.argv[1:])
    mlff_error_analysis = MLFFErrorAnalysis.from_files(
        dft_data=options.DFTfiles, mlff_data=options.MLfiles
    )
    if options.XYtextFile:
        write_energy_error_file(mlff_error_analysis)
        write_force_error_file(mlff_error_analysis)
        write_stress_error_file(mlff_error_analysis)
    if options.MakePlot or options.pdfplot:
        make_plot(mlff_error_analysis, options.MakePlot, options.pdfplot)

# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import argparse
import sys
from argparse import RawTextHelpFormatter

import numpy as np
import pandas as pd

import py4vasp
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
        nargs="+",
        required=True,
        help="Your vaspout.h5 input files obtained from DFT calculations."
        + " Supply in a form, as for example vaspout_{1..200}.h5",
    )
    requiredNamed.add_argument(
        "-ml",
        "--MLfiles",
        nargs="+",
        required=True,
        help="Your vaspout.h5 input files obtained from machine learning calculations"
        + " Supply in a form, as for example vaspout_{1..200}.h5",
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


class Reader:
    """
    Reading a single vaspout.h5 file

    Parameters
    ----------
    fname : str
        Input filename

    Attributes
    ----------
    fname : str
        file name of hdf5 file
    energy : float
        total energy computed by vasp
    force : ndarray
        force array ( NIONS , 3 )
    lattice : ndarray
        Bravais matrix for structure used for VASP calculation
    positions : ndarray
        atomic positions in direct coordinates
    NIONS : int
        number of ions in current structure
    stress : ndarray
        stress tensor computed by vasp

    Methods
    -------
    read_energy
        reading the energy, initialize energy
    read_force
        reading force, initialize force, postions, NIONS, lattice
    read_stress
        reading stress tensor upper triangle
    """

    def __init__(self, fname):
        self.fname = fname
        self.calc = py4vasp.Calculation.from_file(self.fname)
        self.read_energy()
        self.read_forces()
        self.read_stress()

    def read_energy(self):
        """read energy from vaspout.h5 file"""
        energy = self.calc.energy.read()
        self.energy = energy["free energy    TOTEN"]

    def read_forces(self):
        """read from vaspout.h5 file the following arrays:
        forces, positions, NIONS, lattice
        """
        force = self.calc.force.read()
        self.lattice = force["structure"]["lattice_vectors"]
        self.positions = force["structure"]["positions"]
        self.NIONS = self.positions.shape[0]
        self.force = force["forces"]

    def read_stress(self):
        """read upper triangle
        of stress tensor from vaspout.h5 file
        """
        stress = self.calc.stress.read()
        self.stress = stress["stress"][np.triu_indices(3)]


class AnalyseErrorSingleFile:
    """compare a VASP MLFF and DFT file

    Parameters
    ----------
    MLFF : str
        vaspout.h5 file from machine learning calculation
    DFT : str
        vaspout.h5 file from DFT calculation

    Attributes
    ----------
    MLdata : Reader
        contains raw machine learning data
    DFTdata : Reader
        contains DFT data
    force_error : float
        root mean square error force
    energy_error : float
        energy error per atom
    stress_error : float
        root mean square error of stress tensor

    Methods
    -------
    write_array_to_screen
        writing array to screen
    check_structures_for_eqivalence
        compare structure parameters, positions, lattice, number ions for
        equivalence between structures
    compute_force_error
        compute root mean square error between forces
    compute_energy_error_atom
        computing the error in energy per atom
    compute_stress_error
        compute the root mean square error of upper triangle of stress tensor
    root_mean_square_error_numpy_array
        compute the root mean suqare error between two numpy arrays
    root_mean_square_error_list
        compute root mean square error between two python lists containing
        ndarrays arrays
    """

    def __init__(self, MLFF, DFT):
        self.ml_fname = MLFF
        self.dft_fname = DFT

        print("Analysing errors between", self.ml_fname, " and ", self.dft_fname)

        # reading input data with Reader class
        self.ml_data = Reader(self.ml_fname)
        self.dft_data = Reader(self.dft_fname)

        # error checking if the two structures to compare are the same
        self.check_structures_for_equivalence()

        # computing errors of the data extracted from vaspout.h5 file
        self.compute_force_error()
        self.compute_energy_error_atom()
        self.compute_stress_error()

    @staticmethod
    def write_array_to_screen(data):
        """writing an array to the screen"""
        for row in data:
            print(row)

    def check_structures_for_equivalence(self):
        np.testing.assert_equal(
            self.ml_data.NIONS,
            self.dft_data.NIONS,
            err_msg=f"Number of ions does not match in {self.ml_fname} and {self.dft_fname}",
        )

        np.testing.assert_array_almost_equal(
            self.ml_data.positions,
            self.dft_data.positions,
            decimal=6,
            err_msg="The positions of your input data does not match in files "
            + f"{self.ml_fname} and {self.dft_fname}",
        )

        np.testing.assert_array_almost_equal(
            self.ml_data.lattice,
            self.dft_data.lattice,
            decimal=6,
            err_msg="The lattice of your input data does not match in files "
            + f"{self.ml_fname} and {self.dft_fname}",
        )

    def compute_force_error(self):
        """compute the root mean square error between force arrays
        of the DFT and machine learning approach

        Returns
        -------
        force_error : float
            root mean square error of force:
            over x,y,z components and ions
        """
        error = np.linalg.norm(self.dft_data.force[:, :] - self.ml_data.force[:, :])
        self.force_error = error / np.sqrt(self.ml_data.NIONS * 3)

    def compute_energy_error_atom(self):
        """computing the energy error per atom between machine learning and DFT
        error = ( E_{DFT}-E_{MLFF} ) / NIONS

        Returns
        -------
        energy_error : float
            energy difference divided by number atoms
        """
        energy_difference = self.ml_data.energy - self.dft_data.energy
        self.energy_error = energy_difference / self.ml_data.NIONS

    def compute_stress_error(self):
        """computing the root mean square error of the stress tensor
        between DFT and machine learning calculation

        Returns
        -------
        stress_error  --> root mean square error of stress tensor over its components
        """
        self.stress_error = np.linalg.norm(self.ml_data.stress - self.dft_data.stress)
        self.stress_error /= np.sqrt(self.ml_data.stress.shape[0])

    @staticmethod
    def root_mean_square_error_numpy_array(data_A, data_B):
        """computing root mean square error between two data sets

        Parameters
        ----------
        dataA, dataB : ndarray
            numpy ndarray where sizes have to match

        Returns
        -------
        float
            root mean square error between dataA and dataB
        """
        return np.linalg.norm(data_A - data_B) / np.sqrt((np.product(data_A.shape)))

    @classmethod
    def root_mean_square_error_list(cls, data_A, data_B):
        """computing root mean square errors between equally sized python
        lists containing numpy arrays

        Parameters
        ----------
        data_A, data_B : list
            python list containing numpy arrays of equal size and order as in
            the data_B array
        """
        np.testing.assert_equal(
            len(data_A),
            len(data_B),
            err_msg=f"Error in root_mean_square_error_list in {cls.__name__}"
            + "the dimensions of the list do not match",
        )
        error = 0
        number_elements = 0
        for a, b in zip(data_A, data_B):
            error += np.sum((a - b) ** 2)
            number_elements += a.size

        error = np.sqrt(error / number_elements)
        return error


class AnalyseError:
    """compute errors between two vaspout files

    Parameters
    ----------
    ML  : str
        list of machine learning vaspout.h5 files
    Dft : str
        list of DFT vaspout.h5 files to compare MLFF

    Attributes
    ----------
    indx_start : int
        is always one and gives the start point of xaxis
    indx_end : int
        end point of xaxis, determined by number of files
    eV_to_meV : float
        conversion factor from electron volts (eV) to meV
    xaxis : ndarray
        equally spaced axis between indx_start and indx_end
        indx_start  =  1
        indx_end    =  len( ML )
        len( ML ) points
    energy_error : ndrray
        energy erros per atom for all supplied file pairs
    force_error : ndarray
        root mean square erros between the supplied file pairs
    stress_error : ndarray
       root mean square error between supplied file pairs
    energy : ndarray
        DFT and MLFF energies per atom of all supplied files
    force : list
        forces of the DFT and MLFF calculations extracted from supplied files
    stress : ndarray
        upper triangular part of stress tensor of DFT and MLFF calculations

    Methods
    -------
    make_x_axis
         generate xaxis for plots, structure index
    compute_errors
         computing the errors between the two file sets for every file pair
         seperate
    compute_average_errors
         compute root mean sqaure erros between total data set
    print_average_errors
         printing average erros to screen
    plot_energy_error
         generate plot for energy error per atom
    plot_force_error
         plot root mean square error for forces
    plot_stress_error
         plot root mean square erros of stress tensor
    make_plot
         plot energy, force, stress error in subplots
    prepare_output_array
         concatenate two arrays along second axis
    format_float
         convert ndarray to formatted output string
    writing_energy_error_file
         writing energy errors per atom to file
    writing_force_error_file
         writing root mean square errors to file
    writing_stress_error_file
         writing root mean square error of force to file
    write_csv_output_file
         writing all collected errors to csv file
    """

    def __init__(self, ML, Dft):
        self.ml_files = ML
        self.dft_files = Dft
        np.testing.assert_equal(
            len(self.ml_files),
            len(self.dft_files),
            err_msg="Different number of dft and mlff files supplied. Please check your input",
        )

        self.indx_start = 1
        self.indx_end = len(ML)

        # scaling factor to meV
        self.eV_to_meV = 1000

        self.make_x_axis()

        self.compute_errors()
        self.compute_average_errors()
        self.print_average_errors()

    def make_x_axis(self):
        """create a x-axis for plots

        IndxStart and IndxEnd determine start and end
        the interval will be decomposed into
        number of supplied input files
        """
        self.xaxis = np.linspace(self.indx_start, self.indx_end, len(self.ml_files))

    def compute_errors(self):
        """computing the errors in the data with the help of the class
        AnalyseErrorSingleFile

        Returns
        -------
        energy_error : ndarray
            2d array of size (Nstructures , 1 )
            error per atom between DFT and MLFF calculation,
            computed in AnalyseErrorSingleFile
        force_error : ndarray
            2d array of size ( Nstructures , 1 )
            root mean square error in force between the DFT and MLFF calculation
            computed with AnalyseErrorSingleFile
        stress_error : ndarray
            2d array of size (Nstructures , 1 )
            root mean square error of stress tensor between a machine learning
            calculation and DFT calculation
        energy : ndarray
            2d array of size (Nstructures , 2 )
            self.energy[ : , 0 ] -> DFT energies in [eV]
            self.energy[ : , 1 ] -> MLFF energies in [eV]
        force : list
            python list of len 2
            index 0 is the dft data
            index 1 is the machine learning data
            both indexes then contain python lists of the len of reference
            configurations; and in those numpy arrays are contained
            NIONS,3
                   list               list           numpy array
            force[dft/ml][ reference configuration ][ NIONS , 3 ]
            units are in eV/Angstroem
        stress : ndarray
             3d array of size ( Nstructures , 2 , size stress )
             1st index -> structure index
             2nd index refers to DFT ->  0 / MLFF -> 1
             3rd index are components of stress tensor.
             units are in kbar
        """
        self.force_error = np.zeros(len(self.ml_files))
        self.energy_error = np.zeros(len(self.ml_files))
        self.stress_error = np.zeros(len(self.ml_files))

        # extract NIONS
        x = AnalyseErrorSingleFile(self.ml_files[0], self.dft_files[0])
        self.energy = {
            "dft": np.zeros([len(self.ml_files), 1]),
            "ml": np.zeros([len(self.ml_files), 1]),
        }

        self.stress = {
            "dft": np.zeros([len(self.ml_files), x.ml_data.stress.shape[0]]),
            "ml": np.zeros([len(self.ml_files), x.ml_data.stress.shape[0]]),
        }
        # force has to be a list to make variable atom numbers possible
        self.force = {"dft": [], "ml": []}

        for i in range(len(self.ml_files)):
            x = AnalyseErrorSingleFile(self.ml_files[i], self.dft_files[i])
            # extract energies
            self.energy_error[i] = x.energy_error * self.eV_to_meV
            self.energy["dft"][i] = x.dft_data.energy / x.dft_data.NIONS
            self.energy["ml"][i] = x.ml_data.energy / x.ml_data.NIONS

            # extract forces
            self.force_error[i] = x.force_error * self.eV_to_meV
            self.force["dft"].append(x.dft_data.force)
            self.force["ml"].append(x.ml_data.force)

            # extract stress
            self.stress_error[i] = x.stress_error
            self.stress["dft"][i, :] = x.dft_data.stress
            self.stress["ml"][i, :] = x.ml_data.stress

    def compute_average_errors(self):
        """computing single error values for energy, forces, stress
        over the whole data set

        Returns
        -------
        rmse_energy : float
            is the root mean square error of the energies computed over the
            supplied configurations in [meV]
        rmse_force : float
            root mean square error of forces computed over all configurations,
            atoms and x y and z direction in [meV]/Angstroem
        rmse_stress : float
            root mean square error of stress tensor computed over all configurations
            and components of the tensor
        """
        self.rmse_energy = AnalyseErrorSingleFile.root_mean_square_error_numpy_array(
            self.energy["dft"], self.energy["ml"]
        )
        self.rmse_energy *= self.eV_to_meV

        self.rmse_force = AnalyseErrorSingleFile.root_mean_square_error_list(
            self.force["dft"], self.force["ml"]
        )
        self.rmse_force *= self.eV_to_meV

        self.rmse_stress = AnalyseErrorSingleFile.root_mean_square_error_numpy_array(
            self.stress["dft"], self.stress["ml"]
        )

    def print_average_errors(self):
        """printing the root mean square errors of energy, force and stress tensor
        in the test set to the screen
        """
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Root mean square error of energy in meV/atom", self.rmse_energy)
        print("Root mean square error of force in meV/Angstroem", self.rmse_force)
        print("Root mean square error of stress tensor in kbar", self.rmse_stress)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def plot_energy_error(self):
        """plotting energy error/atom with sign versus the input configurations"""
        graph = Graph(
            series=Series(self.xaxis, self.energy_error[:, 0]),
            xlabel="configuration",
            ylabel="energy error [meV/atom]",
        )
        figure = graph.to_plotly()
        figure.show()
        return figure

    def plot_force_error(self):
        """plotting force rmse meV/Angstroem versus input configurations"""
        graph = Graph(
            series=Series(self.xaxis, self.force_error[:, 0]),
            xlabel="configuration",
            ylabel="rmse force [meV/Å]$",
        )
        figure = graph.to_plotly()
        figure.show()
        return figure

    def plot_stress_error(self):
        """plotting force rmse meV/Angstroem versus input configurations"""
        graph = Graph(
            series=Series(self.xaxis, self.stress_error[:, 0]),
            xlabel="configuration",
            ylabel="rmse stress [kbar]",
        )
        figure = graph.to_plotly()
        figure.show()
        return figure

    def make_plot(self, show=False, pdf=False, graph_name="ErrorAnalysis.pdf"):
        """make plot summarizing the energy error per atom [meV/atom]
        the root mean square error of force [meV/Angstroem]
        and the root mean square error of the stress tensor in [kbar]
        """
        energy = Series(self.xaxis, self.energy_error[:], subplot=1)
        force = Series(self.xaxis, self.force_error[:], subplot=2)
        stress = Series(self.xaxis, self.stress_error[:], subplot=3)

        graph = Graph((energy, force, stress))
        graph.xlabel = ("configuration",) * 3
        graph.ylabel = (
            "error energy [meV/atom]",
            "rmse force [meV/Å]",
            "rmse stress [kbar]",
        )
        figure = graph.to_plotly()
        if pdf:
            figure.write_image(graph_name)
        if show:
            figure.show()
        return figure

    @staticmethod
    def prepare_output_array(x, y):
        """concatenate two input arrays of shape (N , ) to a 2d array of shape (N,2)
        that can be used by numpy.savetxt

        Parameters
        ----------
        x, y : ndarray
            numpy array of shape ( N ,  )

        Returns
        ----------
        data : ndarray
            numpy array of shape ( N , 2 ) containing x,y
        """
        xx = np.reshape(x, [x.shape[0], 1])
        yy = np.reshape(y, [y.shape[0], 1])
        data = np.concatenate((xx, yy), axis=1)
        return data

    @staticmethod
    def format_float(data):
        return "".join(["{:20.8f}".format(x) for x in data])

    def writing_energy_error_file(self, fname="EnergyError.out"):
        """writing the energy error to a output file the output file format will be
        structure index  | energy per atom in meV
        """
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Writing the energy error in meV/atom to file ", fname)
        print("the format is structure index vs error")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        data = self.prepare_output_array(self.xaxis, self.energy_error[:])
        with open(fname, "w") as outfile:
            outfile.write(
                f"# RMSE energy/atom in meV/atom {self.format_float([self.rmse_energy])}\n"
            )
            outfile.write(
                "#      structure index   energy difference in meV/atom "
                + "(value > 0 MLFF predicts too high value)\n"
            )
            for row in data:
                outfile.write(self.format_float(row) + "\n")

    def writing_force_error_file(self, fname="ForceError.out"):
        """writing the force error to a output file the output file format will be
        structure index | root mean square force error in meV/Angstroem
        """
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Writing the force error in meV/Angstroem per atom to file ", fname)
        print("the format is structure index vs error")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        data = self.prepare_output_array(self.xaxis, self.force_error[:])
        with open(fname, "w") as outfile:
            outfile.write(
                f"# RMSE force in meV/Angstroem {self.format_float([self.rmse_force])}\n"
            )
            outfile.write("#       structure index   RMSE in meV/Angstroem \n")
            for row in data:
                outfile.write(self.format_float(row) + "\n")

    def writing_stress_error_file(self, fname="StressError.out"):
        """writing the stress error to a output file the output file format will be
        structure index | root mean square stress error in [kBar]
        """
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Writing the stress error in [kBar] to file ", fname)
        print("the format is structure index vs error")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        data = self.prepare_output_array(self.xaxis, self.stress_error[:])
        with open(fname, "w") as outfile:
            outfile.write(
                f"# RMSE stress in kilo-bar {self.format_float([self.rmse_stress])}\n"
            )
            outfile.write("#      structure index      RMSE in kbar \n")
            for row in data:
                outfile.write(self.format_float(row) + "\n")

    def write_csv_output_file(self, fname="ErrorAnalysis.csv"):
        """write collected errors to a csv file"""
        data = {
            "energy error": self.energy_error[:],
            "force error": self.force_error[:],
            "stress error": self.stress_error[:],
        }
        dataframe = pd.DataFrame(data=data)
        dataframe.to_csv(fname)


def main():
    options = get_options(sys.argv[1:])
    x = AnalyseError(options.MLfiles, options.DFTfiles)
    if options.MakePlot or options.pdfplot:
        x.make_plot(options.MakePlot, options.pdfplot)
    if options.XYtextFile:
        x.writing_energy_error_file()
        x.writing_force_error_file()
        x.writing_stress_error_file()
    x.write_csv_output_file()

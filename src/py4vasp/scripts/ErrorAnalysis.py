#!/usr/bin/env python3


import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import py4vasp
from py4vasp._third_party.graph import Graph, Series


__author__ = "Jonathan Lahnsteiner"
__version__ = "1.0.0"
__maintainer__ = "Jonathan Lahnsteiner"
__email__ = "jonathan.lahnsteiner@vasp.at"
__status__ = "Production"


def getOptions(args=sys.argv[1:]):
    """
    argument parser to process command line arguments.
    for information start program with -h
    """
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument(
        "-dft",
        "--DFTfiles",
        nargs="+",
        help="Your input files of DFT calculations."
        + " Supply in a form as for example vaspout_{1..200}.h5",
    )
    parser.add_argument(
        "-ml",
        "--MLfiles",
        nargs="+",
        help="Your input files of ML calculations"
        + " Supply in a form as for example vaspout_{1..200}.h5",
    )
    parser.add_argument(
        "-plt",
        "--MakePlot",
        action="store_true",
        help="supply flag without keyword if you would like to generate a plot",
    )
    options = parser.parse_args(args)

    return options


class Reader:
    def __init__(self, fname):
        """
        reading a single vaspout.h5 file
        input:
             fname      -->  input filename
        output:
             self.fname     -->  file name of hdf5 file
             self.energy    -->  total energy computed by vasp
             self.force     -->  force array ( NIONS , 3 )
             self.lattice   -->  Bravais matrix for structure
                                 used for VASP calculation
             self.positions -->  atomic positions in direct coordinates
             self.NIONS     -->  number of ions in current structure
             self.stress    -->  stress tensor computed by vasp
                                 stored as 1d array xx, xy, xz, yy, yz, zz
        """
        self.fname = fname
        self.calc = py4vasp.Calculation.from_file(self.fname)
        self.read_energy()
        self.read_forces()
        self.read_stress()

    def read_energy(self):
        """
        read energy from vaspout.h5 file
        """
        energy = self.calc.energy.read()
        self.energy = energy["free energy    TOTEN"]

    def read_forces(self):
        """
        read from vaspout.h5 file the following arrays:
            forces
            positions
            NIONS
            lattice
            from vaspout.h5 file
        """
        force = self.calc.force.read()
        self.lattice = force["structure"]["lattice_vectors"]
        self.positions = force["structure"]["positions"]
        self.NIONS = self.positions.shape[0]
        self.force = force["forces"]

    def read_stress(self):
        """
        read stress tensor from vaspout.h5 file
        """
        stress = self.calc.stress.read()
        stress = stress["stress"]
        self.stress = []
        for i in range(stress.shape[0]):
            for j in range(i, stress.shape[1]):
                self.stress.append(stress[i, j])
        self.stress = np.asarray(self.stress)


class AnalyseErrorSingleFile:
    def __init__(self, MLFF, DFT):
        """
        input:
            MLFF  -->  OUTCAR file from machine learning calculation
            DFT   -->  OUTCAR file from DFT calculation
        creates:
            self.MLdata  --> creates class instance Reader for machine learning data
            self.DFTdata --> creates class instance Reader for DFT data
            for more information what these classes contain see ReadOUTCAR class
        """
        self.ml_fname = MLFF
        self.dft_fname = DFT

        print("Analysing errors between", self.ml_fname, " and ", self.dft_fname)

        ## reading input data with Reader class
        self.ml_data = Reader(self.ml_fname)
        self.dft_data = Reader(self.dft_fname)

        ## error checking if the two structures
        ## to compare are the same
        self.check_structures_for_eqivalence()

        ## computing errors of the data extracted
        ## from vaspout.h5 file
        self.compute_force_error()
        self.compute_energy_error_atom()
        self.compute_stress_error()

    def print_error(self):
        """
        printing an error window to the screen
        """
        print("########################################")
        print("########################################")
        print("##############   ERROR   ###############")
        print("##############   ERROR   ###############")
        print("##############   ERROR   ###############")
        print("########################################")
        print("########################################")

    @classmethod
    def write_array_to_screen(cls, data):
        """
        writing an array to the screen
        """
        for i in range(data.shape[0]):
            print(data[i, :])

    def check_structures_for_eqivalence(self):
        if self.ml_data.NIONS != self.dft_data.NIONS:
            self.print_error()
            print("Number of ions does not not match in ")
            print(self.ml_fname, " and ", self.dft_fname)
            print("stopping execution. Please check your input files")
            sys.exit()

        if not np.allclose(self.ml_data.positions, self.dft_data.positions, atol=1e-5):
            self.print_error()
            print("The positions of your input data do not match in ")
            print(self.ml_fname, " and ", self.dft_fname)
            print("stopping execution. Please check your input files")
            print("positions machine learning input ")
            self.write_array_to_screen(self.ml_data.positions)
            print("positions DFT input ")
            self.write_array_to_screen(self.dft_data.positions)
            sys.exit()

        if not np.allclose(self.ml_data.lattice, self.dft_data.lattice, atol=1e-5):
            self.print_error()
            print("The lattice of your input data does not match in ")
            print(self.ml_fname, " and ", self.dft_fname)
            print("stopping execution. Please check your input files")
            sys.exit()

    def compute_force_error(self):
        """
        compute the root mean square error per atom
        for the force array between the DFT and machine
        learning approach

        output:
            self.force_error        -->  root mean square error of force
                                         over x,y,z components and ions
        """
        error = np.sum((self.dft_data.force[:, :] - self.ml_data.force[:, :]) ** 2)
        self.force_error = np.sqrt(error / (self.ml_data.NIONS * 3))

    def compute_energy_error_atom(self):
        """
        computing the error per atom
        between machine learning and DFT
        error = ( E_{DFT}-E_{MLFF} ) / NIONS
        output:
            self.energy_error --> value of energy difference divided by number atoms
        """
        self.energy_error = (
            self.ml_data.energy - self.dft_data.energy
        ) / self.ml_data.NIONS

    def compute_stress_error(self):
        """
        computing the root mean square error of the stress tensor
        between DFT and machine learning calculation
        output:
             self.stress_error  --> root mean square error of stress tensor over its components
        """
        self.stress_error = np.sqrt(
            np.sum((self.ml_data.stress[:] - self.dft_data.stress[:]) ** 2)
            / self.ml_data.stress.shape[0]
        )

    @classmethod
    def root_mean_square_error_numpy_array(cls, data_A, data_B):
        """
        computing root mean square error between two data sets
        input:
             dataA   ->  numpy ndarray of any size and shape
             dataB   ->  numpy ndarray of size and shape matching dataA
        output:
             root mean square error between dataA and dataB
        """
        return np.sqrt(np.sum((data_A - data_B) ** 2) / (np.product(data_A.shape)))

    @classmethod
    def root_mean_square_error_list(cls, data_A, data_B):
        """
        computing root mean square errors between equally sized python
        lists containint numpy arrays
        input:
            data_A -> python list containing numpy arrays of equal size and
                      order as in the data_B array
            data_B -> python list containing numpy arrays which are
                      matching the numpy arrays conatained in data_A
        """
        if len(data_A) != len(data_B):
            print("Error in root_mean_square_error_list in ", type(self).__name__)
            print("the dimensions of the list do  not match ")

        error = 0
        number_elements = 0
        for i in range(len(data_A)):
            error += np.sum((data_A[i] - data_B[i]) ** 2)
            number_elements += np.product(data_A[i].shape)

        error = np.sqrt(error / number_elements)
        return error


class AnalyseError:
    def __init__(self, ML, Dft):
        """
        Input:
             ML  --> list of machine learning vaspout.h5 files
             Dft --> list of DFT vaspout.h5 files to compare MLFF
        contains:
             self.xaxis  -> equally spaced axis between indx_start and indx_end
                            self.indx_start  =  1
                            self.indx_end    =  len( ML )
                            with len( ML ) points
             self.energy_error -> look in self.ComputeErrors member function
             self.force_error  -> look in self.ComputeErrors member function
             self.stress_error -> look in self.ComputeErrors member function

             self.energy -> look in self.compute_errors member function
             self.force  -> look in self.compute_errors member function
             self.stress -> look in self.compute_errors member function
        """
        self.ml_files = ML
        self.dft_files = Dft
        if len(self.ml_files) != len(self.dft_files):
            print("Different number of files supplied analysis does not make sense")
            sys.exit()

        self.indx_start = 1
        self.indx_end = len(ML)

        # scaling factor to meV
        self.factor = 1000

        self.make_x_axis()

        self.compute_errors()
        self.compute_average_errors()
        self.print_average_errors()

    def make_x_axis(self):
        """
        create a x-axis for plots and output from
        self.IndxStart and self.IndxEnd
        the interval will be decomposed into
        number of supplied OUTCAR files segments
        """
        self.xaxis = np.linspace(self.indx_start, self.indx_end, len(self.ml_files))

    def compute_errors(self):
        """
        computing the errors in the data with the help
        of the class AnalyseErrorSingleFile

        output:
           self.energy_error --> 2d array of size (Nstructures , 1 )
                    self.energy_error[ : , 0 ] -> absolute error per atom between
                                            DFT and MLFF calculation, computed in
                                            AnalyseErrorSingleFile
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
           self.force_error --> 2d array of size ( Nstructures , 1 )
                    self.force_error[ : , 0 ] -> root mean square error in force between the
                                           DFT and MLFF calculation
                                           computed with AnalyseErrorSingleFile
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
           self.stress_error --> 2d array of size (Nstructures , 1 )
                    self.stress_error[ : , 0 ] -> root mean square error of stress tensor
                                            between a machine learning calculation and
                                            DFT calculation
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
           self.energy --> 2d array of size (Nstructures , 2 )
                        self.energy[ : , 0 ] -> DFT energies in [eV]
                        self.energy[ : , 1 ] -> MLFF energies in [eV]
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
           self.force --> python list of len 2
                        index 0 is the dft data
                        index 1 is the machine learning data
                        both indexes then contain python lists
                        of the len of reference configurations;
                        and in those numpy arrays are contained
                        NIONS,3
                                    list               list           numpy array
                        self.force[dft/ml][ reference configuration ][ NIONS , 3 ]
                        units are in eV/Angstroem
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
           self.stress --> 3d array of size ( Nstructures , 2 , size stress )
                        1st index -> structure index
                        2nd index refers to DFT ->  0 / MLFF -> 1
                        3rd index are components of stress tensor.
                        units are in kbar
        """
        self.force_error = np.zeros([len(self.ml_files), 1])
        self.energy_error = np.zeros([len(self.ml_files), 1])
        self.stress_error = np.zeros([len(self.ml_files), 1])

        ## extract NIONS
        x = AnalyseErrorSingleFile(self.ml_files[0], self.dft_files[0])
        self.energy = np.zeros([len(self.ml_files), 2])
        self.stress = np.zeros([len(self.ml_files), 2, x.ml_data.stress.shape[0]])
        ## force has to be a list to make variable atom numbers possible
        self.force = [[], []]

        for i in range(len(self.ml_files)):
            x = AnalyseErrorSingleFile(self.ml_files[i], self.dft_files[i])
            ## extract energies
            self.energy_error[i, 0] = x.energy_error * self.factor
            self.energy[i, 0] = x.dft_data.energy / x.dft_data.NIONS
            self.energy[i, 1] = x.ml_data.energy / x.ml_data.NIONS

            # extract forces
            self.force_error[i, 0] = x.force_error * self.factor
            self.force[0].append(x.dft_data.force)
            self.force[1].append(x.ml_data.force)

            # extract stress
            self.stress_error[i, 0] = x.stress_error
            self.stress[i, 0, :] = x.dft_data.stress
            self.stress[i, 1, :] = x.ml_data.stress

    def compute_average_errors(self):
        """
        computing single error values for energy, forces, stress
        over the whole data set
        self.rmse_energy  ->  is the root mean square error of the
                              energies computed over the supplied
                              configurations in [meV]
        self.rmse_force   ->  root mean square error of forces computed
                              over all configurations, atoms and x y and z
                              direction in [meV]/Angstroem
        self.rmse_stress  ->  root mean square error of stress tensor
                              computed over all configurations and
                              components of the tensor
        """
        self.rmse_energy = (
            AnalyseErrorSingleFile.root_mean_square_error_numpy_array(
                self.energy[:, 0], self.energy[:, 1]
            )
            * self.factor
        )

        self.rmse_force = (
            AnalyseErrorSingleFile.root_mean_square_error_list(
                self.force[:][0], self.force[:][1]
            )
            * self.factor
        )

        self.rmse_stress = AnalyseErrorSingleFile.root_mean_square_error_numpy_array(
            self.stress[:, 0, :], self.stress[:, 1, :]
        )

    def print_average_errors(self):
        """
        printing the average errors
        of energy, force and stress tensor in the
        test set to the screen
        """
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Root mean square error of energy in meV/atom", self.rmse_energy)
        print("Root mean square error of force in meV/Angstroem", self.rmse_force)
        print("Root mean square error of stress tensor in kbar", self.rmse_stress)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    @classmethod
    def get_axis_bound(cls, array):
        """
        input:
            1D input numpy array -> array
        returns:
            xmin value -> minV
            xmax value -> maxV
        can be used as bounds for an matplotlib axis
        """
        minV = np.min(array[:])
        maxV = np.max(array[:])
        minV -= maxV / 10
        maxV += maxV / 10
        return minV, maxV

    def plot_energy_error(self):
        """
        plotting energy error/atom with sign versus the
        input configurations
        """
        graph = Graph(
            series=Series(self.xaxis, self.energy_error[:, 0]),
            xlabel="configuration",
            ylabel="energy error [meV/atom]",
        )
        figure = graph.to_plotly()
        figure.show()
        return figure

    def plot_force_error(self):
        """
        plotting force rmse meV/Angstroem versus
        input configurations
        """
        graph = Graph(
            series=Series(self.xaxis, self.force_error[:, 0]),
            xlabel="configuration",
            ylabel="rmse force [meV/Å]$",
        )
        figure = graph.to_plotly()
        figure.show()
        return figure

    def plot_stress_error(self):
        """
        plotting force rmse meV/Angstroem versus
        input configurations
        """
        graph = Graph(
            series=Series(self.xaxis, self.stress_error[:, 0]),
            xlabel="configuration",
            ylabel="rmse stress [kbar]",
        )
        figure = graph.to_plotly()
        figure.show()
        return figure

    def make_plot(self):
        """
        make plot summarizing the energy error per atom [meV/atom]
        the root mean square error of force [meV/Angstroem]
        and the root mean square error of the stress tensor in [kbar]
        """
        energy = Series(self.xaxis, self.energy_error[:, 0], subplot=1)
        force = Series(self.xaxis, self.force_error[:, 0], subplot=2)
        stress = Series(self.xaxis, self.stress_error[:, 0], subplot=3)

        graph = Graph((energy, force, stress))
        graph.xlabel = ("configuration",) * 3
        graph.ylabel = (
            "error energy [meV/atom]",
            "root mean square error force [meV/Å]",
            "root mean square error stress [kbar]",
        )
        figure = graph.to_plotly()
        figure.show()
        return figure

    @classmethod
    def prepare_output_array(cls, x, y):
        """
        concatenate two input arrays of shape
        (Nstruc , ) to a 2d array of shape (Nstruc,2) that can be used
        by numpy.savetxt
        input:
            x -->  numpy array of shape ( Nstruc ,  )
            y -->  numpy array of shape ( Nstruc ,  )
        output:
            data --> numpy array of shape ( Nstruc , 2 ) containing x,y
        """
        xx = np.reshape(x, [x.shape[0], 1])
        yy = np.reshape(y, [y.shape[0], 1])
        data = np.concatenate((xx, yy), axis=1)
        return data

    @classmethod
    def format_float(cls, data):
        return "".join(["{:20.8f}".format(x) for x in data])

    def writing_energy_error_file(self, fname="EnergyError.out"):
        """
        writing the energy error to a output file
        the output file format will be
            structure index  | energy per atom in meV
        """
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Writing the energy error in meV/atom to file ", fname)
        print("the format is structure index vs error")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        data = self.prepare_output_array(self.xaxis, self.energy_error[:, 0])
        with open(fname, "w") as outfile:
            outfile.write(
                "# RMSE energy/atom in meV/atom "
                + self.format_float([self.rmse_energy])
                + "\n"
            )
            outfile.write(
                "#      structure index "
                + "  energy difference in meV/atom "
                + "(value > 0 MLFF predicts too high value)"
                + "\n"
            )
            for i in range(data.shape[0]):
                outfile.write(self.format_float(data[i, :]) + "\n")

    def writing_force_error_file(self, fname="ForceError.out"):
        """
        writing the force error to a output file
        the output file format will be
            structure index | root mean square force error in meV/Angstroem
        """
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Writing the force error in meV/Angstroem per atom to file ", fname)
        print("the format is structure index vs error")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        data = self.prepare_output_array(self.xaxis, self.force_error[:, 0])
        with open(fname, "w") as outfile:
            outfile.write(
                "# RMSE force in meV/Angstroem "
                + self.format_float([self.rmse_force])
                + "\n"
            )
            outfile.write(
                "#       structure index " + "  RMSE in meV/Angstroem " + "\n"
            )
            for i in range(data.shape[0]):
                outfile.write(self.format_float(data[i, :]) + "\n")

    def writing_stress_error_file(self, fname="StressError.out"):
        """
        writing the stress error to a output file
        the output file format will be
            structure index | root mean square stress error in [kBar]
        """
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Writing the stress error in [kBar] to file ", fname)
        print("the format is structure index vs error")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        data = self.prepare_output_array(self.xaxis, self.stress_error[:, 0])
        with open(fname, "w") as outfile:
            outfile.write(
                "# RMSE stress in kilo-bar "
                + self.format_float([self.rmse_stress])
                + "\n"
            )
            outfile.write("#      structure index " + "     RMSE in kbar " + "\n")
            for i in range(data.shape[0]):
                outfile.write(self.format_float(data[i, :]) + "\n")


def main(argv):
    options = getOptions(sys.argv[1:])
    x = AnalyseError(options.MLfiles, options.DFTfiles)
    if options.MakePlot:
        x.make_plot()
    x.writing_energy_error_file()
    x.writing_force_error_file()
    x.writing_stress_error_file()


if __name__ == "__main__":
    main(sys.argv)

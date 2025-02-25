from ase.calculators.calculator import Calculator, all_changes
import ase.units as unit
from ase.atoms import Atoms
import numpy as np
import pycalci.calcicpp as pycal


class LennardJones(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, atoms: Atoms = None, **kwargs):
        """
        Lennard-Jones calculator implementation
        Args:
            atoms (Atoms, optional):
                Atoms object. Defaults to None.
            n_atoms (int, optional):
                Number of atoms. Needs to be divisible by 3.
                Can be specified instead of n_atoms. Defaults to None.
        """

        default_parameters = {
            "epsilon": 1.0,
            "sigma": 1.0,
            "rc": None,
            "ro": None,
            "smooth": False,
        }

        default_parameters.update(kwargs)

        Calculator.__init__(self, **default_parameters)

        if self.parameters.rc is None:
            self.parameters.rc = 3 * self.parameters.sigma

        if self.parameters.ro is None:
            self.parameters.ro = 0.66 * self.parameters.rc

        if atoms is None:
            raise Exception("Specify an Atoms object")

        self.lj = pycal.LennardJones(
            self.parameters.sigma,
            self.parameters.epsilon,
            self.parameters.rc,
            self.parameters.ro,
        )

        # Read in the information from the atoms object
        if not atoms is None:
            self.update_system_from_atoms_object(atoms)

        # Set up the results dict
        self.results = dict(energy=None, forces=None, dipole=None)

    def update_system_from_atoms_object(self, atoms: Atoms):
        lattice = np.diagonal(np.array(atoms.get_cell()))
        pbc = np.array(atoms.get_pbc(), dtype=bool)
        box = pycal.SimulationBoxInfo(lattice, pbc)
        self.lj.box = box

    def calculate(
        self,
        atoms,
        properties=["energy", "forces"],
        system_changes=all_changes,
    ):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.update_system_from_atoms_object(atoms)

        self.forces = np.zeros((len(atoms), 3))

        self.lj.recompute_neighbour_lists(atoms.get_positions())
        self.energy = self.lj.energy_and_forces(atoms.get_positions(), self.forces)

        self.results["energy"] = self.energy
        self.results["forces"] = self.forces
        self.results["virial"] = self.lj.virial

        self.forces_all = self.lj.compute_virial(np.array(atoms.get_positions()))
        self.results["virial_general"] = self.lj.virial_general

        temperature = atoms.get_temperature()
        kBT = unit.kB * temperature
        pressure = self.get_pressure(atoms, kBT)

        self.results["pressure"] = pressure  # in eV/A^3

    def get_pressure(self, atoms, kBT, ndims=3):
        """ "
        params:
            atoms : the atoms object
            kBT : k_B * T in units of eV (should be consistent with the energy in the calculator, usually eV)
            ndims : dimensionality of the system (default 3)
        returns:
            pressure in eV/Angstrom^3
        """

        volume = atoms.get_volume()
        dof = atoms.get_number_of_degrees_of_freedom()

        pressure = (dof / ndims) * kBT / (volume) + 1.0 / (
            volume * ndims
        ) * self.results["virial"]
        return pressure

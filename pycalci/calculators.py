from ase.calculators.calculator import Calculator, all_changes
import ase.units as unit
from ase.atoms import Atoms
import numpy as np
import pycalci.calcicpp as pycal


class SCMECalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        atoms: Atoms = None,
        **kwargs,
    ):
        """
        Lennard-Jones calculator implementation
        Args:
            atoms (Atoms, optional):
                Atoms object. Defaults to None.
            n_atoms (int, optional):
                Number of atoms. Needs to be divisible by 3.
                Can be specified instead of n_atoms. Defaults to None.
            unit_length (float, optional):
                Value of the internally used length unit in angstroem.
                The required value depends on the units of the parameters.
                The coordinates of the atoms and the cell lengths are divided by this value
                before they are passed to the SCME core.
                The returned forces are divided by it too.
                Defaults to unit.Bohr.
            unit_energy (float, optional):
                Value of the internally used energy unit in eV.
                The required value depends on the units of the parameters.
                The energy and forces returned by the SCME core are multiplied by this value.
                Defaults to unit.Hartree.
        """
        implemented_properties = ['energy', 'forces']
        default_parameters = {
        'epsilon': 1.0,
        'sigma': 1.0,
        'rc': None,
        'ro': None,
        'smooth': False,
    }

        Calculator.__init__(self)
        
        if self.parameters.rc is None:
            self.parameters.rc = 3 * self.parameters.sigma

        if self.parameters.ro is None:
            self.parameters.ro = 0.66 * self.parameters.rc

        if atoms is None:
            raise Exception("Specify an Atoms object")

        self.lj = pycal.LennardJones(self.parameters.sigma, self.parameters.epsilon, self.parameters.rc, self.parameters.ro)

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

        self.forces = np.zeros(len(atoms),3)

        self.energy = self.lj.energy_and_forces(atoms.get_positions(), self.forces)

        self.results["energy"] = self.energy
        self.results["forces"] = self.forces

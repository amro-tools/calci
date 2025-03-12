from util import finite_difference
from ase.io import read, write
from ase.units import fs
from ase.md.verlet import VelocityVerlet
from ase.optimize.fire2 import FIRE2
import numpy as np
from pathlib import Path

from pycalci.calculators import LennardJones
from pycalci import find_ghost_atoms

from ase.units import kB

# Liquid argon parameters (Rahman 1964)
para_dict = {
    "epsilon": 120 * kB,
    "sigma": 3.4,
    "rc": 5.0,
    "ro": None,
    "smooth": False,
}


def test_ase_calculator():
    input_file_path = Path(__file__).parent / "resources/fcc_small.xyz"

    # Read the system using ASE
    with open(input_file_path, "r") as f:
        system = read(f, format="extxyz")

    system.calc = LennardJones(atoms=system, **para_dict)

    # # Check that everything was read in
    assert len(system) == 32

    system.center()

    assert np.all(
        np.isclose(
            np.diagonal(system.cell),
            system.calc.lj.box.get_lattice(),
        )
    )

    assert np.all(system.calc.lj.box.pbc == system.get_pbc())

    wrapped_positions, ghost_atoms_shell, idx_original = find_ghost_atoms(
        5, system.calc.lj.box, system.positions
    )

    print(f"{len(wrapped_positions) = }")
    print(f"{len(ghost_atoms_shell) = }")

    # Test forces against finite difference
    def get_energy_and_forces_ase(pos):
        system.set_positions(pos)
        system.calc.calculate(system)
        forces = system.get_forces()
        energy = system.get_potential_energy()
        return energy, forces

    pos = system.get_positions()

    def test_forces_and_virial():
        energy_ase, forces_ase = get_energy_and_forces_ase(pos)
        forces_fd = -finite_difference(
            lambda p: get_energy_and_forces_ase(p)[0], pos, epsilon=1e-7
        )

        print(f"{energy_ase = } eV")
        max_force_diff = np.max(np.abs(forces_ase - forces_fd))
        print(f"{max_force_diff = }")

        assert np.all(np.isclose(forces_fd, forces_ase, atol=1e-7))

        virial_pairwise = system.calc.results["virial_pairwise"]
        print(f"virial (pairwise) contribution={virial_pairwise} eV")

        # Check that the virial matches with pairwise virial
        virial_general = system.calc.results["virial"]
        print(f"virial contribution using general formulation={virial_general} eV")
        print(f" {virial_pairwise / virial_general = } ")

        assert np.isclose(virial_general, virial_pairwise)

        return energy_ase, forces_ase

    test_forces_and_virial()
    pos += 1e-2 * np.random.uniform(size=pos.shape)
    test_forces_and_virial()
    pos += 1e-2 * np.random.uniform(size=pos.shape)
    test_forces_and_virial()
    pos += 2e-2 * np.random.uniform(size=pos.shape)
    test_forces_and_virial()

if __name__ == "__main__":
    test_ase_calculator()

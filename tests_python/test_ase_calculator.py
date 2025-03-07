from util import finite_difference
from ase.io import read, write
from ase.units import fs
from ase.md.verlet import VelocityVerlet
from ase.optimize.fire2 import FIRE2
import numpy as np
from pathlib import Path

from pycalci.calculators import LennardJones

from ase.units import kB

# Liquid argon parameters (Rahman 1964)
para_dict = {
    "epsilon": 120 * kB,
    "sigma": 3.4,
    "rc": 10,
    "ro": None,
    "smooth": False,
}


def test_ase_calculator():
    input_file_path = Path(__file__).parent / "resources/system.xyz"

    # Read the system using ASE
    with open(input_file_path, "r") as f:
        system = read(f, format="extxyz")

    system.set_pbc([True, True, False])
    system.calc = LennardJones(atoms=system, **para_dict)

    # Check that everything was read in
    assert len(system) == 48

    assert np.all(
        np.isclose(
            np.diagonal(system.cell),
            system.calc.lj.box.get_lattice(),
        )
    )

    assert np.all(system.calc.lj.box.pbc == system.get_pbc())

    # Test forces against finite difference
    def get_energy_and_forces_ase(pos):
        system.set_positions(pos)
        system.calc.calculate(system)
        forces = system.get_forces()
        energy = system.get_potential_energy()
        return energy, forces

    pos = system.get_positions()

    def test_forces():
        energy_ase, forces_ase = get_energy_and_forces_ase(pos)
        forces_fd = -finite_difference(
            lambda p: get_energy_and_forces_ase(p)[0], pos, epsilon=1e-7
        )

        print(f"{energy_ase = } eV")
        max_force_diff = np.max(np.abs(forces_ase - forces_fd))
        print(f"{max_force_diff = }")

        assert np.all(np.isclose(forces_fd, forces_ase, atol=1e-7))

    # test forces for different random displacements as well

    test_forces()
    pos += 1e-2 * np.random.uniform(size=pos.shape)
    test_forces()
    pos += 1e-1 * np.random.uniform(size=pos.shape)
    test_forces()
    pos += 2e-1 * np.random.uniform(size=pos.shape)
    test_forces()

    print(f"virial contribution={system.calc.results["virial"]} eV")
    assert system.calc.results["virial"] != 0


def test_ase_calculator_parameter_map():
    input_file_path = Path(__file__).parent / "resources/system.xyz"

    # Read the system using ASE
    with open(input_file_path, "r") as f:
        system = read(f, format="extxyz")

    system.set_pbc([True, True, False])

    n_atoms = len(system)

    type_ids = np.zeros(n_atoms, dtype=int)

    type_ids[1] = 1
    parameter_map = {(0, 1): (0.0, 1.0)}

    system.calc = LennardJones(
        atoms=system, type_ids=type_ids, parameter_map=parameter_map, **para_dict
    )

    # Check that everything was read in
    assert len(system) == 48

    assert np.all(
        np.isclose(
            np.diagonal(system.cell),
            system.calc.lj.box.get_lattice(),
        )
    )

    assert np.all(system.calc.lj.box.pbc == system.get_pbc())

    # Test forces against finite difference
    def get_energy_and_forces_ase(pos):
        system.set_positions(pos)
        system.calc.calculate(system)
        forces = system.get_forces()
        energy = system.get_potential_energy()
        return energy, forces

    pos = system.get_positions()

    def test_forces():
        energy_ase, forces_ase = get_energy_and_forces_ase(pos)
        forces_fd = -finite_difference(
            lambda p: get_energy_and_forces_ase(p)[0], pos, epsilon=1e-7
        )

        print(f"{energy_ase = } eV")
        max_force_diff = np.max(np.abs(forces_ase - forces_fd))
        print(f"{max_force_diff = }")

        assert np.all(np.isclose(forces_fd, forces_ase, atol=1e-7))

    # test forces for different random displacements as well

    test_forces()
    pos += 1e-2 * np.random.uniform(size=pos.shape)
    test_forces()
    pos += 1e-1 * np.random.uniform(size=pos.shape)
    test_forces()
    pos += 2e-1 * np.random.uniform(size=pos.shape)
    test_forces()

    print(f"virial contribution={system.calc.results["virial"]} eV")
    assert system.calc.results["virial"] != 0


if __name__ == "__main__":
    test_ase_calculator()

from util import finite_difference
from ase.io import read, write
from ase.units import fs
from ase.md.verlet import VelocityVerlet
from ase.optimize.fire2 import FIRE2
from ase import Atoms
import numpy as np
from pathlib import Path

from pycalci import find_ghost_atoms, SimulationBoxInfo

from ase.units import kB


def test_ghost_atoms():
    input_file_path = Path(__file__).parent / "resources/lj.xyz"

    # Read the system using ASE
    with open(input_file_path, "r") as f:
        system = read(f, format="extxyz")

    # Check that everything was read in
    assert len(system) == 3

    rc = 3.0  # Cutoff for creating an open boundary system with a shell around the local cell of thickness rc

    lattice = system.cell.cellpar()[:3]
    pbc = system.get_pbc()
    box = SimulationBoxInfo(lattice, pbc)

    positions = np.array(system.positions)

    wrapped_positions, ghost_atoms = find_ghost_atoms(rc, box, positions)

    print(f"{wrapped_positions = }")
    print(f"{ghost_atoms = }")

    # Expected wrapped positions
    wrapped_positions_exp = np.array(
        [[5.0, 5.0, 5.0], [9.8, 4.8, 4.9], [1.0, 5.0, 5.0]]
    )
    ghost_atoms_exp = np.array([[-0.2, 4.8, 4.9], [11.0, 5.0, 5.0]])

    assert np.all(np.isclose(wrapped_positions, wrapped_positions_exp))
    assert np.all(np.isclose(ghost_atoms, ghost_atoms_exp))


def test_ghost_atoms_fcc():
    input_file_path = Path(__file__).parent / "resources/fcc_center.xyz"

    # Read the system using ASE
    with open(input_file_path, "r") as f:
        system = read(f, format="extxyz")

    system.center()

    # Check that everything was read in
    assert len(system) == 864

    lattice = system.cell.cellpar()[:3]

    lx, ly, lz = lattice

    rc = 3.0

    pbc = system.get_pbc()
    box = SimulationBoxInfo(lattice, pbc)

    # needs to be a cubic system
    assert np.isclose(lx, ly)
    assert np.isclose(lx, lz)

    original_pos = np.array(system.positions)

    wrapped_positions, ghost_atoms = find_ghost_atoms(rc, box, original_pos)

    n_ghost_atoms_calci = len(ghost_atoms)
    print(f"{n_ghost_atoms_calci = }")

    system = system.repeat([3, 3, 3])
    system.center()

    write("fcc_rep.xyz", system)

    new_pos = np.array(system.positions)

    mask1 = (
        np.linalg.norm(new_pos - system.get_center_of_mass(), axis=1, ord=np.inf)
        < lx / 2 + rc
    )

    n_ghost_atoms_np = np.sum(mask1) - len(original_pos)

    print(f"{n_ghost_atoms_np = }")


if __name__ == "__main__":
    test_ghost_atoms_fcc()

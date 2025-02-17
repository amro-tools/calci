from util import finite_difference
import numpy as np
from pycalci import LennardJones, SimulationBoxInfo


def test_finite_difference():
    a, b, c = 5, 5, 5

    # create a regular grid of atoms
    positions = []
    for i in range(a):
        for j in range(b):
            for k in range(c):
                positions.append(np.array([i, j, k], dtype=float))

    positions = np.array(positions)
    # add some noise
    positions += 0.01 * np.random.uniform(size=positions.shape)

    forces = np.zeros(positions.shape)

    sigma = 1.0
    epsilon = 1.0
    rc = 10.0
    ro = 10.0

    box = SimulationBoxInfo()
    box.pbc = [True, False, False]
    LJ = LennardJones(sigma, epsilon, rc, ro)
    LJ.box = box

    def energy_and_force(positions):
        energy = LJ.energy_and_forces(np.array(positions), forces)
        return energy, forces

    energy_lj, force_lj = energy_and_force(positions)
    force_fd = -finite_difference(
        lambda pos: energy_and_force(pos)[0], x=positions, epsilon=1e-8
    )

    max_diff = np.max(np.abs(force_lj - force_fd))

    print(energy_lj)
    print(f"{max_diff = }")

    assert np.all(np.isclose(force_lj, force_fd, atol=1e-5))


if __name__ == "__main__":
    test_finite_difference()

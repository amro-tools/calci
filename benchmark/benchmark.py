import numpy as np
import time

import pycalci
from pycalci import LennardJones, SimulationBoxInfo


def benchmark():

    a, b, c = 16, 16, 16

    N_FORCECALLS = 100
    THREADS_LIST = range(1,65)

    SIGMA = 1.0
    EPSILON = 1.0
    RC = 10.0
    RO = 10.0
    PBC = [True, True, True]
    BOXDIMS = [a+5, b+5, c+5]

    # create a regular grid of atoms
    positions = []
    for i in range(a):
        for j in range(b):
            for k in range(c):
                positions.append(np.array([i, j, k], dtype=float))

    positions = np.array(positions)
    # add some noise
    positions += 0.001 * np.random.uniform(size=positions.shape)

    forces = np.zeros(positions.shape)

    box = SimulationBoxInfo()
    box.set_lattice(BOXDIMS)
    box.pbc = PBC
    LJ = LennardJones(SIGMA, EPSILON, RC, RO)
    LJ.box = box

    def energy_and_force(positions):
        energy = LJ.energy_and_forces(np.array(positions), forces)
        return energy, forces


    time_list = []

    for threads in THREADS_LIST:
        pycalci.set_num_threads(threads)
        print(f"Testing with {threads} threads")
        t_start = time.time()
        for _ in range(N_FORCECALLS):
            energy_lj, force_lj = energy_and_force(positions)
        t_end = time.time()

        time_list.append(t_end - t_start)

    np.savetxt("timings.txt", np.vstack( [THREADS_LIST, time_list] ).T )


if __name__ == "__main__":
    benchmark()
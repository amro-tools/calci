import numpy as np
import time

from ase import Atoms
from ase.calculators.lj import LennardJones as ASELennardJones
import pycalci
from pycalci.calculators import LennardJones as CalciLennardJones


def benchmark():

    a, b, c = 8, 8, 8

    N_FORCECALLS_ASE = 10
    N_FORCECALLS = 1000

    THREADS_LIST = range(1, 7)

    SIGMA = 1.0
    EPSILON = 1.0
    RC = 3000.0
    RO = 10.0
    PBC = [False, False, False]
    BOXDIMS = [a + 5, b + 5, c + 5]

    # create a regular grid of atoms
    positions = []
    for i in range(a):
        for j in range(b):
            for k in range(c):
                positions.append(np.array([i, j, k], dtype=float))

    positions = np.array(positions)
    # add some noise
    rng = np.random.default_rng(0)
    positions += 0.001 * rng.uniform(size=positions.shape)

    calci_atoms = Atoms(
        positions=positions,
        numbers=np.full(len(positions), 18),
        cell=BOXDIMS,
        pbc=PBC,
    )
    calci_cached_lj = CalciLennardJones(
        sigma=SIGMA, epsilon=EPSILON, rc=RC, ro=RO, smooth=False
    )
    calci_rebuild_lj = CalciLennardJones(
        sigma=SIGMA, epsilon=EPSILON, rc=RC, ro=RO, smooth=False
    )
    # A non-positive skin depth forces recompute_neighbour_lists() to rebuild
    # the list. Zero keeps the neighbour-list cutoff equal to RC.
    calci_rebuild_lj.lj.verlet_skin_depth = 0.0

    def calci_energy_and_forces(calculator):
        # Calling calculate directly avoids ASE returning cached results.
        calculator.calculate(calci_atoms)
        return calculator.results["energy"], calculator.results["forces"]

    ase_atoms = Atoms(positions=positions, numbers=np.full(len(positions), 18))
    ase_atoms.set_cell(BOXDIMS)
    ase_atoms.set_pbc(PBC)
    ase_lj = ASELennardJones(sigma=SIGMA, epsilon=EPSILON, rc=RC, ro=RO, smooth=False)

    def ase_energy_and_forces():
        # Calling calculate directly avoids ASE returning cached results.
        ase_lj.calculate(ase_atoms)
        return ase_lj.results["energy"], ase_lj.results["forces"]

    calci_energy_and_forces(calci_cached_lj)
    calci_energy_and_forces(calci_rebuild_lj)
    ase_energy_and_forces()

    print("Testing ASE baseline")
    t_start = time.perf_counter()
    for _ in range(N_FORCECALLS_ASE):
        ase_energy_and_forces()
    ase_time = time.perf_counter() - t_start
    ase_time /= N_FORCECALLS_ASE

    calci_cached_times = []
    calci_rebuild_times = []

    for threads in THREADS_LIST:
        pycalci.set_num_threads(threads)
        print(f"Testing cached neighbour list with {threads} threads")
        calci_energy_and_forces(calci_cached_lj)
        t_start = time.perf_counter()
        for _ in range(N_FORCECALLS):
            calci_energy_and_forces(calci_cached_lj)
        calci_cached_times.append((time.perf_counter() - t_start) / N_FORCECALLS)

        print(f"Testing rebuilt neighbour list with {threads} threads")
        calci_energy_and_forces(calci_rebuild_lj)
        t_start = time.perf_counter()
        for _ in range(N_FORCECALLS):
            calci_energy_and_forces(calci_rebuild_lj)
        calci_rebuild_times.append((time.perf_counter() - t_start) / N_FORCECALLS)

    np.savetxt(
        "timings.txt",
        np.column_stack(
            [
                THREADS_LIST,
                calci_cached_times,
                calci_rebuild_times,
                np.full(len(calci_cached_times), ase_time),
            ]
        ),
        header="threads calci_cached_seconds calci_rebuild_seconds ase_seconds",
    )


if __name__ == "__main__":
    benchmark()

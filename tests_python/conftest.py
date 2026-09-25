import pytest
from pycalci.calculators import LennardJones
from ase.units import kB
from ase import Atoms
from ase.io import read
from pathlib import Path


@pytest.fixture
def ar_fcc() -> Atoms:
    p = Path(__file__).parent / "resources/fcc_Ar.xyz"
    atoms = read(p)
    return atoms

@pytest.fixture
def LJ_liquid_argon(ar_fcc : Atoms) -> Atoms:

    # Liquid argon parameters (Rahman 1964)
    para_dict = {
        "epsilon": 120 * kB,
        "sigma": 3.4,
        "rc": 5.0,
        "ro": None,
        "smooth": False,
        "compute_general_virial": True,
    }

    ar_fcc.calc = LennardJones(atoms=ar_fcc, **para_dict)
    return ar_fcc

@pytest.fixture
def ar_fcc_with_h_atoms() -> Atoms:
    p = Path(__file__).parent / "resources/fcc_Ar_with_H.xyz"
    atoms = read(p)
    return atoms

@pytest.fixture
def LJ_liquid_argon_with_H_satellites(ar_fcc_with_h_atoms : Atoms) -> Atoms:

    # Liquid argon parameters (Rahman 1964)
    para_dict = {
        "epsilon": 120 * kB,
        "sigma": 3.4,
        "rc": 5.0,
        "ro": None,
        "smooth": False,
        "compute_general_virial": True,
    }

    type_ids = ar_fcc_with_h_atoms.numbers

    parameter_map = {
        (type_ids[0], type_ids[1]): (0.0, 1.0), # Ar-H interaction
        (type_ids[1], type_ids[1]): (0.0, 1.0), # H-H interaction
        (type_ids[0], type_ids[0]): (para_dict["epsilon"], para_dict["sigma"]),
    } # set interactions between Ar and H to zero

    ar_fcc_with_h_atoms.calc = LennardJones(atoms=ar_fcc_with_h_atoms, **para_dict, type_ids=type_ids, parameter_map=parameter_map)
    return ar_fcc_with_h_atoms

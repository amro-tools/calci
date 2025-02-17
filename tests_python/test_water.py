from ase.io import read, write, Trajectory
from ase.units import fs
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.constraints import FixBondLengths
from pathlib import Path
import pycalci as pyca

def test_calculator():

    assert 2 == 2


test_calculator()

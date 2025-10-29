import sys

sys.path.insert(1, "./xdave")

from unit_conversions import (
    amu_TO_kg,
    eV_TO_K,
    K_TO_eV,
    per_cm3_TO_per_m3,
    kg_per_m3_TO_g_per_cm3,
    per_A_TO_per_aB,
    per_m3_TO_per_cm3,
)
from constants import BOLTZMANN_CONSTANT, VACUUM_PERMITTIVITY, BOHR_RADIUS, ELEMENTARY_CHARGE
from plasma_state import PlasmaState
from static_sf import OCPStaticStructureFactor, MCPStaticStructureFactor

from xdave import xDave
from models import ModelOptions

import numpy as np
import matplotlib.pyplot as plt
import os


THIS_DIR = os.path.dirname(__file__)


if __name__ == "__main__":
    pass

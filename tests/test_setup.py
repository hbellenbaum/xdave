import sys

sys.path.insert(1, "./xdave")
sys.path.insert(1, "./mcss_tests")


from plasma_state import PlasmaState, get_rho_T_from_rs_theta, get_fractions_from_Z, get_rho_T_from_rs_theta_SI
from models import ModelOptions
from unit_conversions import *
from constants import BOHR_RADIUS, PLANCK_CONSTANT, ELECTRON_MASS
from freefree_dsf import FreeFreeDSF
from boundfree_dsf import BoundFreeDSF
from utils import calculate_angle, calculate_q, load_itcf_from_file, load_mcss_result
from xdave import xDave

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


def test_full_spectrum():
    return


if __name__ == "__main__":
    test_full_spectrum()

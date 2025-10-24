import sys

sys.path.insert(1, "/home/bellen85/code/dev/xdave/xdave")

from unit_conversions import per_cm3_TO_per_m3, J_TO_eV, J_TO_Ryd
from constants import BOLTZMANN_CONSTANT, VACUUM_PERMITTIVITY, BOHR_RADIUS, ELEMENTARY_CHARGE
from plasma_state import PlasmaState
from static_sf import OCPStaticStructureFactor, MCPStaticStructureFactor

from potentials import *
from xdave import xDave
from models import ModelOptions

import numpy as np
import matplotlib.pyplot as plt
import os


def test_ii_potentials():
    n = 8192
    r0 = 0.5e-1 * BOHR_RADIUS  # [m]
    rf = 1.0e2 * BOHR_RADIUS  # [m]
    dr = (rf - r0) / n
    dk = np.pi / (n * dr)  # [1/m] as it should be [1/m],
    kf = r0 + n * dk
    rs = np.linspace(r0, rf, n)  # [m]
    ks = np.linspace(r0, kf, n)  # [1/m]

    T = 1.0e5  # K
    ni = 1.23e23 * per_cm3_TO_per_m3  # cm^-3
    Zi = 2

    beta = 1 / (BOLTZMANN_CONSTANT * T)

    Rii = (3 / (4 * np.pi * ni)) ** (1 / 3)
    alpha = 1 / Rii
    lambda_ii = 1.0e2 * Rii

    yukawa_test_r = yukawa_r(Qa=Zi, Qb=Zi, r=rs, alpha=alpha) * J_TO_Ryd
    deutsch_test_r = deutsch_r(Qa=Zi, Qb=Zi, r=rs, alpha=alpha) * J_TO_Ryd
    kelbg_test_r = kelbg_r(Qa=Zi, Qb=Zi, r=rs, alpha=alpha) * J_TO_Ryd

    fig, ax = plt.subplots(1, 1)
    # ax = axes[0]
    ax.plot(rs / BOHR_RADIUS, yukawa_test_r, label="Yukawa", ls="-.", c="navy")
    ax.plot(rs / BOHR_RADIUS, deutsch_test_r, label="Deutsch", ls="-.", c="crimson")
    ax.plot(rs / BOHR_RADIUS, kelbg_test_r, label="Kelb", ls="-.", c="orange")
    ax.legend()
    ax.set_ylim(-0.5, 10)
    ax.set_xlabel(r"$r$ [$a_B$]")
    ax.set_ylabel(r"$V_{ab}(r)$ [eV]")
    plt.tight_layout()
    # ax = axes[1]
    # ax.plot(ks * BOHR_RADIUS, yukawa_test_k, label="Yukawa", ls="-.", c="navy")
    # ax.plot(ks * BOHR_RADIUS, deutsch_test_k, label="Deutsch", ls="-.", c="crimson")
    # ax.plot(ks * BOHR_RADIUS, kelbg_test_k, label="Kelbg", ls="-.", c="orange")
    # ax.legend()

    plt.show()


def test_ei_potentials():
    return


def test_ee_potentials():
    return


if __name__ == "__main__":
    test_ii_potentials()

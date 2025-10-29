import sys

sys.path.insert(1, "./xdave")

from xdave import *
from screening_cloud import ScreeningCloud

import numpy as np
import matplotlib.pyplot as plt
import os


THIS_DIR = os.path.dirname(__file__)


def test():
    T = 4 * eV_TO_K
    Zi = 2
    rho = 498.16  # kg/m^3
    state = PlasmaState(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_state=Zi,
        atomic_mass=2,
        atomic_number=2,
        binding_energies=None,
    )

    k = np.linspace(1.0e-1 / BOHR_RADIUS, 10 / BOHR_RADIUS, 200)

    sigma_c = 2.15 * BOHR_RADIUS

    kernel = ScreeningCloud(state=state, overlord_state=state)
    f_fws = kernel.get_screening_cloud(
        k=k, ion_core_radius=sigma_c, screening_model="FINITE_WAVELENGTH", ei_potential="COULOMB"
    )
    f_dh = kernel.get_screening_cloud(
        k=k, ion_core_radius=sigma_c, screening_model="DEBYE_HUCKEL", ei_potential="COULOMB"
    )

    plt.figure()
    plt.plot(k * BOHR_RADIUS, f_fws, label="FWS")
    plt.plot(k * BOHR_RADIUS, f_dh, label="DH")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test()

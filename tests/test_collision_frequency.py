import sys

# sys.path.insert(1, "./xdave")

from xdave.constants import BOHR_RADIUS
from xdave.unit_conversions import g_per_cm3_TO_kg_per_m3, eV_TO_K, eV_TO_J, RYDBERG_TO_eV, J_TO_eV
from xdave.plasma_state import PlasmaState, get_rho_T_from_rs_theta
from xdave.models import ModelOptions
from xdave.freefree_dsf import FreeFreeDSF
from xdave.collision_frequency import CollisionFrequency
from xdave.constants import *
from xdave.utils import calculate_q
from xdave.plasma_state import get_fractions_from_Z

from xdave import *

import numpy as np
import matplotlib.pyplot as plt


def test():
    rho = 1.85
    T = 12
    Zi = 2.45
    angle = 40

    models = ModelOptions()

    Z1, Z2, x1, x2 = get_fractions_from_Z(Z=Zi)

    kernel = xDave(
        models=models,
        mass_density=rho,
        electron_temperature=T,
        ion_temperature=T,
        elements=np.array(["Al", "Al"]),
        charge_states=np.array([Z1, Z2]),
        partial_densities=np.array([x1, x2]),
    )

    state = kernel.overlord_state

    plasma_freq = state.plasma_frequency(-1, state.free_electron_number_density, ELECTRON_MASS)
    EF = state.fermi_energy(state.free_electron_number_density, ELECTRON_MASS)

    freq = CollisionFrequency(state=state)

    w = np.linspace(-200, 200, 1000) * eV_TO_J
    k = calculate_q(angle=angle, energy=8.0e3) / BOHR_RADIUS
    lfc = 0.0
    omega, mu_ei, mu_ei_interp = freq.get(k=k, w=w, lfc=lfc, model="BORN")
    mu_ei_ziman = freq.get(k=k, w=w, lfc=lfc, model="ZIMAN")
    print(mu_ei_ziman)
    # print(omega)
    # print(mu_ei)

    # plt.figure()
    fig, ax = plt.subplots(1, 1)
    # ax = axes[0]
    ax.plot(omega * J_TO_eV, mu_ei.real * DIRAC_CONSTANT / EF, label="real", c="navy", ls="-.")
    # ax.axhline(mu_ei_ziman / plasma_freq, label="Ziman", c="purple")
    ax.plot(omega * J_TO_eV, mu_ei.imag * DIRAC_CONSTANT / EF, label="imag", c="crimson")
    # ax.legend()
    # ax = axes[1]
    ax.plot(w * J_TO_eV, mu_ei_interp.real * DIRAC_CONSTANT / EF, label="interp real", c="dodgerblue", ls=":")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test()

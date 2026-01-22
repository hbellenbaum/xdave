from xdave.constants import BOHR_RADIUS
from xdave.unit_conversions import g_per_cm3_TO_kg_per_m3, eV_TO_K, eV_TO_J, RYDBERG_TO_eV, J_TO_eV
from xdave.plasma_state import PlasmaState, get_rho_T_from_rs_theta
from xdave.models import ModelOptions
from xdave.freefree_dsf import FreeFreeDSF

from xdave.constants import *
from xdave.utils import calculate_q
from xdave.plasma_state import get_fractions_from_Z

from xdave import *

import numpy as np
import matplotlib.pyplot as plt
import os


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

    ff_kernel = FreeFreeDSF(state=kernel.overlord_state)

    w = np.linspace(-100, 100, 1000) * eV_TO_J
    k = calculate_q(angle=angle, energy=8.0e3) / BOHR_RADIUS
    lfc = 0.0
    mu_ei_born = ff_kernel.get_collision_frequency(k=k, w=w, lfc=lfc, model="BORN")
    mu_ei_ziman = ff_kernel.get_collision_frequency(k=k, w=w, lfc=lfc, model="ZIMAN")
    # print(mu_ei_ziman)
    # print(mu_ei_born)

    fig, ax = plt.subplots(1, 1)
    ax.set_xscale("log")
    ax.plot(w * J_TO_eV, mu_ei_born.real, label="Re[Born]", c="navy", ls="-.")  #  * DIRAC_CONSTANT / EF
    # ax.axhline(mu_ei_ziman, label="Ziman", c="gray", ls="-.")
    # ax.axhline(plasma_freq, label=r"$\omega_p$", c="black", ls="-.")
    ax.plot(w * J_TO_eV, mu_ei_born.imag, label="Im[Born]", c="crimson", ls="-.")  #  * DIRAC_CONSTANT / EF

    ax.set_xlim(1.0e-6, 100)
    ax.legend()
    plt.tight_layout()
    plt.show()


def test_fortmann_2010_Fig1():
    T = 0  # K
    Zi = 1
    rss = np.array([1, 5])

    fn = os.path.join(os.path.dirname(__file__), "comparison_data/collision_frequency/")

    dat1_re = np.genfromtxt(fn + f"Fortmann_2010_Fig1_rs_1_re.csv", delimiter=",")
    dat1_im = np.genfromtxt(fn + f"Fortmann_2010_Fig1_rs_1_im.csv", delimiter=",")
    dat2_re = np.genfromtxt(fn + f"Fortmann_2010_Fig1_rs_5_re.csv", delimiter=",")
    dat2_im = np.genfromtxt(fn + f"Fortmann_2010_Fig1_rs_5_im.csv", delimiter=",")

    def fermi_energy(rs):
        ne = 3 / FOUR_PI * 1 / (BOHR_RADIUS * rs) ** 3
        kF = (3 * PI_SQR * ne) ** (1 / 2)
        EF = DIRAC_CONSTANT_SQR * kF**2 / (2 * ELECTRON_MASS)
        return EF

    norm1 = 1  # DIRAC_CONSTANT / fermi_energy(1)
    norm5 = 1  # DIRAC_CONSTANT / fermi_energy(5)

    rs1 = rss[0]
    rho1, T1 = get_rho_T_from_rs_theta(rs=rs1, theta=1)

    state1 = PlasmaState(
        electron_temperature=T1 * eV_TO_K,
        ion_temperature=T1 * eV_TO_K,
        mass_density=rho1 * g_per_cm3_TO_kg_per_m3,
        charge_state=1.0,
        atomic_mass=1,
        atomic_number=1,
        binding_energies=None,
    )
    ff_kernel = FreeFreeDSF(state=state1)
    w = np.linspace(0, 200, 500) * eV_TO_J
    angle = 75
    k = calculate_q(angle=angle, energy=8.0e3) / BOHR_RADIUS
    born_mu_ei = ff_kernel.get_collision_frequency(k=k, w=w, lfc=0.0, model="BORN")

    plt.figure()
    plt.xscale("log")
    # plt.scatter(dat1_re[:, 0] * norm1, dat1_re[:, 1] * norm1, label=f"rs=1, Re", marker="x", c="navy")
    # plt.scatter(dat1_im[:, 0] * norm1, dat1_im[:, 1] * norm1, label=f"rs=1, Im", marker="<", c="navy")
    plt.plot(w * J_TO_eV, born_mu_ei.real, label=f"xDave rs=1, Re", c="navy", ls=":")
    plt.plot(w * J_TO_eV, born_mu_ei.imag, label=f"xDave rs=1, Im", c="navy", ls="-.")
    # plt.scatter(dat2_re[:, 0] * norm5, dat2_re[:, 1] * norm5, label=f"rs=5, Re", marker="x", c="crimson")
    # plt.scatter(dat2_im[:, 0] * norm5, dat2_im[:, 1] * norm5, label=f"rs=5, Im", marker="<", c="crimson")
    plt.legend()
    plt.xlabel(r"$\omega$ [$E_F/\hbar$]")
    plt.ylabel(r"$\nu_{ei}$ [$E_F/\hbar$]")
    plt.show()


if __name__ == "__main__":
    # test_fortmann_2010_Fig1()
    test()

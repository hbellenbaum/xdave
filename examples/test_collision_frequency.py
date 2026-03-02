from xdave.constants import BOHR_RADIUS
from xdave.unit_conversions import g_per_cm3_TO_kg_per_m3, eV_TO_K, eV_TO_J, RYDBERG_TO_eV, J_TO_eV, per_cm3_TO_per_m3
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

    w = np.linspace(0, 500, 1000) * eV_TO_J
    k = calculate_q(angle=angle, energy=8.0e3) / BOHR_RADIUS
    lfc = 0.0
    mu_ei_born = ff_kernel.get_collision_frequency(k=k, w=w, lfc=lfc, model="BORN")
    mu_ei_ziman = ff_kernel.get_collision_frequency(k=k, w=w, lfc=lfc, model="ZIMAN")
    mu_ei_full = ff_kernel._born_ei_collision_frequency(k=k, w=w, lfc=lfc)

    EF = state.fermi_energy(state.free_electron_number_density, mass=ELECTRON_MASS)
    norm_factor = 1 / plasma_freq  # DIRAC_CONSTANT / EF
    print(norm_factor)
    w_freq = w / DIRAC_CONSTANT

    fig, ax = plt.subplots(1, 1)
    ax.set_xscale("log")
    ax.plot(w * J_TO_eV, mu_ei_full.real * norm_factor, label="Born real")
    ax.plot(w * J_TO_eV, mu_ei_full.imag * norm_factor, label="Born imag")
    # ax.axhline(mu_ei_ziman * norm_factor, label="Ziman", c="black")
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.set_ylabel(r"$\upsilon$ [$\omega_p$]")
    ax.legend()
    plt.tight_layout()
    plt.show()


def test_fortmann_2010_Fig1():
    T = 1  # eV
    Zi = 1
    rss = np.array([1, 5])

    fn = os.path.join(os.path.dirname(__file__), "comparison_data/collision_frequency/")

    dat1_re = np.genfromtxt(fn + f"Fortmann_2010_Fig1_rs_1_re.csv", delimiter=",")
    dat1_im = np.genfromtxt(fn + f"Fortmann_2010_Fig1_rs_1_im.csv", delimiter=",")
    dat2_re = np.genfromtxt(fn + f"Fortmann_2010_Fig1_rs_5_re.csv", delimiter=",")
    dat2_im = np.genfromtxt(fn + f"Fortmann_2010_Fig1_rs_5_im.csv", delimiter=",")

    w = np.linspace(0.01, 500, 1000) * eV_TO_J
    w_freq = w / DIRAC_CONSTANT

    rs1 = 1
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
    angle = 75
    k = calculate_q(angle=angle, energy=8.0e3) / BOHR_RADIUS
    born_mu_ei = ff_kernel.get_collision_frequency(k=k, w=w, lfc=0.0, model="BORN")
    w_plasma = state1.plasma_frequency(
        charge=state1.charge_state, number_density=state1.ion_number_density, mass=state1.atomic_mass
    )
    EF = state1.fermi_energy(state1.ion_number_density, state1.atomic_mass)
    omega_F = state1.fermi_frequency(state1.free_electron_number_density, ELECTRON_MASS)

    rs5 = 5
    rho5, T1 = get_rho_T_from_rs_theta(rs=rs5, theta=1)

    state5 = PlasmaState(
        electron_temperature=T1 * eV_TO_K,
        ion_temperature=T1 * eV_TO_K,
        mass_density=rho5 * g_per_cm3_TO_kg_per_m3,
        charge_state=1.0,
        atomic_mass=1,
        atomic_number=1,
        binding_energies=None,
    )
    ff_kernel5 = FreeFreeDSF(state=state5)
    born_mu_ei5 = ff_kernel5.get_collision_frequency(k=k, w=w, lfc=0.0, model="BORN")
    omega_F5 = state5.fermi_frequency(state5.free_electron_number_density, ELECTRON_MASS)
    EF5 = state5.fermi_energy(state5.free_electron_number_density, ELECTRON_MASS)

    plt.figure()
    plt.xscale("log")
    # plt.scatter(w1_real, dat1_re[:, 1], label=f"rs=1, Re", marker="x", c="navy")
    # plt.scatter(dat1_im[:, 0], dat1_im[:, 1], label=f"rs=1, Im", marker="<", c="navy")
    plt.plot(w_freq * DIRAC_CONSTANT / EF, born_mu_ei.real, label=f"xDave rs=1, Re", c="navy", ls=":")
    plt.plot(w_freq * DIRAC_CONSTANT / EF, born_mu_ei.imag, label=f"xDave rs=1, Im", c="navy", ls="-.")
    # plt.scatter(dat2_re[:, 0], dat2_re[:, 1], label=f"rs=5, Re", marker="x", c="crimson")
    # plt.scatter(dat2_im[:, 0], dat2_im[:, 1], label=f"rs=5, Im", marker="<", c="crimson")
    plt.plot(w_freq * DIRAC_CONSTANT / EF5, born_mu_ei5.real, label=f"xDave rs=5, Re", c="crimson", ls=":")
    plt.plot(w_freq * DIRAC_CONSTANT / EF5, born_mu_ei5.imag, label=f"xDave rs=5, Im", c="crimson", ls="-.")
    plt.legend()
    plt.xlabel(r"$\omega$ [eV]")
    plt.ylabel(r"$\nu_{ei}$ [$s^{-1}$]")
    # plt.xlabel(r"$\omega$ [$E_F/\hbar$]")
    # plt.ylabel(r"$\nu_{ei}$ [$E_F/\hbar$]")
    plt.show()


def test_hoell_2007_Fig5():
    fn = os.path.join(os.path.dirname(__file__), "comparison_data/collision_frequency/")

    T = 12 * eV_TO_K  # eV
    nes = np.array([1, 2.5, 5.0, 7.5, 10]) * 1.0e21 * 1.0e6  # 1/m^3
    # nes *= per_cm3_TO_per_m3

    w = np.linspace(0.01, 100, 500) * eV_TO_J
    angle = 75
    k = calculate_q(angle=angle, energy=8.0e3) / BOHR_RADIUS

    # plt.figure()
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    colors = ["black", "red", "green", "blue", "orange"]
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")

    for ne, c in zip(nes, colors):
        # ne *= per_cm3_TO_per_m3
        print(f"ne = {ne * 1.e-21 / per_cm3_TO_per_m3} 1.e21 1/cc")
        mass_density = ATOMIC_MASS_UNIT * ne
        # print(f"rho = {mass_density / g_per_cm3_TO_kg_per_m3}")
        state = PlasmaState(
            electron_temperature=T,
            ion_temperature=T,
            electron_number_density=ne,
            ion_number_density=ne,
            charge_state=1,
            mass_density=mass_density,
            atomic_mass=1,
            atomic_number=1,
            binding_energies=None,
        )
        w_freq = w / DIRAC_CONSTANT
        w_pe = np.sqrt(
            state.free_electron_number_density * ELEMENTARY_CHARGE_SQR / (VACUUM_PERMITTIVITY * ELECTRON_MASS)
        )
        mu_ei_born = FreeFreeDSF(state=state).get_collision_frequency(k=k, w=w, lfc=0.0, model="BORN")
        axes[0].plot(w_freq / w_pe, mu_ei_born.real * 10, c=c, label=f"ne={ne}")
        axes[1].plot(w_freq / w_pe, mu_ei_born.imag * 10, c=c, label=f"ne={ne}")

        dat1 = np.genfromtxt(
            os.path.join(fn, f"Hoell_2007_Fig5a_ne_{ne / (per_cm3_TO_per_m3 * 1.e21):.1f}.csv"), delimiter=","
        )
        axes[0].scatter(dat1[:, 0], dat1[:, 1] * 1.0e14, label=f"Hoell: ne={ne}", c=c)

        dat2 = np.genfromtxt(
            os.path.join(fn, f"Hoell_2007_Fig5b_ne_{ne / (per_cm3_TO_per_m3 * 1.e21):.1f}.csv"), delimiter=","
        )
        axes[1].scatter(dat2[:, 0], dat2[:, 1] * 1.0e14, label=f"Hoell: ne={ne}", c=c)

    axes[0].legend()
    plt.show()


if __name__ == "__main__":
    # test_hoell_2007_Fig5()
    test_fortmann_2010_Fig1()
    # test()

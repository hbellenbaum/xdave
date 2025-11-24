import sys

sys.path.insert(1, "./mcss_tests")

from xdave import *
from xdave.screening_cloud import ScreeningCloud
from xdave.constants import BOHR_RADIUS

from run_mcss_sim import run_ch_ar_mode

import numpy as np
import matplotlib.pyplot as plt
import os


THIS_DIR = os.path.dirname(__file__)


def test_Fig1c():

    T1 = 10  # eV
    T2 = 1.0e3  # eV
    # T = 4 * eV_TO_K
    Zi = 2
    ZH = 1
    ZC = 4
    rho = 5.84  # kg/m^3

    models = ModelOptions(screening_model="FINITE_WAVELENGTH", ee_potential="YUKAWA", ei_potential="YUKAWA")
    elements = np.array(["H", "C"])
    partial_densities = np.array([0.5, 0.5])
    charge_states = np.array([ZH, ZC])

    fn1 = "tests/comparison_data/screening/Chapman_NatCommun_2015_Fig1c_FWS_T_10.csv"
    dat1 = np.genfromtxt(fn1, delimiter=",")

    kernel1 = xDave(
        mass_density=rho,
        electron_temperature=T1,
        ion_temperature=T1,
        elements=elements,
        partial_densities=partial_densities,
        charge_states=charge_states,
        models=models,
    )

    k = np.linspace(1.0e-1 / BOHR_RADIUS, 10 / BOHR_RADIUS, 200)

    screening_C = ScreeningCloud(state=kernel1.states[1], overlord_state=kernel1.overlord_state)
    f_fws_C = screening_C.get_screening_cloud(
        k=k, screening_model="FINITE_WAVELENGTH", ei_potential="YUKAWA", ee_potential="YUKAWA"
    )
    f_dh_C = screening_C.get_screening_cloud(
        k=k, screening_model="DEBYE_HUCKEL", ei_potential="YUKAWA", ee_potential="YUKAWA"
    )

    screening_H = ScreeningCloud(state=kernel1.states[0], overlord_state=kernel1.overlord_state)
    f_fws_H = screening_H.get_screening_cloud(
        k=k, screening_model="FINITE_WAVELENGTH", ei_potential="YUKAWA", ee_potential="YUKAWA"
    )
    f_dh_H = screening_H.get_screening_cloud(
        k=k, screening_model="DEBYE_HUCKEL", ei_potential="YUKAWA", ee_potential="YUKAWA"
    )

    f_tot_fws = partial_densities[0] * f_fws_H / ZH + partial_densities[1] * f_fws_C / ZC

    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    ax.set_title(f"T={T1} eV")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.plot(k * BOHR_RADIUS, f_fws_C / ZC, label="C: FWS", ls="solid", c="orange")
    ax.plot(k * BOHR_RADIUS, f_fws_H / ZH, label="H: FWS", ls="solid", c="limegreen")
    ax.plot(k * BOHR_RADIUS, f_tot_fws, label="Tot", ls="solid", c="magenta")
    ax.scatter(dat1[:, 0], dat1[:, 1], label="Chapman 2015", marker="x", c="crimson")
    ax.plot(k * BOHR_RADIUS, f_dh_C / ZC, label="C: DH", c="navy")
    ax.legend()

    fn2 = "tests/comparison_data/screening/Chapman_NatCommun_2015_Fig1c_FWS_T_1000.csv"
    dat2 = np.genfromtxt(fn2, delimiter=",")

    kernel2 = xDave(
        mass_density=rho,
        electron_temperature=T2,
        ion_temperature=T2,
        elements=elements,
        partial_densities=partial_densities,
        charge_states=charge_states,
        models=models,
    )

    k = np.linspace(1.0e-1 / BOHR_RADIUS, 10 / BOHR_RADIUS, 200)

    screening_C = ScreeningCloud(state=kernel2.states[1], overlord_state=kernel2.overlord_state)
    f_fws_C = screening_C.get_screening_cloud(k=k, screening_model="FINITE_WAVELENGTH", ei_potential="YUKAWA")
    f_dh_C = screening_C.get_screening_cloud(k=k, screening_model="DEBYE_HUCKEL", ei_potential="YUKAWA")

    screening_H = ScreeningCloud(state=kernel2.states[0], overlord_state=kernel2.overlord_state)
    f_fws_H = screening_H.get_screening_cloud(k=k, screening_model="FINITE_WAVELENGTH", ei_potential="YUKAWA")
    f_dh_H = screening_H.get_screening_cloud(k=k, screening_model="DEBYE_HUCKEL", ei_potential="YUKAWA")

    ax = axes[1]
    ax.set_title(f"T={T2} eV")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.plot(k * BOHR_RADIUS, f_fws_C / ZC, label="C: FWS", ls="solid", c="orange")
    ax.plot(k * BOHR_RADIUS, f_fws_H / ZH, label="H: FWS", ls="solid", c="limegreen")
    ax.scatter(dat2[:, 0], dat2[:, 1], label="Chapman 2015", marker="x", c="crimson")
    ax.plot(k * BOHR_RADIUS, f_dh_C / ZC, label="C: DH", c="navy")
    ax.legend()

    plt.show()


def test_Fig1b():
    fn_C = "tests/comparison_data/screening/Chapman_NatCommun_2015_Fig1b_q_FWS_C.csv"
    dat_C = np.genfromtxt(fn_C, delimiter=",")
    fn_H = "tests/comparison_data/screening/Chapman_NatCommun_2015_Fig1b_q_FWS_H.csv"
    dat_H = np.genfromtxt(fn_H, delimiter=",")

    ZH = 1
    ZC = 4
    T = 10.0  # eV
    rho = 5.84  # 5.84

    models = ModelOptions(
        polarisation_model="NUMERICAL",
        bf_model="SCHUMACHER",
        lfc_model="NONE",
        ipd_model="NONE",
        ee_potential="COULOMB",
        ei_potential="YUKAWA",
        ii_potential="YUKAWA",
    )

    elements = np.array(["H", "C"])
    partial_densities = np.array([0.5, 0.5])
    charge_states = np.array([ZH, ZC])

    mcss_result = run_ch_ar_mode(T=T, rho=rho, ZH=ZH, ZC=ZC, xH=partial_densities[0], angle=10)

    kernel = xDave(
        mass_density=rho,
        electron_temperature=T,
        ion_temperature=T,
        elements=elements,
        partial_densities=partial_densities,
        charge_states=charge_states,
        models=models,
    )

    ei_potential_model = models.ei_potential
    ee_potential_model = models.ee_potential

    k = np.linspace(1.0e-2 / BOHR_RADIUS, 10 / BOHR_RADIUS, 1000)

    screening_H = ScreeningCloud(state=kernel.states[0], overlord_state=kernel.overlord_state)
    f_fws_H = screening_H.get_screening_cloud(
        k=k, screening_model="FINITE_WAVELENGTH", ei_potential=ei_potential_model, ee_potential=ee_potential_model
    )
    f_dh_H = screening_H.get_screening_cloud(
        k=k, screening_model="DEBYE_HUCKEL", ei_potential=ei_potential_model, ee_potential=ee_potential_model
    )

    screening_C = ScreeningCloud(state=kernel.states[1], overlord_state=kernel.overlord_state)
    f_fws_C = screening_C.get_screening_cloud(
        k=k, screening_model="FINITE_WAVELENGTH", ei_potential=ei_potential_model, ee_potential=ee_potential_model
    )
    f_dh_C = screening_C.get_screening_cloud(
        k=k, screening_model="DEBYE_HUCKEL", ei_potential=ei_potential_model, ee_potential=ee_potential_model
    )

    fig, ax = plt.subplots(1, 1)
    ax.plot(k * BOHR_RADIUS, f_fws_C, label="C: FWS", ls="solid", c="orange")
    # ax.plot(k * BOHR_RADIUS, f_dh_C, label="C: DH", ls="-.", c="orange")
    # ax.plot(k * BOHR_RADIUS, f_dh_H, label="H: DH", ls="-.", c="limegreen")
    ax.plot(k * BOHR_RADIUS, f_fws_H, label="H: FWS", ls="solid", c="limegreen")
    # ax.plot(mcss_result[0], mcss_result[5], label="MCSS C: FWS", ls="solid", c="crimson")
    # ax.plot(mcss_result[0], mcss_result[4], label="MCSS H: FWS", ls="solid", c="darkgreen")
    ax.scatter(dat_C[:, 0], dat_C[:, 1], label="Chapman 2015: C", marker="x", c="crimson")
    ax.scatter(dat_H[:, 0], dat_H[:, 1], label="Chapman 2015: H", marker="x", c="darkgreen")
    ax.legend()
    ax.set_xlabel(r"$k$ [$a_B^{-1}$]")
    ax.set_ylabel(r"$q_x(k)$ [#]")
    plt.show()


if __name__ == "__main__":
    test_Fig1b()
    # test_Fig1c()

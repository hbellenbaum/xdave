from xdave import *
from xdave.screening_cloud import ScreeningCloud
from xdave.constants import BOHR_RADIUS

import numpy as np
import matplotlib.pyplot as plt
import os


THIS_DIR = os.path.dirname(__file__)


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
        ei_potential="COULOMB",
        ii_potential="YUKAWA",
    )

    elements = np.array(["H", "C"])
    partial_densities = np.array([0.5, 0.5])
    charge_states = np.array([ZH, ZC])

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
    ax.plot(k * BOHR_RADIUS, f_dh_C, label="C: DH", ls="solid", c="magenta")
    ax.plot(k * BOHR_RADIUS, f_dh_H, label="H: DH", ls="solid", c="cornflowerblue")
    ax.plot(k * BOHR_RADIUS, f_fws_H, label="H: FWS", ls="solid", c="limegreen")
    ax.scatter(dat_C[:, 0], dat_C[:, 1], label="Chapman 2015: C", marker="x", c="crimson")
    ax.scatter(dat_H[:, 0], dat_H[:, 1], label="Chapman 2015: H", marker="x", c="darkgreen")
    ax.legend()
    ax.set_xlabel(r"$k$ [$a_B^{-1}$]")
    ax.set_ylabel(r"$q_x(k)$ [#]")
    plt.show()

    atol = 1.0e-1
    if not np.isclose(f_fws_C, np.interp(x=k * BOHR_RADIUS, xp=dat_C[:, 0], fp=dat_C[:, 1]), atol=atol).all():
        print(f"FWD Test failed for Carbon.")
    if not np.isclose(f_fws_H, np.interp(x=k * BOHR_RADIUS, xp=dat_H[:, 0], fp=dat_H[:, 1]), atol=atol).all():
        print(f"FWD Test failed for Hydrogen.")


def test_version():
    return


if __name__ == "__main__":
    test_Fig1b()

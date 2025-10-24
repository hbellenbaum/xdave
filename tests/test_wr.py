import sys

sys.path.insert(1, "/home/bellen85/code/dev/xdave/xdave")

from unit_conversions import ang_TO_m, eV_TO_K, g_per_cm3_TO_kg_per_m3, per_aB_TO_per_A
from constants import BOLTZMANN_CONSTANT, VACUUM_PERMITTIVITY, BOHR_RADIUS, ELEMENTARY_CHARGE
from plasma_state import PlasmaState
from rayleigh_weight import OCPRayleighWeight, MCPRayleighWeight

from xdave import xDave
from models import ModelOptions

import numpy as np
import matplotlib.pyplot as plt
import os

THIS_DIR = os.path.dirname(__file__)


def test_Fig615a():

    T = 12  # * eV_TO_K  # K
    Z = 2
    AN = 4

    ion_core_radius = 1 * ang_TO_m
    rho = 1.845  # * g_per_cm3_TO_kg_per_m3

    models = ModelOptions()
    elements = np.array(["Be"])
    charge_states = np.array([2])
    partial_densities = np.array([1.0])
    code_kernel = xDave(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        elements=elements,
        partial_densities=partial_densities,
        charge_states=charge_states,
        models=models,
    )

    kernel = OCPRayleighWeight(
        overlord_state=code_kernel.overlord_state, state=code_kernel.states[0], ion_core_radius=ion_core_radius
    )

    ks = np.linspace(0.1, 20, 1000) / BOHR_RADIUS

    WR_noscreening = kernel.get_rayleigh_weight(
        k=ks,
        sf_model="HNC",
        ii_potential="YUKAWA",
        ee_potential="YUKAWA",
        ei_potential="YUKAWA",
        screening_model="NONE",
        bridge_function=None,
        return_full=False,
    )

    fn = os.path.join(THIS_DIR, "comparison_data/wr/Wunesch_Thesis/Fig6-15a_coulomb.csv")
    dat_coulomb = np.genfromtxt(fn, delimiter=",")
    fn = os.path.join(THIS_DIR, "comparison_data/wr/Wunesch_Thesis/Fig6-15a_empty-core.csv")
    dat_emptycore = np.genfromtxt(fn, delimiter=",")
    fn = os.path.join(THIS_DIR, "comparison_data/wr/Wunesch_Thesis/Fig6-15a_no-screening.csv")
    dat_noscreening = np.genfromtxt(fn, delimiter=",")
    fn = os.path.join(THIS_DIR, "comparison_data/wr/Wunesch_Thesis/Fig6-15a_soft-cutoff2.csv")
    dat_soft = np.genfromtxt(fn, delimiter=",")

    plt.figure()
    plt.scatter(dat_coulomb[:, 0], dat_coulomb[:, 1], label="Coulomb", c="black", marker="*")
    plt.plot(ks * BOHR_RADIUS * per_aB_TO_per_A, WR_noscreening)
    plt.scatter(dat_emptycore[:, 0], dat_emptycore[:, 1], label="Empty core", c="yellow", marker="<")
    plt.scatter(dat_noscreening[:, 0], dat_noscreening[:, 1], label="No screening", c="red", marker=">")
    plt.scatter(dat_soft[:, 0], dat_soft[:, 1], label="Soft", c="green", marker="x")
    plt.legend()
    plt.tight_layout()
    plt.show()


def test_Fig616a():

    fn = os.path.join(THIS_DIR, "comparison_data/wr/Wunesch_Thesis/Fig6-16a_CC.csv")
    dat_CC = np.genfromtxt(fn, delimiter=",")
    fn = os.path.join(THIS_DIR, "comparison_data/wr/Wunesch_Thesis/Fig6-16a_HH.csv")
    dat_HH = np.genfromtxt(fn, delimiter=",")
    fn = os.path.join(THIS_DIR, "comparison_data/wr/Wunesch_Thesis/Fig6-16a_CH.csv")
    dat_CH = np.genfromtxt(fn, delimiter=",")
    fn = os.path.join(THIS_DIR, "comparison_data/wr/Wunesch_Thesis/Fig6-16a_CH_tot.csv")
    dat_tot = np.genfromtxt(fn, delimiter=",")

    plt.figure()
    plt.scatter(dat_CC[:, 0], dat_CC[:, 1], label="CC", c="blue", marker="*")
    plt.scatter(dat_HH[:, 0], dat_HH[:, 1], label="HH", c="black", marker="<")
    plt.scatter(dat_CH[:, 0], dat_CH[:, 1], label="CH", c="green", marker=">")
    plt.scatter(dat_tot[:, 0], dat_tot[:, 1], label="Tot", c="red", marker="x")
    plt.legend()
    plt.tight_layout()
    plt.show()


def test_Fig617a():

    fn = os.path.join(THIS_DIR, "comparison_data/wr/Wunesch_Thesis/Fig6-17a_C.csv")
    dat_C = np.genfromtxt(fn, delimiter=",")
    fn = os.path.join(THIS_DIR, "comparison_data/wr/Wunesch_Thesis/Fig6-17a_CH.csv")
    dat_CH = np.genfromtxt(fn, delimiter=",")
    fn = os.path.join(THIS_DIR, "comparison_data/wr/Wunesch_Thesis/Fig6-17a_CH2.csv")
    dat_CH2 = np.genfromtxt(fn, delimiter=",")

    plt.figure()
    plt.scatter(dat_C[:, 0], dat_C[:, 1], label="C", c="black", marker="*")
    plt.scatter(dat_CH[:, 0], dat_CH[:, 1], label="CH", c="red", marker="<")
    plt.scatter(dat_CH2[:, 0], dat_CH2[:, 1], label="CH2", c="blue", marker=">")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_Fig615a()
    # test_Fig616a()
    # test_Fig617a()

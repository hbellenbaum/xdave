from xdave.unit_conversions import (
    ang_TO_m,
    per_aB_TO_per_A,
    per_cm3_TO_per_m3,
    amu_TO_kg,
    kg_per_m3_TO_g_per_cm3,
    per_aB_TO_per_A,
)
from xdave.constants import BOHR_RADIUS
from xdave.rayleigh_weight import OCPRayleighWeight, MCPRayleighWeight

from xdave.xdave import xDave
from xdave.models import ModelOptions

import numpy as np
import matplotlib.pyplot as plt
import os

THIS_DIR = os.path.dirname(__file__)


def test_Fig615a():
    """I need different tests, since this one uses a form of the short-range repulsion potential that is wrong."""

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

    kernel = OCPRayleighWeight(overlord_state=code_kernel.overlord_state, state=code_kernel.states[0])

    ks = np.linspace(0.1, 20, 1000) / BOHR_RADIUS

    WR_noscreening = kernel.get_rayleigh_weight(
        k=ks,
        sf_model="HNC",
        ii_potential="YUKAWA",
        ee_potential="COULOMB",
        ei_potential="YUKAWA",
        screening_model="NONE",
        hnc_mix_fraction=0.9,
        hnc_delta=1.0e-12,
        hnc_max_iterations=1000,
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
    plt.scatter(dat_emptycore[:, 0], dat_emptycore[:, 1], label="Empty core", c="yellow", marker="<")
    plt.scatter(dat_noscreening[:, 0], dat_noscreening[:, 1], label="No screening", c="red", marker=">")
    plt.scatter(dat_soft[:, 0], dat_soft[:, 1], label="Soft", c="green", marker="x")
    plt.plot(ks * BOHR_RADIUS * per_aB_TO_per_A, WR_noscreening, label="xDave", c="red")
    plt.legend()
    plt.tight_layout()
    plt.show()


def test_Fig616a():
    """
    Comparing to Figure 6.16 (a) in K. Wunesch, PhD Thesis (2011).
    Uses a slightly different ion-ion potential in the HNC solver, electronic screening is unclear and form factors are obtained from DFT, so
    I would not expect perfect agreement.
    Using this to test the general trends more than anything.
    """

    nC = 5.0e22  # per cc
    nH = nC

    nC_SI = nC * per_cm3_TO_per_m3
    NH_SI = nC_SI

    amu_H = 1.0078 * amu_TO_kg
    amu_C = 12.011 * amu_TO_kg
    rho = amu_H * NH_SI + amu_C * nC_SI
    rho *= kg_per_m3_TO_g_per_cm3 * 1
    ZH = 1
    ZC = 2

    T = 8  # eV

    xH = 0.5
    xC = 0.5

    models = ModelOptions(
        sf_model="HNC",
        ee_potential="COULOMB",
        ei_potential="COULOMB",
        ii_potential="DEBYE_HUCKEL",
        screening_model="FINITE_WAVELENGTH",
    )

    kernel = xDave(
        mass_density=rho,
        electron_temperature=T,
        ion_temperature=T,
        elements=np.array(["H", "C"]),
        partial_densities=np.array([xH, xC]),
        charge_states=np.array([ZH, ZC]),
        models=models,
    )

    k = np.linspace(0.1, 10, 1000) / BOHR_RADIUS

    wr_kernel = MCPRayleighWeight(overlord_state=kernel.overlord_state, states=kernel.states, verbose=True)
    k_xdave, Sab, WR, qs, fs = wr_kernel.get_rayleigh_weight(
        k=k,
        lfc=0.0,
        sf_model=models.sf_model,
        ii_potential=models.ii_potential,
        ee_potential=models.ee_potential,
        ei_potential=models.ei_potential,
        screening_model=models.screening_model,
        hnc_max_iterations=1000,
        hnc_delta=1.0e-10,
        hnc_mix_fraction=0.9,
        return_full=True,
    )

    k_xdave *= BOHR_RADIUS * per_aB_TO_per_A

    fn = os.path.join(THIS_DIR, "comparison_data/wr/Wunesch_Thesis/Fig6-16a_CC.csv")
    dat_CC = np.genfromtxt(fn, delimiter=",")
    fn = os.path.join(THIS_DIR, "comparison_data/wr/Wunesch_Thesis/Fig6-16a_HH.csv")
    dat_HH = np.genfromtxt(fn, delimiter=",")
    fn = os.path.join(THIS_DIR, "comparison_data/wr/Wunesch_Thesis/Fig6-16a_CH.csv")
    dat_CH = np.genfromtxt(fn, delimiter=",")
    fn = os.path.join(THIS_DIR, "comparison_data/wr/Wunesch_Thesis/Fig6-16a_CH_tot.csv")
    dat_tot = np.genfromtxt(fn, delimiter=",")

    HH_sf = np.genfromtxt(
        os.path.join(THIS_DIR, "comparison_data/static_sf/Wuensch_Thesis_Fig6-16/Fig6-16b_HH.csv"), delimiter=","
    )
    CH_sf = np.genfromtxt(
        os.path.join(THIS_DIR, "comparison_data/static_sf/Wuensch_Thesis_Fig6-16/Fig6-16b_CH.csv"), delimiter=","
    )
    CC_sf = np.genfromtxt(
        os.path.join(THIS_DIR, "comparison_data/static_sf/Wuensch_Thesis_Fig6-16/Fig6-16b_CC.csv"), delimiter=","
    )

    fig, axes = plt.subplots(1, 3)
    ax = axes[0]
    ax.scatter(dat_CC[:, 0], dat_CC[:, 1], label="CC", c="blue", marker="*")
    ax.scatter(dat_HH[:, 0], dat_HH[:, 1], label="HH", c="black", marker="<")
    ax.scatter(dat_CH[:, 0], dat_CH[:, 1], label="CH", c="green", marker=">")
    ax.scatter(dat_tot[:, 0], dat_tot[:, 1], label="Tot", c="red", marker="x")
    ax.plot(k_xdave, WR, c="red", ls="-.", label="xDave")
    ax.set_xlim(0, 6)
    ax.set_xlabel(r"$k$ [$A^{-1}$]")
    ax.set_ylabel(r"$W_R$")
    ax.legend()

    ax = axes[1]
    ax.plot(k_xdave, Sab[0, 0], label=r"$S_{HH}$", c="navy", ls="-.")
    ax.plot(k_xdave, Sab[0, 1], label=r"$S_{CH}$", c="crimson", ls="-.")
    ax.plot(k_xdave, Sab[1, 1], label=r"$S_{CC}$", c="purple", ls="-.")
    ax.scatter(HH_sf[:, 0], HH_sf[:, 1], label=r"Wuensch $S_{HH}$", c="navy")
    ax.scatter(CH_sf[:, 0], CH_sf[:, 1], label=r"Wuensch $S_{CH}$", c="crimson")
    ax.scatter(CC_sf[:, 0], CC_sf[:, 1], label=r"Wuensch $S_{CC}$", c="purple")
    ax.set_xlim(0, 6)
    ax.set_xlabel(r"$k$ [$A^{-1}$]")
    ax.set_ylabel(r"$S_{ab}$")
    ax.legend()

    ax = axes[2]
    ax.plot(k_xdave, qs[0], label=r"$q_{H}$")
    ax.plot(k_xdave, qs[1], label=r"$q_{C}$")
    ax.set_xlim(0, 6)
    ax.set_xlabel(r"$k$ [$A^{-1}$]")
    ax.set_ylabel(r"$q_a$")
    ax.legend()

    plt.tight_layout()
    plt.show()


def test_Fig617a():

    fn = os.path.join(THIS_DIR, "comparison_data/wr/Wunesch_Thesis/Fig6-17a_C.csv")
    dat_C = np.genfromtxt(fn, delimiter=",")
    fn = os.path.join(THIS_DIR, "comparison_data/wr/Wunesch_Thesis/Fig6-17a_CH.csv")
    dat_CH = np.genfromtxt(fn, delimiter=",")
    fn = os.path.join(THIS_DIR, "comparison_data/wr/Wunesch_Thesis/Fig6-17a_CH2.csv")
    dat_CH2 = np.genfromtxt(fn, delimiter=",")

    models = ModelOptions(
        sf_model="HNC",
        ee_potential="COULOMB",
        ei_potential="COULOMB",
        ii_potential="DEBYE_HUCKEL",
        screening_model="FINITE_WAVELENGTH",
    )
    k = np.linspace(0.1, 10, 1000) / BOHR_RADIUS

    T = 8  # eV
    ni = 5.0e22  # per cc
    ni_SI = 5.0e22 * per_cm3_TO_per_m3
    amu_H = 1.0078 * amu_TO_kg
    amu_C = 12.011 * amu_TO_kg

    ZC = 2
    ZH = 1

    T_C = T
    rho_C = amu_C * ni_SI * kg_per_m3_TO_g_per_cm3

    kernel_C = xDave(
        mass_density=rho_C,
        electron_temperature=T_C,
        ion_temperature=T_C,
        elements=np.array(["C"]),
        partial_densities=np.array([1]),
        charge_states=np.array([ZC]),
        models=models,
    )

    wr_ocp = OCPRayleighWeight(overlord_state=kernel_C.overlord_state, state=kernel_C.states[0], verbose=True)
    k_xdave, Siik_C, WR_C, qs, fs = wr_ocp.get_rayleigh_weight(
        k=k,
        lfc=0.0,
        sf_model=models.sf_model,
        ii_potential=models.ii_potential,
        ee_potential=models.ee_potential,
        ei_potential=models.ei_potential,
        screening_model=models.screening_model,
        hnc_max_iterations=1000,
        hnc_delta=1.0e-10,
        hnc_mix_fraction=0.9,
        bridge_function=models.bridge_function,
        return_full=True,
    )

    T_CH = T
    rho_CH = (amu_C + amu_H) * ni_SI * kg_per_m3_TO_g_per_cm3

    kernel_CH = xDave(
        mass_density=rho_CH,
        electron_temperature=T_CH,
        ion_temperature=T_CH,
        elements=np.array(["H", "C"]),
        partial_densities=np.array([0.5, 0.5]),
        charge_states=np.array([ZH, ZC]),
        models=models,
    )

    wr_ocp = MCPRayleighWeight(overlord_state=kernel_CH.overlord_state, states=kernel_CH.states, verbose=True)
    k_xdave, Siik_CH, WR_CH, qs, fs = wr_ocp.get_rayleigh_weight(
        k=k,
        lfc=0.0,
        sf_model=models.sf_model,
        ii_potential=models.ii_potential,
        ee_potential=models.ee_potential,
        ei_potential=models.ei_potential,
        screening_model=models.screening_model,
        hnc_max_iterations=1000,
        hnc_delta=1.0e-10,
        hnc_mix_fraction=0.9,
        return_full=True,
    )

    T_CH2 = T
    rho_CH2 = (amu_C + 2 * amu_H) * ni_SI * kg_per_m3_TO_g_per_cm3

    kernel_CH2 = xDave(
        mass_density=rho_CH2,
        electron_temperature=T_CH2,
        ion_temperature=T_CH2,
        elements=np.array(["H", "H", "C"]),
        partial_densities=np.array([0.25, 0.25, 0.5]),
        charge_states=np.array([ZH, ZH, ZC]),
        models=models,
    )

    wr_ocp = MCPRayleighWeight(overlord_state=kernel_CH2.overlord_state, states=kernel_CH2.states, verbose=True)
    k_xdave, Siik_CH2, WR_CH2, qs, fs = wr_ocp.get_rayleigh_weight(
        k=k,
        lfc=0.0,
        sf_model=models.sf_model,
        ii_potential=models.ii_potential,
        ee_potential=models.ee_potential,
        ei_potential=models.ei_potential,
        screening_model=models.screening_model,
        hnc_max_iterations=1000,
        hnc_delta=1.0e-10,
        hnc_mix_fraction=0.9,
        return_full=True,
    )

    k_xdave *= BOHR_RADIUS * per_aB_TO_per_A

    plt.figure()
    plt.scatter(dat_C[:, 0], dat_C[:, 1], label="C", c="black", marker="*")
    plt.plot(k_xdave, WR_C, label="xDave: C", c="black", ls="-.")
    plt.scatter(dat_CH[:, 0], dat_CH[:, 1], label="CH", c="red", marker="<")
    plt.plot(k_xdave, WR_CH, label="xDave: CH", c="red", ls="-.")
    plt.scatter(dat_CH2[:, 0], dat_CH2[:, 1], label="CH2", c="blue", marker=">")
    plt.plot(k_xdave, WR_CH2, label="xDave: CH2", c="blue", ls="-.")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # test_Fig615a()
    test_Fig616a()
    # test_Fig617a()

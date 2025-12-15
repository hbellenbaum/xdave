from xdave.unit_conversions import (
    amu_TO_kg,
    eV_TO_K,
    K_TO_eV,
    per_cm3_TO_per_m3,
    kg_per_m3_TO_g_per_cm3,
    per_A_TO_per_aB,
    per_m3_TO_per_cm3,
)
from xdave.constants import BOLTZMANN_CONSTANT, VACUUM_PERMITTIVITY, BOHR_RADIUS, ELEMENTARY_CHARGE
from xdave.plasma_state import PlasmaState
from xdave.static_sf import OCPStaticStructureFactor, MCPStaticStructureFactor

from xdave.xdave import xDave
from xdave.models import ModelOptions

import numpy as np
import matplotlib.pyplot as plt
import os


THIS_DIR = os.path.dirname(__file__)


def test_ocp():
    r"""
    Comparison against K W\"unsch PhD Thesis (2011), Fig. 4.5
    """
    # plt.style.use("~/Desktop/resources/plotting/poster.mplstyle") # TG: Removed as I don't have this file :)

    # Case 1: Gamma_ii = 12.3, Ti = 4 eV
    T = 4 * eV_TO_K
    Zi = 2
    rho = 498.16  # kg/m^3

    sigma_c = 2.15 * BOHR_RADIUS
    state = PlasmaState(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_state=Zi,
        atomic_mass=2,
        atomic_number=2,
        binding_energies=None,
        ion_core_radius=sigma_c,
    )
    ni = rho / (2 * amu_TO_kg)
    Rii = (3 / (4 * np.pi * ni)) ** (1 / 3)
    Rii1 = Rii
    beta = 1 / (BOLTZMANN_CONSTANT * T)
    Gamma = Zi**2 * ELEMENTARY_CHARGE**2 * beta / (4 * np.pi * VACUUM_PERMITTIVITY * Rii)
    print(f"Gamma1 = {Gamma}")

    k = np.linspace(1.0e-1 / BOHR_RADIUS, 10 / BOHR_RADIUS, 200)
    kernel = OCPStaticStructureFactor(state=state, max_iterations=1000)
    Sii_HNC = kernel.get_ii_static_structure_factor(k=k, sf_model="HNC")
    Sii_xHNC = kernel.get_ii_static_structure_factor(k=k, sf_model="EXTENDED_HNC")
    Sii_MSA = kernel.get_ii_static_structure_factor(k=k, sf_model="MSA")

    fn = os.path.join(THIS_DIR, "comparison_data/static_sf/Wuensch_Thesis_Fig4-5/T_4eV_Gamma_12.3_HNC-OCP.csv")
    dat1 = np.genfromtxt(fn, delimiter=",")
    fn = os.path.join(THIS_DIR, "comparison_data/static_sf/Wuensch_Thesis_Fig4-5/T_4eV_Gamma_12.3_MSA-OCP.csv")
    dat11 = np.genfromtxt(fn, delimiter=",")

    T = 20 * eV_TO_K
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
        ion_core_radius=sigma_c,
    )
    ni = rho / (2 * amu_TO_kg)
    Rii = (3 / (4 * np.pi * ni)) ** (1 / 3)
    Rii2 = Rii
    beta = 1 / (BOLTZMANN_CONSTANT * T)
    Gamma2 = Zi**2 * ELEMENTARY_CHARGE**2 * beta / (4 * np.pi * VACUUM_PERMITTIVITY * Rii)
    print(f"Gamma2 = {Gamma2}")

    fn = os.path.join(THIS_DIR, "comparison_data/static_sf/Wuensch_Thesis_Fig4-5/T_20eV_Gamma_2.7_HNC-OCP.csv")
    dat2 = np.genfromtxt(fn, delimiter=",")
    fn = os.path.join(THIS_DIR, "comparison_data/static_sf/Wuensch_Thesis_Fig4-5/T_20eV_Gamma_2.7_MSA-OCP.csv")
    dat22 = np.genfromtxt(fn, delimiter=",")

    sigma_c = 1.5 * BOHR_RADIUS
    state2 = PlasmaState(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_state=Zi,
        atomic_mass=2,
        atomic_number=2,
        binding_energies=None,
        ion_core_radius=sigma_c,
    )

    kernel2 = OCPStaticStructureFactor(state=state2, max_iterations=5000, mix_fraction=0.9)
    Sii_HNC2 = kernel2.get_ii_static_structure_factor(k=k, sf_model="HNC")
    Sii_MSA2 = kernel2.get_ii_static_structure_factor(k=k, sf_model="MSA")

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(k * BOHR_RADIUS, Sii_MSA, label=f"MSA, Gamma={Gamma:.1f}", c="orange", ls="--")
    ax.plot(k * BOHR_RADIUS, Sii_HNC, label=f"HNC, Gamma={Gamma:.1f}", c="crimson", ls="-.")
    ax.plot(k * BOHR_RADIUS, Sii_xHNC, label=f"xHNC, Gamma={Gamma:.1f}", c="crimson", ls=":")
    ax.scatter(
        dat1[:, 0] * BOHR_RADIUS / Rii1, dat1[:, 1], label=f"HNC - Wuensch, Gamma={Gamma:.1f}", c="brown", marker="x"
    )
    ax.scatter(
        dat11[:, 0] * BOHR_RADIUS / Rii1,
        dat11[:, 1],
        label=f"MSA - Wuensch, Gamma={Gamma:.1f}",
        c="orange",
        marker="o",
    )
    ax.plot(k * BOHR_RADIUS, Sii_HNC2, label=f"HNC, Gamma={Gamma2:.1f}", c="navy", ls="-.")
    ax.plot(k * BOHR_RADIUS, Sii_MSA2, label=f"MSA, Gamma={Gamma2:.1f}", c="dodgerblue", ls="--")
    ax.scatter(
        dat2[:, 0] * BOHR_RADIUS / Rii2, dat2[:, 1], label=f"HNC - Wuensch, Gamma={Gamma2:.1f}", c="black", marker="x"
    )
    ax.scatter(
        dat22[:, 0] * BOHR_RADIUS / Rii2,
        dat22[:, 1],
        label=f"MSA - Wuensch, Gamma={Gamma2:.1f}",
        c="dodgerblue",
        marker="o",
    )
    ax.legend()
    ax.axhline(1.0, lw=1, ls=":", c="gray")
    ax.set_xlim(-0.1, 6.0)
    ax.set_xlabel(r"$k$ [$a_B^{-1}$]")
    ax.set_ylabel(r"$S_{ii}(k)$")
    plt.show()


def test_mcp():
    # Comparison against Wuensch Thesis Fig. 4.12 (a)

    T = 2.0e4  # K
    nH = nC = 2.5e23 * per_cm3_TO_per_m3  # /m^3
    ZH = 1
    ZC = 4
    xH = 0.5
    xC = 0.5

    mC = 12.0096 * amu_TO_kg
    mH = 1.00784 * amu_TO_kg

    rho = nH / xH * (xH * mH + xC * mC) * kg_per_m3_TO_g_per_cm3

    models = ModelOptions()
    elements = np.array(["H", "C"])
    charge_states = np.array([ZH, ZC])
    partial_densities = np.array([0.5, 0.5])
    code_kernel = xDave(
        electron_temperature=T * K_TO_eV,
        ion_temperature=T * K_TO_eV,
        mass_density=rho,
        elements=elements,
        partial_densities=partial_densities,
        charge_states=charge_states,
        models=models,
    )

    sf_kernel = MCPStaticStructureFactor(
        overlord_state=code_kernel.overlord_state,
        states=code_kernel.states,
        mix_fraction=0.99,
        max_iterations=10000,
        delta=1.0e-6,
    )

    Rab = code_kernel.overlord_state.mean_sphere_radius(code_kernel.overlord_state.ion_number_density)

    ks = np.linspace(0.01, 20, 1000) / BOHR_RADIUS
    k, r, gabr, hab, Sabs = sf_kernel.get_ab_static_structure_factor(
        k=ks, sf_model="HNC", pseudo_potential="YUKAWA", return_full=True
    )

    HH_dat = np.genfromtxt(
        os.path.join(THIS_DIR, "comparison_data/static_sf/Wuensch_Thesis_Fig4-12/Fig4-12a_HH.csv"), delimiter=","
    )
    CH_dat = np.genfromtxt(
        os.path.join(THIS_DIR, "comparison_data/static_sf/Wuensch_Thesis_Fig4-12/Fig4-12a_CH.csv"), delimiter=","
    )
    CC_dat = np.genfromtxt(
        os.path.join(THIS_DIR, "comparison_data/static_sf/Wuensch_Thesis_Fig4-12/Fig4-12a_CC.csv"), delimiter=","
    )

    HH_dat2 = np.genfromtxt(
        os.path.join(THIS_DIR, "comparison_data/static_sf/Wuensch_Thesis_Fig4-12/Fig4-12b_HH.csv"), delimiter=","
    )
    CH_dat2 = np.genfromtxt(
        os.path.join(THIS_DIR, "comparison_data/static_sf/Wuensch_Thesis_Fig4-12/Fig4-12b_CH.csv"), delimiter=","
    )
    CC_dat2 = np.genfromtxt(
        os.path.join(THIS_DIR, "comparison_data/static_sf/Wuensch_Thesis_Fig4-12/Fig4-12b_CC.csv"), delimiter=","
    )

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(r / Rab, gabr[0, 0, :], label="HH", c="black", ls="-.")
    axes[0].plot(r / Rab, gabr[0, 1, :], label="CH", c="blue", ls="-.")
    axes[0].plot(r / Rab, gabr[1, 1, :], label="CC", c="red", ls="-.")
    axes[0].scatter(HH_dat[:, 0], HH_dat[:, 1], label="Wuensch: HH", c="black", marker="<")
    axes[0].scatter(CH_dat[:, 0], CH_dat[:, 1], label="Wuensch: CH", c="blue", marker="o")
    axes[0].scatter(CC_dat[:, 0], CC_dat[:, 1], label="Wuensch: CC", c="red", marker="x")
    axes[0].legend()
    axes[0].set_xlim(-1, 12)
    axes[1].plot(k * Rab, Sabs[0, 0, :], label="HH", c="black", ls="-.")
    axes[1].plot(k * Rab, Sabs[0, 1, :], label="CH", c="blue", ls="-.")
    axes[1].plot(k * Rab, Sabs[1, 1, :], label="CC", c="red", ls="-.")
    axes[1].scatter(HH_dat2[:, 0], HH_dat2[:, 1], label="Wuensch: HH", c="black", marker="<")
    axes[1].scatter(CH_dat2[:, 0], CH_dat2[:, 1], label="Wuensch: CH", c="blue", marker="o")
    axes[1].scatter(CC_dat2[:, 0], CC_dat2[:, 1], label="Wuensch: CC", c="red", marker="x")
    axes[1].legend()
    axes[1].set_xlim(-1, 12)
    plt.show()


def test_wuensch_Fig616():
    # Comparison against Wuensch Thesis Fig. 6.16 (b)
    # Note that the difference at small k is likely due to the difference in inverse screening lengths used in the DH ii potential
    # THEMIS uses the Yukawa model with alpha = 1 / inverse_screening_length whereas I define alpha as 2/Rii
    # Yukawa ii potential, FWS screening, DFT form factor

    nH = 5e22 * per_cm3_TO_per_m3
    nC = nH
    ZC = 2
    ZH = 1
    T = 8  # eV
    mC = 12.011 * amu_TO_kg
    mH = 1.008 * amu_TO_kg
    rho = nC * mC + nH * mH
    rho *= kg_per_m3_TO_g_per_cm3
    models = ModelOptions()
    elements = np.array(["H", "C"])
    charge_states = np.array([ZH, ZC])
    partial_densities = np.array([0.5, 0.5])
    code_kernel = xDave(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        elements=elements,
        partial_densities=partial_densities,
        charge_states=charge_states,
        models=models,
    )

    state = code_kernel.overlord_state
    mix_fraction = 0.999

    ks = np.linspace(0.1, 10, 1000) / BOHR_RADIUS
    sf = MCPStaticStructureFactor(
        overlord_state=code_kernel.overlord_state,
        states=code_kernel.states,
        mix_fraction=mix_fraction,
        max_iterations=15000,
        delta=1.0e-12,
    )
    Sab = sf.get_ab_static_structure_factor(k=ks, sf_model="HNC", pseudo_potential="DEBYE_HUCKEL")

    HH_dat = np.genfromtxt(
        os.path.join(THIS_DIR, "comparison_data/static_sf/Wuensch_Thesis_Fig6-16/Fig6-16b_HH.csv"), delimiter=","
    )
    CH_dat = np.genfromtxt(
        os.path.join(THIS_DIR, "comparison_data/static_sf/Wuensch_Thesis_Fig6-16/Fig6-16b_CH.csv"), delimiter=","
    )
    CC_dat = np.genfromtxt(
        os.path.join(THIS_DIR, "comparison_data/static_sf/Wuensch_Thesis_Fig6-16/Fig6-16b_CC.csv"), delimiter=","
    )

    fig, ax = plt.subplots(1, 1)
    ax.scatter(HH_dat[:, 0] * per_A_TO_per_aB, HH_dat[:, 1], label="Wuensch: HH", c="black", marker="<")
    ax.plot(ks * BOHR_RADIUS, Sab[0, 0, :], label="HH", c="black", ls="-.")
    ax.scatter(CH_dat[:, 0] * per_A_TO_per_aB, CH_dat[:, 1], label="Wuensch: CH", c="blue", marker="o")
    ax.plot(ks * BOHR_RADIUS, Sab[0, 1, :], label="CH", c="blue", ls="-.")
    ax.scatter(CC_dat[:, 0] * per_A_TO_per_aB, CC_dat[:, 1], label="Wuensch: CC", c="red", marker="x")
    ax.plot(ks * BOHR_RADIUS, Sab[1, 1, :], label="CC", c="red", ls="-.")
    ax.legend()
    plt.show()


def test_wuensch_Fig617():
    # Comparison against Wuensch Thesis Fig. 6.17 (b)
    # Yukawa ii potential, FWS screening, DFT form factor
    # Tricky because of the input format I've chosen here,
    # getting to the same number densities while being forced to input a mass density
    # and not knowing the composition of CH and CH2 in Wuensch's work
    # Yukawa ii potential, FWS screening, DFT form factor

    ni = 5.0e22 * per_cm3_TO_per_m3
    ZC = 2
    ZH = 1
    T = 8  # eV
    nC = 5.0e22 * per_cm3_TO_per_m3
    nH = nC

    mC = 12.011 * amu_TO_kg
    mH = 1.008 * amu_TO_kg

    rho_C = nC * mC * kg_per_m3_TO_g_per_cm3
    rho_CH = 1.2
    rho_CH2 = 0.97 * 0.3

    # set up kernel for CH case
    models = ModelOptions()
    elements = np.array(["C"])
    charge_states = np.array([ZC])
    partial_densities_C = np.array([1.0])
    code_kernel_C = xDave(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho_C,
        elements=elements,
        partial_densities=partial_densities_C,
        charge_states=charge_states,
        models=models,
    )
    print(f"C: ni = {code_kernel_C.overlord_state.ion_number_density * per_m3_TO_per_cm3} 1/cc")

    # set up kernel for CH case
    models = ModelOptions()
    elements = np.array(["H", "C"])
    charge_states = np.array([ZH, ZC])
    partial_densities_CH = np.array([0.5, 0.5])
    code_kernel_CH = xDave(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho_CH,
        elements=elements,
        partial_densities=partial_densities_CH,
        charge_states=charge_states,
        models=models,
    )
    print(f"CH: ni = {code_kernel_CH.overlord_state.ion_number_density * per_m3_TO_per_cm3} 1/cc")
    print(f"CH: nH = {code_kernel_CH.states[0].ion_number_density * per_m3_TO_per_cm3} 1/cc")
    print(f"CH: nC = {code_kernel_CH.states[1].ion_number_density * per_m3_TO_per_cm3} 1/cc")

    # set up kernel for CH2 case
    models = ModelOptions()
    elements = np.array(["H", "H", "C"])
    charge_states = np.array([ZH, ZH, ZC])
    partial_densities_CH2 = np.array([0.25, 0.25, 0.5])
    code_kernel_CH2 = xDave(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho_CH2,
        elements=elements,
        partial_densities=partial_densities_CH2,
        charge_states=charge_states,
        models=models,
    )
    print(f"CH2: ni = {code_kernel_CH2.overlord_state.ion_number_density * per_m3_TO_per_cm3} 1/cc")
    print(f"CH2: nH1 = {code_kernel_CH2.states[0].ion_number_density * per_m3_TO_per_cm3} 1/cc")
    print(f"CH2: nH2 = {code_kernel_CH2.states[1].ion_number_density * per_m3_TO_per_cm3} 1/cc")
    print(f"CH2: nC = {code_kernel_CH2.states[2].ion_number_density * per_m3_TO_per_cm3} 1/cc")

    mix_fraction = 0.8

    ks = np.linspace(0.5, 10, 1000) / BOHR_RADIUS
    sf_C = OCPStaticStructureFactor(
        state=code_kernel_C.states[0],
        mix_fraction=mix_fraction,
        max_iterations=15000,
        delta=1.0e-12,
    )
    Sab_C = sf_C.get_ii_static_structure_factor(k=ks, sf_model="HNC", pseudo_potential="DEBYE_HUCKEL")

    mix_fraction = 0.99

    # ks = np.linspace(0.05, 10, 1000) / BOHR_RADIUS
    sf_CH = MCPStaticStructureFactor(
        overlord_state=code_kernel_CH.overlord_state,
        states=code_kernel_CH.states,
        mix_fraction=mix_fraction,
        max_iterations=15000,
        delta=1.0e-10,
    )
    Sab_CH = sf_CH.get_ab_static_structure_factor(k=ks, sf_model="HNC", pseudo_potential="DEBYE_HUCKEL")

    mix_fraction = 0.99

    # ks = np.linspace(0.05, 10, 1000) / BOHR_RADIUS
    sf_CH2 = MCPStaticStructureFactor(
        overlord_state=code_kernel_CH2.overlord_state,
        states=code_kernel_CH2.states,
        mix_fraction=mix_fraction,
        max_iterations=15000,
        delta=1.0e-10,
    )
    Sab_CH2 = sf_CH2.get_ab_static_structure_factor(k=ks, sf_model="HNC", pseudo_potential="DEBYE_HUCKEL")

    S_CH = np.zeros_like(ks)
    nspecies = 2
    for n1 in range(0, nspecies):
        for n2 in range(0, nspecies):
            S_CH += np.sqrt(partial_densities_CH[n1] * partial_densities_CH[n2]) * Sab_CH[n1, n2]

    S_CH2 = np.zeros_like(ks)
    nspecies = 3
    for n1 in range(0, nspecies):
        for n2 in range(0, nspecies):
            S_CH2 += np.sqrt(partial_densities_CH2[n1] * partial_densities_CH2[n2]) * Sab_CH2[n1, n2]

    fn = os.path.join(THIS_DIR, "comparison_data/static_sf/Wuensch_Thesis_Fig6-17/Fig6-17b_C.csv")
    dat_C = np.genfromtxt(fn, delimiter=",")
    fn = os.path.join(THIS_DIR, "comparison_data/static_sf/Wuensch_Thesis_Fig6-17/Fig6-17b_CH.csv")
    dat_CH = np.genfromtxt(fn, delimiter=",")
    fn = os.path.join(THIS_DIR, "comparison_data/static_sf/Wuensch_Thesis_Fig6-17/Fig6-17b_CH2.csv")
    dat_CH2 = np.genfromtxt(fn, delimiter=",")

    plt.figure()
    plt.scatter(dat_C[:, 0] * per_A_TO_per_aB, dat_C[:, 1], label="C", c="black", marker="*")
    plt.plot(ks * BOHR_RADIUS, Sab_C, c="black", label="C")
    plt.scatter(dat_CH[:, 0] * per_A_TO_per_aB, dat_CH[:, 1], label="CH", c="red", marker="<")
    plt.plot(ks * BOHR_RADIUS, S_CH, c="red", label="CH")
    plt.scatter(dat_CH2[:, 0] * per_A_TO_per_aB, dat_CH2[:, 1], label="CH2", c="blue", marker=">")
    plt.plot(ks * BOHR_RADIUS, S_CH2, c="blue", label="CH2")
    plt.legend()
    plt.tight_layout()
    plt.show()


def update_sf_file_ocp(fn, material, ks, S):
    arr = np.array([ks, S]).T
    file = fn + f"static_sf_{material}.txt"
    np.savetxt(file, arr, header="ks S")


def update_sf_file_mcp(fn, material, ks, S):

    arr = np.array([ks, S[0, 0], S[0, 1], S[1, 1]]).T
    file = fn + f"static_sf_{material}.txt"
    np.savetxt(file, arr, header="ks S11 S12 S22")


def test_version():
    ni = 5.0e22 * per_cm3_TO_per_m3
    ZC = 2
    ZH = 1
    T = 8  # eV
    nC = 5.0e22 * per_cm3_TO_per_m3
    nH = nC

    mC = 12.011 * amu_TO_kg
    mH = 1.008 * amu_TO_kg

    rho_C = nC * mC * kg_per_m3_TO_g_per_cm3
    rho_CH = 1.2

    fn = os.path.join(os.path.dirname(__file__), "xdave_results/static_sf/")
    if not os.path.exists(fn):
        os.mkdir(fn)

    # set up kernel for CH case
    models = ModelOptions()
    elements = np.array(["C"])
    charge_states = np.array([ZC])
    partial_densities_C = np.array([1.0])
    code_kernel_C = xDave(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho_C,
        elements=elements,
        partial_densities=partial_densities_C,
        charge_states=charge_states,
        models=models,
    )

    mix_fraction = 0.8
    ks = np.linspace(0.5, 10, 1000) / BOHR_RADIUS
    sf_C = OCPStaticStructureFactor(
        state=code_kernel_C.states[0], mix_fraction=mix_fraction, max_iterations=15000, delta=1.0e-12, verbose=False
    )
    Sab_C = sf_C.get_ii_static_structure_factor(k=ks, sf_model="HNC", pseudo_potential="DEBYE_HUCKEL")
    # update_sf_file_ocp(fn=fn, material="C", ks=ks, S=Sab_C)
    Sab_C_save = np.genfromtxt(fn + "static_sf_C.txt", delimiter=" ", skip_header=1)

    # multi-component
    # set up kernel for CH case
    models = ModelOptions()
    elements = np.array(["H", "C"])
    charge_states = np.array([ZH, ZC])
    partial_densities_CH = np.array([0.5, 0.5])
    code_kernel_CH = xDave(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho_CH,
        elements=elements,
        partial_densities=partial_densities_CH,
        charge_states=charge_states,
        models=models,
    )
    mix_fraction = 0.9
    sf_CH = MCPStaticStructureFactor(
        overlord_state=code_kernel_CH.overlord_state,
        states=code_kernel_CH.states,
        mix_fraction=mix_fraction,
        max_iterations=15000,
        delta=1.0e-10,
        verbose=False,
    )
    Sab_CH = sf_CH.get_ab_static_structure_factor(k=ks, sf_model="HNC", pseudo_potential="DEBYE_HUCKEL")
    # update_sf_file_mcp(fn=fn, material="CH", ks=ks, S=Sab_CH)
    Sab_CH_save = np.genfromtxt(fn + "static_sf_CH.txt", delimiter=" ", skip_header=1)

    rtol = 1.0e-2
    if not np.isclose(Sab_C, Sab_C_save[:, 1], rtol=rtol).all():
        print(f"OCP structure factor for Carbon failed.")
    if not np.isclose(Sab_CH[0, 0], Sab_CH_save[:, 1], rtol=rtol).all():
        print(f"OCP structure factor for CH failed for HH component.")
    if not np.isclose(Sab_CH[0, 1], Sab_CH_save[:, 2], rtol=rtol).all():
        print(f"OCP structure factor for CH failed for CH component.")
    if not np.isclose(Sab_CH[1, 1], Sab_CH_save[:, 3], rtol=rtol).all():
        print(f"OCP structure factor for CH failed for CC component.")


if __name__ == "__main__":
    # test_version()
    test_ocp()
    test_mcp()
    test_wuensch_Fig616()
    test_wuensch_Fig617()

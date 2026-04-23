from xdave.plasma_state import PlasmaState, get_fractions_from_Z_partial
from xdave.boundfree_dsf import BoundFreeDSF
from xdave.constants import BOHR_RADIUS
from xdave.unit_conversions import (
    eV_TO_K,
    g_per_cm3_TO_kg_per_m3,
    eV_TO_J,
    J_TO_eV,
    per_cm3_TO_per_m3,
)
from xdave.utils import calculate_q, load_mcss_result, calculate_angle
from xdave import xDave, ModelOptions

import numpy as np
import matplotlib.pyplot as plt
import os


def test_bf_mcss():

    Te = 10 * eV_TO_K
    rho = 0.1 * g_per_cm3_TO_kg_per_m3
    charge_state = 0.0
    atomic_number = 1
    atomic_mass = 1.0
    beam_energy = 8.0e3

    angles = np.array([13, 30, 45, 60, 80, 100, 120, 140, 160])
    ks = calculate_q(angle=angles, energy=beam_energy) / BOHR_RADIUS
    omega_array = np.linspace(-40, 300, 500) * eV_TO_J
    EB = (
        np.array(
            [
                -13.6,
            ]
        )
        * eV_TO_J
    )
    state = PlasmaState(
        electron_temperature=Te,
        ion_temperature=Te,
        mass_density=rho,
        charge_state=charge_state,
        atomic_mass=atomic_mass,
        atomic_number=atomic_number,
        binding_energies=EB,
    )

    colors = ["red", "green", "blue", "orange", "gray", "black", "yellow", "magenta", "purple"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for i in range(len(ks)):
        k = ks[i]
        k_bohr = k * BOHR_RADIUS
        angle = np.round(calculate_angle(q=k * BOHR_RADIUS, energy=beam_energy))  # angles[i]
        print(f"Running for k={k * BOHR_RADIUS} 1/aB and angle={angle}")
        c = colors[i]
        kernel = BoundFreeDSF(state=state)
        dsf = kernel.get_dsf(ZA=1, Zb=1, Eb=EB, w=omega_array, k=k, model="SCHUMACHER")
        dsf_hr = kernel.get_dsf(ZA=1, Zb=1, Eb=EB, w=omega_array, k=k, model="HR_CORRECTION")
        dsf_tr = kernel.get_dsf(ZA=1, Zb=1, Eb=EB, w=omega_array, k=k, model="TRUNCATED_IA")

        En, wff, wbf, ff, bf, el = load_mcss_result(
            filename=os.path.dirname(__file__)
            + f"/comparison_data/bf_dsf/mcss_test/mcss_bf_test_angle={angle:0.0f}.csv"
        )
        ax.plot(
            omega_array * J_TO_eV,
            np.array(dsf) / J_TO_eV,
            label=f"IA: k={k_bohr:.2f}",
            c=c,
            ls="solid",
            marker=".",
            markevery=10,
        )
        ax.plot(
            omega_array * J_TO_eV,
            np.array(dsf_hr) / J_TO_eV,
            label=f"HR: k={k_bohr:.2f}",
            c=c,
            ls="dotted",
            marker="<",
            markevery=14,
        )
        ax.plot(
            omega_array * J_TO_eV,
            np.array(dsf_tr) / J_TO_eV,
            label=f"IA tr: k={k_bohr:.2f}",
            c=c,
            ls="-.",
            marker="*",
            markevery=12,
        )
        ax.plot(En, wbf, label=f"MCSS k={k_bohr:.2f}", c=c, ls="dashed")
    for eb in EB:
        ax.axvline(np.abs(eb) * J_TO_eV, c="gray", ls="dotted")
    ax.legend(ncol=3)
    ax.set_xlim(-40, 300)
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.set_ylabel(r"$S_{ee}^{bf}(k,\omega)$ [1/eV]")
    plt.show()

    assert np.isclose(
        dsf / J_TO_eV, np.interp(x=omega_array * J_TO_eV, xp=En, fp=wbf), rtol=1.0e-6
    ).all(), f"IA test has failed."


def test_be_bf():
    # Comparison to Fig. 2 Mattern and Seidel, Phys. Plasmas 20 (2013)
    # Be at q~10.2 1/A, phi = 171, Eb = 9890 eV
    data_dir = os.path.dirname(__file__)

    # values from Fortman et al., PRL 108, 175006 (2012)
    Te = 13 * eV_TO_K
    Zf = 2.0
    ZA = 4
    Zb = ZA - Zf
    ne = 1.8e24 * per_cm3_TO_per_m3
    ni = ne / Zf

    rho = 10 * g_per_cm3_TO_kg_per_m3
    charge_state = 0.0
    atomic_number = ZA
    atomic_mass = 9.012182  # amu
    beam_energy = 9890  # eV
    scattering_angle = 171  # degree
    q_approx = 10.2  # 1/Angstrom
    A_TO_m = 1.0e-10
    k = q_approx / A_TO_m

    # angles = np.array([13, 30, 45, 60, 80, 100, 120, 140, 160])
    omega_array = np.linspace(-500, 900, 1000) * eV_TO_J
    binding_energies = np.array([-111.5, -111.5, -111.5]) * eV_TO_J
    state = PlasmaState(
        electron_temperature=Te,
        ion_temperature=Te,
        mass_density=rho,
        electron_number_density=ne,
        ion_number_density=ni,
        charge_state=Zf,
        atomic_mass=atomic_mass,
        atomic_number=atomic_number,
        binding_energies=binding_energies,
    )

    bf_dsf = BoundFreeDSF(state=state)
    xdave_bf_IA = bf_dsf.get_dsf(ZA=ZA, Zb=Zb, k=k, w=omega_array, Eb=binding_energies, model="SCHUMACHER")
    xdave_bf_HR = bf_dsf.get_dsf(ZA=ZA, Zb=Zb, k=k, w=omega_array, Eb=binding_energies, model="HR_CORRECTION")

    exp_data = np.genfromtxt(data_dir + f"/comparison_data/bf_dsf/mattern2013/Be_Fig1.csv", delimiter=",")
    hm_data = np.genfromtxt(data_dir + f"/comparison_data/bf_dsf/mattern2013/Be_HM.csv", delimiter=",")
    ia_data = np.genfromtxt(data_dir + f"/comparison_data/bf_dsf/mattern2013/Be_IA.csv", delimiter=",")
    pwffa = np.genfromtxt(data_dir + f"/comparison_data/bf_dsf/mattern2013/Be_PWFFA.csv", delimiter=",")
    rsgf = np.genfromtxt(data_dir + f"/comparison_data/bf_dsf/mattern2013/Be_RSGF.csv", delimiter=",")
    rsgf_core = np.genfromtxt(data_dir + f"/comparison_data/bf_dsf/mattern2013/RSGF_Be_3.6_core.csv", delimiter=",")

    plt.figure()
    plt.scatter(exp_data[:, 0], exp_data[:, 1], label="Exp")
    plt.plot(hm_data[:, 0], hm_data[:, 1], label="HM")
    plt.plot(ia_data[:, 0], ia_data[:, 1], label="IA")
    plt.plot(pwffa[:, 0], pwffa[:, 1], label="PWFFA")
    plt.plot(rsgf[:, 0], rsgf[:, 1], label="RSGF")
    plt.plot(rsgf_core[:, 0], rsgf_core[:, 1], label="RSGF Core")
    plt.plot(omega_array * J_TO_eV, xdave_bf_IA * 1.0e2 / J_TO_eV, label="xdave: IA", ls="--", c="black")
    plt.plot(omega_array * J_TO_eV, xdave_bf_HR * 1.0e2 / J_TO_eV, label="xdave: HR", ls="--", c="green")
    plt.legend()
    plt.show()


def test_multispecies_bf():
    T = 100  # eV
    rho = 2 * 1.845  # two times solid density [g/cc]
    Z_C = 4.5

    xH = 0.5
    ZH = 1.0

    Zmin, Zmax, xmin, xmax = get_fractions_from_Z_partial(Z=Z_C, x0=xH)

    models = ModelOptions(
        ei_potential="YUKAWA",
        ii_potential="YUKAWA",
        ee_potential="COULOMB",
        polarisation_model="NUMERICAL",
        sf_model="HNC",
        lfc_model="DORNHEIM_ESA",
        ipd_model="NONE",
        bf_model="SCHUMACHER",
        screening_model="FINITE_WAVELENGTH",
    )

    kernel = xDave(
        mass_density=rho,
        electron_temperature=T,
        ion_temperature=T,
        elements=np.array(["H", "C", "C"]),
        charge_states=np.array([ZH, Zmin, Zmax]),
        partial_densities=np.array([xH, xmin, xmax]),
        models=models,
        enforce_fsum=False,
        user_defined_inputs=None,
        verbose=True,
        hnc_max_iterations=10000,
        hnc_mix_fraction=0.99,
        hnc_delta=1.0e-7,
    )

    w = np.linspace(-1000, 1500, 10000)
    bf_tot, ff_tot, dsf, rayleigh_weight, ff_i, bf_i = kernel.run(w=w, angle=130, beam_energy=9.0e3, mode="DYNAMIC")

    plt.figure()
    # plt.plot(w, bf_i[0], label="H")
    plt.plot(w, bf_i[1], label="C4", c="crimson", ls="-.")
    plt.plot(w, bf_i[2], label="C5", c="navy", ls="-.")
    plt.plot(w, bf_tot, label="Tot", c="green", ls="-.")
    plt.xlabel(r"$\omega$ [eV]")
    plt.ylabel(r"$S^{bf} (k,\omega)$ [1/eV]")

    models = ModelOptions(
        ei_potential="YUKAWA",
        ii_potential="YUKAWA",
        ee_potential="COULOMB",
        polarisation_model="NUMERICAL",
        sf_model="HNC",
        lfc_model="DORNHEIM_ESA",
        ipd_model="CROWLEY",
        bf_model="SCHUMACHER",
        screening_model="FINITE_WAVELENGTH",
    )
    kernel.models = models
    bf_tot, ff_tot, dsf, rayleigh_weight, ff_i, bf_i = kernel.run(w=w, angle=130, beam_energy=9.0e3, mode="DYNAMIC")
    # plt.show()
    # plt.plot(w, bf_i[1], c="crimson", ls=":")
    # plt.plot(w, bf_i[2], c="navy", ls=":")
    # plt.plot(w, bf_tot, c="green", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bf_example.pdf", dpi=200)


if __name__ == "__main__":

    # plt.style.use("~/my_style.mplstyle")
    # test_multispecies_bf()
    test_bf_mcss()
    # test_be_bf()

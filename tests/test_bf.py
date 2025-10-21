import sys

sys.path.insert(1, "/home/bellen85/code/dev/xdave/xdave")

from plasma_state import PlasmaState
from models import ModelOptions
from boundfree_dsf import BoundFreeDSF
from constants import BOHR_RADIUS
from unit_conversions import eV_TO_K, g_per_cm3_TO_kg_per_m3, eV_TO_J, J_TO_eV, per_cm3_TO_per_m3
from utils import calculate_q, load_mcss_result, calculate_angle


import numpy as np


def test_bf():
    import matplotlib.pyplot as plt

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
                -13.7,  # -13.6,  # -13.6,
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
    models = ModelOptions(bf_model="SCHUMACHER")

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
            filename=f"./mcss_tests/mcss_outputs_model=IA/mcss_bf_test_angle={angle:0.0f}.csv"
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
    ax.legend()
    ax.set_xlim(-40, 300)
    plt.show()


def test_be_bf():
    # Comparison to Fig. 2 Mattern and Seidel, Phys. Plasmas 20 (2013)
    # Be at q~10.2 1/A, phi = 171, Eb = 9890 eV
    import matplotlib.pyplot as plt

    # values from Fortman et al., PRL 108, 175006 (2012)
    Te = 13 * eV_TO_K
    ne = 1.8e24 * per_cm3_TO_per_m3
    Zf = 2.0

    rho = 0.1 * g_per_cm3_TO_kg_per_m3
    charge_state = 0.0
    atomic_number = 1
    atomic_mass = 1.0
    beam_energy = 9890  # eV
    scattering_angle = 171  # degree
    q_approx = 10.2  # 1/Angstrom

    angles = np.array([13, 30, 45, 60, 80, 100, 120, 140, 160])
    omega_array = np.linspace(-40, 300, 500) * eV_TO_J
    EB = (
        np.array(
            [
                -111.5,
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
    )
    models = ModelOptions(bf_model="SCHUMACHER")

    exp_data = np.genfromtxt(f"validation/bf_dsf/mattern2013/Be_Fig1.csv", delimiter=",")
    hm_data = np.genfromtxt(f"validation/bf_dsf/mattern2013/Be_HM.csv", delimiter=",")
    ia_data = np.genfromtxt(f"validation/bf_dsf/mattern2013/Be_IA.csv", delimiter=",")
    pwffa = np.genfromtxt(f"validation/bf_dsf/mattern2013/Be_PWFFA.csv", delimiter=",")
    rsgf = np.genfromtxt(f"validation/bf_dsf/mattern2013/Be_RSGF.csv", delimiter=",")
    rsgf_core = np.genfromtxt(f"validation/bf_dsf/mattern2013/RSGF_Be_3.6_core.csv", delimiter=",")

    # plt.figure()
    # plt.scatter(exp_data[:, 0], exp_data[:, 1], label="Exp", c="k")
    # plt.plot(hm_data[:, 0], hm_data[:, 1], label="HM", c="magenta")
    # plt.plot(ia_data[:, 0], ia_data[:, 1], label="IA", c="navy")
    # plt.plot(pwffa[:, 0], pwffa[:, 1], label="PWFFA", c="crimson")
    # plt.plot(rsgf[:, 0], rsgf[:, 1], label="RSGF", c="orange", marker=".")
    # plt.plot(rsgf_core[:, 0], rsgf_core[:, 1], label="RSGF - core", c="darkgreen", marker=".")
    # plt.legend()
    # plt.ylim(-0.05, 0.35)
    # plt.show()


if __name__ == "__main__":
    test_bf()

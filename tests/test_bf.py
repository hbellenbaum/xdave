from xdave.plasma_state import PlasmaState
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
    ax.legend()
    ax.set_xlim(-40, 300)
    plt.show()

    if not np.isclose(dsf / J_TO_eV, np.interp(x=omega_array * J_TO_eV, xp=En, fp=wbf), rtol=1.0e-6).all():
        print(f"IA test has failed.")


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


def update_bf_results(model, w, k_bohr, dsf, fn):
    file = fn + f"_k={k_bohr:.1f}_model={model}.csv"
    np.savetxt(file, np.array([w, dsf]).T)
    print(f"Updating BF test for model = {model}: \nfile = {file}")


def test_version():
    Te = 110 * eV_TO_K
    rho = 10.0 * g_per_cm3_TO_kg_per_m3
    atomic_number = 4
    atomic_mass = 1.0
    beam_energy = 9.0e3
    charge_state = 3

    ZA = atomic_number
    Zb = atomic_number - charge_state

    angles = np.array([13, 30, 45, 60, 80, 100, 120, 140, 160])
    ks = calculate_q(angle=angles, energy=beam_energy) / BOHR_RADIUS
    omega_array = np.linspace(-450, 800, 1000) * eV_TO_J
    binding_energies = np.array([-111.5, -111.5, -111.5]) * eV_TO_J
    state = PlasmaState(
        electron_temperature=Te,
        ion_temperature=Te,
        mass_density=rho,
        charge_state=charge_state,
        atomic_mass=atomic_mass,
        atomic_number=atomic_number,
        binding_energies=binding_energies,
    )

    colors = ["red", "green", "blue", "orange", "gray", "black", "yellow", "magenta", "purple"]

    output_dir = os.path.join(os.path.dirname(__file__), "xdave_results/bf/")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    fn = output_dir + f"dsf_check_be_T={Te/eV_TO_K:.0f}_rho={rho/g_per_cm3_TO_kg_per_m3:.0f}_Z={charge_state}"

    test_IA = np.full_like(ks, True)
    test_HR = np.full_like(ks, True)
    test_trIA = np.full_like(ks, True)

    for i in range(len(ks)):
        k = ks[i]
        k_bohr = k * BOHR_RADIUS
        angle = np.round(calculate_angle(q=k * BOHR_RADIUS, energy=beam_energy))  # angles[i]
        print(f"Running for k={k * BOHR_RADIUS} 1/aB and angle={angle}")

        kernel = BoundFreeDSF(state=state)
        dsf = kernel.get_dsf(ZA=ZA, Zb=Zb, Eb=binding_energies, w=omega_array, k=k, model="SCHUMACHER")
        # update_bf_results(model="IA", w=omega_array, k_bohr=k_bohr, dsf=dsf, fn=fn)
        dsf_ia_save = np.genfromtxt(fn + f"_k={k_bohr:.1f}_model=IA.csv", delimiter=" ")
        dsf_hr = kernel.get_dsf(ZA=ZA, Zb=Zb, Eb=binding_energies, w=omega_array, k=k, model="HR_CORRECTION")
        # update_bf_results(model="HR", w=omega_array, k_bohr=k_bohr, dsf=dsf, fn=fn)
        dsf_hr_save = np.genfromtxt(fn + f"_k={k_bohr:.1f}_model=HR.csv", delimiter=" ")
        dsf_tr = kernel.get_dsf(ZA=ZA, Zb=Zb, Eb=binding_energies, w=omega_array, k=k, model="TRUNCATED_IA")
        # update_bf_results(model="trIA", w=omega_array, k_bohr=k_bohr, dsf=dsf, fn=fn)
        dsf_tr_save = np.genfromtxt(fn + f"_k={k_bohr:.1f}_model=trIA.csv", delimiter=" ")

        if not np.isclose(dsf, dsf_ia_save[:, 1], rtol=1.0e-2).all(axis=-1):
            print(f"Impulse approximation BF model has failed the test at k={k_bohr:.1f} 1/aB.")
            test_IA[i] = False
        if not np.isclose(dsf_hr, dsf_hr_save[:, 1], rtol=1.0e-2).all(axis=-1):
            print(f"Holm-Ribberfors correction to the IA BF model has failed the test at k={k_bohr:.1f} 1/aB.")
            test_HR[i] = False
        if np.isclose(dsf_tr, dsf_tr_save[:, 1], rtol=1.0e-2).all(axis=-1):
            print(f"truncated IA BF model has failed the test at k={k_bohr:.1f} 1/aB.")
            test_trIA[i] = False


if __name__ == "__main__":
    test_bf_mcss()
    # test_be_bf()
    test_version()

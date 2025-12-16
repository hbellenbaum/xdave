from xdave.constants import BOHR_RADIUS
from xdave.unit_conversions import g_per_cm3_TO_kg_per_m3, eV_TO_K, per_m3_TO_per_cm3
from xdave.plasma_state import PlasmaState, get_rho_T_from_rs_theta
from xdave.utils import load_mcss_result_ar

from xdave.lfc import LFC

from xdave import xDave
from xdave.models import ModelOptions

import numpy as np
import matplotlib.pyplot as plt

import os


def test():

    ks = np.linspace(0.01, 10, 500) / BOHR_RADIUS
    rs = 2
    theta = 1

    rho, T = get_rho_T_from_rs_theta(rs=rs, theta=theta)

    elements = np.array(["C", "C"])
    partial_densities = np.array([0, 1])
    charge_states = np.array([2, 2])
    models = ModelOptions()

    xrts_code = xDave(
        mass_density=rho,
        electron_temperature=T,
        ion_temperature=T,
        elements=elements,
        partial_densities=partial_densities,
        charge_states=charge_states,
        models=models,
    )
    state = xrts_code.overlord_state

    lfc_interp = np.zeros_like(ks)
    lfc_ui = np.zeros_like(ks)
    lfc_gv = np.zeros_like(ks)
    lfc_dornheim = np.zeros_like(ks)
    lfc_farid = np.zeros_like(ks)

    kernel = LFC(state=state)
    lfc_interp = kernel.calculate_lfc(k=ks, w=0.0, model="PADE_INTERP")
    lfc_ui = kernel.calculate_lfc(k=ks, w=0.0, model="UI")
    lfc_gv = kernel.calculate_lfc(k=ks, w=0.0, model="GV")
    lfc_dornheim = kernel.calculate_lfc(k=ks, w=0.0, model="DORNHEIM_ESA")
    lfc_farid = kernel.calculate_lfc(k=ks, w=0.0, model="FARID")

    kF = state.fermi_wave_number(state.free_electron_number_density)
    plt.figure()
    plt.plot(ks / kF, lfc_interp, label="Interp")
    plt.plot(ks / kF, lfc_ui, label="UI")
    plt.plot(ks / kF, lfc_gv, label="GV")
    plt.plot(ks / kF, lfc_farid, label="Farid")
    plt.plot(ks / kF, lfc_dornheim, label="ESA")
    plt.xlabel(r"$k/k_F$")
    plt.ylabel(r"$G_{ee}(k)$")
    plt.legend()
    plt.show()


def test_ui_gv_mcss():
    rs = 1.86
    T1 = 20  # eV
    T2 = 4  # eV

    Z = 1

    rho, _ = get_rho_T_from_rs_theta(rs=rs, theta=1)
    state1 = PlasmaState(
        electron_temperature=T1 * eV_TO_K,
        mass_density=rho * g_per_cm3_TO_kg_per_m3,
        ion_temperature=T1 * eV_TO_K,
        charge_state=Z,
        atomic_mass=6,
        atomic_number=6,
        binding_energies=None,
    )

    elements = np.array(["C", "C"])
    partial_densities = np.array([0, 1])
    charge_states = np.array([2, 2])
    models = ModelOptions()

    xrts_code = xDave(
        mass_density=rho,
        electron_temperature=T1,
        ion_temperature=T1,
        elements=elements,
        partial_densities=partial_densities,
        charge_states=charge_states,
        models=models,
    )

    data_dir = os.path.dirname(__file__) + "/comparison_data/lfc/mcss_tests"
    assert os.path.exists(data_dir), f"Data folder cannot be found. Check your paths."
    fn_UI = os.path.join(data_dir, f"mcss_ar_run_c_T=20.00_rho=0.42_Z=2_model=STATIC_UTSUMI_ICHIMARU.csv")
    fn_GV = os.path.join(data_dir, f"mcss_ar_run_c_T=20.00_rho=0.42_Z=2_model=STATIC_GELDART_VOSKO.csv")
    fn_interp = os.path.join(data_dir, f"mcss_ar_run_c_T=20.00_rho=0.42_Z=2_model=STATIC_INTERP.csv")
    k_UI, WR, f1, f2, q1, q2, S11, S12, S22, lfc_UI = load_mcss_result_ar(fn_UI, use_lfc_model=True)
    k_GV, WR, f1, f2, q1, q2, S11, S12, S22, lfc_GV = load_mcss_result_ar(fn_GV, use_lfc_model=True)
    k_interp, WR, f1, f2, q1, q2, S11, S12, S22, lfc_interp = load_mcss_result_ar(fn_interp, use_lfc_model=True)

    THIS_DIR = os.path.dirname(__file__)
    datT1 = np.genfromtxt(
        os.path.join(THIS_DIR, f"comparison_data/lfc/Gregori_2007_Fig1a_rs_1.86_T_20eV.csv"), delimiter=","
    )

    kF1 = 1 / rs * (3 / 4 * np.pi) ** 3
    ks1 = np.linspace(0.01, 20, 500) / BOHR_RADIUS
    lfcs1_interp = np.zeros_like(ks1)
    lfcs1_gv = np.zeros_like(ks1)
    lfcs1_ui = np.zeros_like(ks1)

    kernel1 = LFC(state=xrts_code.overlord_state)

    for i in range(0, len(ks1)):
        lfcs1_interp[i] = kernel1.calculate_lfc(k=ks1[i], w=0, model="PADE_INTERP")
        lfcs1_gv[i] = kernel1.calculate_lfc(k=ks1[i], w=0, model="GV")
        lfcs1_ui[i] = kernel1.calculate_lfc(k=ks1[i], w=0, model="UI")

    plt.figure()
    plt.plot(ks1 * BOHR_RADIUS, lfcs1_interp, label="Interp", c="navy", ls="-.")
    plt.plot(ks1 * BOHR_RADIUS, lfcs1_gv, label="GV", c="crimson", ls="-.")
    plt.plot(ks1 * BOHR_RADIUS, lfcs1_ui, label="UI", c="purple", ls="-.")
    plt.plot(k_UI, lfc_UI, label="MCSS: UI", c="purple", ls="solid")
    plt.plot(k_GV, lfc_GV, label="MCSS: GV", c="crimson", ls="solid")
    plt.plot(k_interp, lfc_interp, label="MCSS: Interp", c="navy", ls="solid")
    plt.legend()
    plt.show()


def test_gregori_2007():

    ne = 2.5e23  # cm^{-3}
    rs = 1.86
    T1 = 20  # eV
    T2 = 4  # eV

    rho, _ = get_rho_T_from_rs_theta(rs=rs, theta=1)

    elements = np.array(["H"])
    partial_densities = np.array([1])
    charge_states = np.array([1])
    models = ModelOptions()

    xrts_code = xDave(
        mass_density=rho,
        electron_temperature=T1,
        ion_temperature=T1,
        elements=elements,
        partial_densities=partial_densities,
        charge_states=charge_states,
        models=models,
    )
    state1 = xrts_code.overlord_state

    print(f"ne = {state1.free_electron_number_density * per_m3_TO_per_cm3} 1/cc")
    print(rf"$\theta$ = {state1.theta}")

    xrts_code = xDave(
        mass_density=rho,
        electron_temperature=T2,
        ion_temperature=T2,
        elements=elements,
        partial_densities=partial_densities,
        charge_states=charge_states,
        models=models,
    )
    state2 = xrts_code.overlord_state

    print(f"ne = {state2.free_electron_number_density * per_m3_TO_per_cm3} 1/cc")
    print(f"$\\theta$ = {state2.theta}")

    kernel1 = LFC(state=state1)
    kernel2 = LFC(state=state2)

    ks1 = np.linspace(0.01, 20, 500) / BOHR_RADIUS
    ks2 = np.linspace(0.01, 20, 500) / BOHR_RADIUS
    lfcs1 = np.zeros_like(ks1)
    lfcs1_gv = np.zeros_like(ks1)
    lfcs1_ui = np.zeros_like(ks2)

    lfcs2 = np.zeros_like(ks2)

    for i in range(0, len(ks1)):
        lfcs1[i] = kernel1.calculate_lfc(k=ks1[i], w=0, model="PADE_INTERP")
        lfcs1_gv[i] = kernel1.calculate_lfc(k=ks1[i], w=0, model="GV")
        lfcs1_ui[i] = kernel1.calculate_lfc(k=ks1[i], w=0, model="UI")
    for i in range(0, len(ks2)):
        lfcs2[i] = kernel2.calculate_lfc(k=ks2[i], w=0, model="PADE_INTERP")

    THIS_DIR = os.path.dirname(__file__)
    datT1 = np.genfromtxt(
        os.path.join(THIS_DIR, f"comparison_data/lfc/Gregori_2007_Fig1a_rs_1.86_T_20eV.csv"), delimiter=","
    )
    datT2 = np.genfromtxt(
        os.path.join(THIS_DIR, f"comparison_data/lfc/Gregori_2007_Fig1a_rs_1.86_T_4eV.csv"), delimiter=","
    )

    plt.figure()
    plt.plot(ks1 * BOHR_RADIUS, lfcs1, label=f"T=20", ls="-.", c="navy")
    plt.plot(
        ks1 * BOHR_RADIUS,
        lfcs1_gv,
        label=f"GV: T=20",
        ls="-.",
        c="dodgerblue",
        marker="x",
    )
    plt.plot(
        ks1 * BOHR_RADIUS,
        lfcs1_ui,
        label=f"UI: T=20",
        ls="-.",
        c="magenta",
        marker="<",
    )
    kF1 = 1 / rs * (3 / 4 * np.pi) ** 3
    # Note to self: I don't think the data in Gianluca's paper Fig. 1 (a) is actually plotted against k/kF
    # I think it's actually k in a_B^{-1}
    plt.plot(datT1[:, 0], datT1[:, 1], label=f"Gregori et al., T=20", ls="solid", c="navy")
    plt.legend()
    plt.show()


def test_fortmann_2010():

    ne = 2.5e23  # cm^{-3}
    rs = 2
    T = 10  #

    rho, _ = get_rho_T_from_rs_theta(rs=rs, theta=1)

    state = PlasmaState(
        electron_temperature=T * eV_TO_K,
        mass_density=rho * g_per_cm3_TO_kg_per_m3,
        ion_temperature=T * eV_TO_K,
        charge_state=1.0,
        atomic_mass=1,
        atomic_number=1,
        binding_energies=None,
    )

    kernel = LFC(state=state)

    ks = np.linspace(0.01, 4, 100) * state.fermi_wave_number(state.free_electron_number_density)
    lfcs_iu = np.zeros_like(ks)
    lfcs_FARID = np.zeros_like(ks)

    for i in range(0, len(ks)):
        lfcs_iu[i] = kernel.calculate_lfc(k=ks[i], w=0, model="UI")
        lfcs_FARID[i] = kernel.calculate_lfc(k=ks[i], w=0, model="FARID")

    THIS_DIR = os.path.dirname(__file__)
    dat_iu = np.genfromtxt(
        os.path.join(THIS_DIR, f"comparison_data/lfc/Fortmann_2010_Fig2_utsumi_ichimaru.csv"), delimiter=","
    )
    dat_farid = np.genfromtxt(
        os.path.join(THIS_DIR, f"comparison_data/lfc/Fortmann_2010_Fig2_farid.csv"), delimiter=","
    )

    kF = state.fermi_wave_number(state.free_electron_number_density)

    plt.figure()
    plt.plot(ks / kF, lfcs_iu, label=f"UI", ls="-.", c="navy")
    plt.scatter(dat_iu[:, 0], dat_iu[:, 1], label=f"Fortmann et al., UI", c="navy")
    plt.plot(
        ks / state.fermi_wave_number(state.free_electron_number_density),
        lfcs_FARID,
        label=f"FARID",
        ls="-.",
        c="crimson",
    )
    plt.scatter(dat_farid[:, 0], dat_farid[:, 1], label=f"Fortmann et al., Farid", c="crimson")
    plt.legend()
    plt.show()

    if not np.isclose(lfcs_iu, np.interp(x=ks / kF, xp=dat_iu[:, 0], fp=dat_iu[:, 1]), atol=1.0e-1).all():
        print(f"UI LFC test has failed.")
    if not np.isclose(lfcs_FARID, np.interp(x=ks / kF, xp=dat_farid[:, 0], fp=dat_farid[:, 1]), atol=1.0e-1).all():
        print(f"Farid LFC test has failed.")


def test_farid():

    rss = np.array([1, 2, 5, 10, 15])

    colors = ["navy", "crimson", "magenta", "dodgerblue", "limegreen"]

    plt.figure()

    for rs, c in zip(rss, colors):
        rho, T = get_rho_T_from_rs_theta(rs=rs, theta=1)
        state = PlasmaState(
            electron_temperature=T * eV_TO_K,
            mass_density=rho * g_per_cm3_TO_kg_per_m3,
            ion_temperature=T * eV_TO_K,
            charge_state=1.0,
            atomic_mass=1,
            atomic_number=1,
            binding_energies=None,
        )
        kF = state.fermi_wave_number(state.free_electron_number_density)
        ks = np.linspace(0.01, 5, 1000) / BOHR_RADIUS
        lfcs = np.zeros_like(ks)
        lfcs_iu = np.zeros_like(ks)

        kernel = LFC(state=state)

        for i in range(0, len(ks)):
            lfcs[i] = kernel.calculate_lfc(k=ks[i], w=0, model="FARID")
            lfcs_iu[i] = kernel.calculate_lfc(k=ks[i], w=0, model="UI")

        kF1 = 1 / rs * (3 / 4 * np.pi) ** 3

        THIS_DIR = os.path.dirname(__file__)
        fn = os.path.join(THIS_DIR, f"comparison_data/lfc/Farid_et_al_Geek0_rs={rs:.0f}.csv")
        dat = np.genfromtxt(fn, delimiter=",")
        plt.plot(ks / kF, lfcs, label=f"rs={rs}", ls="-.", c=c)
        plt.plot(ks / kF, lfcs_iu, label=f"UI: rs={rs}", ls=":", c=c)
        plt.scatter(dat[:, 0], dat[:, 1], label=f"Farid et al., rs={rs}", c=c)

    plt.legend()
    plt.ylim(-0.1, 1.3)
    plt.xlim(0, 6)
    plt.ylabel(r"$G_{ee}(k)$")
    plt.xlabel(r"$k/k_F$")
    plt.show()


def test_dornheim_2021():

    rs = 2
    theta1 = 1
    theta2 = 4

    rho1, T1 = get_rho_T_from_rs_theta(rs=rs, theta=theta1)
    rho2, T2 = get_rho_T_from_rs_theta(rs=rs, theta=theta2)

    state1 = PlasmaState(
        electron_temperature=T1 * eV_TO_K,
        mass_density=rho1 * g_per_cm3_TO_kg_per_m3,
        ion_temperature=T1 * eV_TO_K,
        charge_state=1.0,
        atomic_mass=1,
        atomic_number=1,
        binding_energies=None,
    )
    kernel1 = LFC(state=state1)

    state2 = PlasmaState(
        electron_temperature=T2 * eV_TO_K,
        mass_density=rho2 * g_per_cm3_TO_kg_per_m3,
        ion_temperature=T2 * eV_TO_K,
        charge_state=1.0,
        atomic_mass=1,
        atomic_number=1,
        binding_energies=None,
    )
    kernel2 = LFC(state=state2)

    ks = np.linspace(0.01, 100, 100) / BOHR_RADIUS
    lfc_theta1 = np.zeros_like(ks)
    lfc_theta2 = np.zeros_like(ks)

    for i in range(0, len(ks)):
        lfc_theta1[i] = kernel1.calculate_lfc(k=ks[i], w=0, model="DORNHEIM_ESA")
        lfc_theta2[i] = kernel2.calculate_lfc(k=ks[i], w=0, model="DORNHEIM_ESA")

    THIS_DIR = os.path.dirname(__file__)
    if rs == 2:
        fn = os.path.join(THIS_DIR, f"comparison_data/lfc/Dornheim_2021_Fig7b")
        dat_theta1 = np.genfromtxt(fn + f"_theta_{theta1:.0f}.csv", delimiter=",")
        dat_theta2 = np.genfromtxt(fn + f"_theta_{theta2:.0f}.csv", delimiter=",")
    elif rs == 5:
        fn = os.paht.join(THIS_DIR, f"comparison_data/lfc/Dornheim_et_al_Geek0_rs=5")
        dat_theta1 = np.genfromtxt(fn + f"_Theta={theta1:.0f}.csv", delimiter=",")
        dat_theta2 = np.genfromtxt(fn + f"_Theta={theta2:.0f}.csv", delimiter=",")

    kF1 = 1 / rs * (3 / 4 * np.pi) ** 3

    plt.figure()
    plt.plot(
        ks * BOHR_RADIUS,
        lfc_theta1,
        label=rf"$\theta$={theta1}",
        marker="x",
    )
    plt.plot(
        ks * BOHR_RADIUS,
        lfc_theta2,
        label=rf"$\theta$={theta2}",
        marker="x",
    )
    plt.plot(dat_theta1[:, 0] * kF1, dat_theta1[:, 1], label=f"Dornheim et al., theta={theta1}")
    plt.plot(dat_theta2[:, 0] * kF1, dat_theta2[:, 1], label=f"Dornheim et al., theta={theta2}")
    plt.title(rf"$r_s$={rs}")
    plt.ylabel(r"$G_{ee}(k)$")
    plt.xlabel(r"$k$ [$a_B^{-1}$]")
    plt.legend()
    plt.show()

    ks_bohr = ks * BOHR_RADIUS

    if not np.isclose(
        lfc_theta1, np.interp(x=ks_bohr, xp=dat_theta1[:, 0] * kF1, fp=dat_theta1[:, 1]), atol=1.0e-2
    ).all():
        # print(np.abs(lfc_theta1 - np.interp(x=ks_bohr, xp=dat_theta1[:, 0] * kF1, fp=dat_theta1[:, 1])))
        print(f"Dornheim LFC test has failed for theta = 1.")
    if not np.isclose(
        lfc_theta2, np.interp(x=ks_bohr, xp=dat_theta2[:, 0] * kF1, fp=dat_theta2[:, 1]), atol=1.0e-1
    ).all():
        # print(np.abs(lfc_theta2 - np.interp(x=ks_bohr, xp=dat_theta2[:, 0] * kF1, fp=dat_theta2[:, 1])))
        print(f"Dornheim LFC test has failed for theta = 2.")


def test_ui():

    rss = np.array([1, 4, 10])

    colors = ["navy", "crimson", "magenta", "dodgerblue", "limegreen"]
    THIS_DIR = os.path.dirname(__file__)

    plt.figure()

    for rs, c in zip(rss, colors):
        rho, T = get_rho_T_from_rs_theta(rs=rs, theta=1)
        xrts_code = xDave(
            mass_density=rho,
            electron_temperature=T,
            ion_temperature=T,
            elements=np.array(["H"]),
            partial_densities=np.array([1]),
            charge_states=np.array([1]),
            models=ModelOptions(),
        )
        state = xrts_code.overlord_state
        kF = state.fermi_wave_number(state.free_electron_number_density)
        ks = np.linspace(0.01, 5, 500) * kF
        lfcs_iu = np.zeros_like(ks)

        kernel = LFC(state=state)

        for i in range(0, len(ks)):
            lfcs_iu[i] = kernel.calculate_lfc(k=ks[i], w=0, model="UI")

        fn = os.path.join(THIS_DIR, f"comparison_data/lfc/Utsumi_Ichimaru_Geek0_rs={rs:.0f}.csv")
        dat = np.genfromtxt(fn, delimiter=",")
        plt.plot(ks / kF, lfcs_iu, label=f"UI: rs={rs}", ls=":", c=c)
        plt.scatter(dat[:, 0], dat[:, 1], label=f"Ichimaru et al., rs={rs}", c=c)

    plt.legend()
    plt.xlim(0, 5)
    plt.ylabel(r"$G_{ee}(k)$")
    plt.xlabel(r"$k/k_F$")
    plt.show()


def test_gv():

    theta = 1
    rss = np.array([2, 3])

    colors = ["navy", "crimson", "magenta", "dodgerblue", "limegreen"]

    plt.figure()

    for rs, c in zip(rss, colors):
        rho, T = get_rho_T_from_rs_theta(rs=rs, theta=1)
        xrts_code = xDave(
            mass_density=rho,
            electron_temperature=T,
            ion_temperature=T,
            elements=np.array(["H"]),
            partial_densities=np.array([1]),
            charge_states=np.array([1]),
            models=ModelOptions(),
        )
        state = xrts_code.overlord_state
        kF = state.fermi_wave_number(state.free_electron_number_density)
        ks = np.linspace(0, 5, 500) * kF
        # lfcs = np.zeros_like(ks)
        lfcs_iu = np.zeros_like(ks)

        kernel = LFC(state=state)

        for i in range(0, len(ks)):
            # lfcs[i] = kernel.calculate_lfc(k=ks[i], w=0, model="FARID")
            lfcs_iu[i] = kernel.calculate_lfc(k=ks[i], w=0, model="GV")

        fn = os.path.join(os.path.dirname(__file__), f"comparison_data/lfc/mcss_tests/lfc=gv_rs={rs:.0f}_theta=1.csv")
        dat = np.genfromtxt(fn, delimiter=",", skip_header=1)
        plt.plot(ks / kF, lfcs_iu, label=f"GV: rs={rs}", ls=":", c=c)
        plt.plot(dat[:, 0], dat[:, -1], label=f"MCSS, rs={rs}", ls="solid", c=c)

    plt.legend()
    plt.xlim(0, 5)
    plt.ylabel(r"$G_{ee}(k)$")
    plt.xlabel(r"$k/k_F$")
    plt.show()


if __name__ == "__main__":
    # test()
    # test_ui_gv_mcss()
    # test_gv()
    # test_gregori_2007()
    test_fortmann_2010()
    test_dornheim_2021()
    # test_farid()
    # test_ui()

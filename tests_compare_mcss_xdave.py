import sys

sys.path.insert(1, "/home/bellen85/code/dev/xdave/xdave")
sys.path.insert(1, "/home/bellen85/code/dev/xdave/mcss_tests")

from xdave import *

# from utils import calculate_q
from mcss_tests.run_mcss_sim import run_be_sr_mode, run_ch_sr_mode, run_c_sr_mode, run_ch_ar_mode

import numpy as np
import matplotlib.pyplot as plt
import os

from datetime import datetime


THIS_DIR = os.path.dirname(__file__)

mcss_dir = "~/code/mcss/mcss_ndtt/pro/mcss"
mcss_executable = "mcss_60"  # "mcss_ndtt"  'mcss_51'


def compare_mcss_xdave_be():
    plt.style.use("~/Desktop/resources/plotting/my_style.mplstyle")
    T = 155.5  # eV
    rho = 30.0  # g/cc
    Z = 3.5
    angle = 75  # degrees
    beam_energy = 20.0e3  # eV
    q = calculate_q(angle=angle, energy=beam_energy)
    print(f"Running at q={q:.3f}")

    # try:
    #     fname = "mcss_tests/be_runs_T=155.50_rho=30.00/mcss_run_be_T=155.50_rho=30.00_Z=3.0_angle=75.csv"
    #     En_mcss, wff_mcss, wbf_mcss, ff_mcss, bf_mcss, el_mcss = load_mcss_result(filename=fname)
    #     WR_mcss = get_mcss_wr_from_status_file(
    #         status_file="mcss_tests/be_runs_T=155.50_rho=30.00/mcss_run_be_T=155.50_rho=30.00_Z=3.0_angle=75_status.txt"
    #     )
    # except FileNotFoundError:
    En_mcss, wff_mcss, wbf_mcss, ff_mcss, bf_mcss, el_mcss, WR_mcss = run_be_sr_mode(
        T=T, rho=rho, Z=Z, angle=angle, user_defined_ipd=0.0, user_defined_lfc=0.0, plot=False
    )

    mcss_norm = 4

    elements = np.array(["Be", "Be"])
    partial_densities = np.array([0.5, 0.5])
    charge_states = np.array([3, 4])

    models = ModelOptions(polarisation_model="NUMERICAL", bf_model="SCHUMACHER", lfc_model="NONE", ipd_model="NONE")
    k = q  # 1/aB

    omega_array = np.arange(-4000, 4000, 0.5)  # eV
    kernel = xDave(
        models=models,
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        elements=elements,
        charge_states=charge_states,
        partial_densities=partial_densities,
        user_defined_inputs=None,
    )

    bf_tot, ff_tot, dsf, WR, ff_i, bf_i = kernel.run(k=k, w=omega_array)
    ff_tot[np.isnan(ff_tot)] = 0.0

    # plot results
    fig, axes = plt.subplots(2, 3, figsize=(16, 16))

    ax = axes[0, 0]
    ax.set_title("Total DSF")
    ax.plot(omega_array, dsf, label="Inel", ls="-.", c="magenta")
    ax.plot(En_mcss, (wbf_mcss + wff_mcss) / mcss_norm, ls=":", c="purple", label="MCSS / AN")
    ax.legend()

    ax = axes[0, 1]
    ax.set_title("FF DSF")
    ax.plot(omega_array, ff_tot, label="FF", ls="--", c="orange")
    ax.plot(En_mcss, wff_mcss / mcss_norm, c="navy", ls=":", label="MCSS: ff")
    ax.legend()

    ax = axes[0, 2]
    ax.set_title("BF DSF")
    ax.plot(omega_array, bf_tot, label="BF", ls="solid", c="dodgerblue")
    ax.plot(En_mcss, wbf_mcss / mcss_norm, c="brown", ls=":", label="MCSS: bf")
    ax.legend()

    tau_array, F_tot_inel, F_wff, F_wbf = kernel.get_itcf(w=omega_array, ff=ff_tot, bf=bf_tot)
    tau_array, F_tot_inel_mcss, F_wff_mcss, F_wbf_mcss = kernel.get_itcf(w=En_mcss, ff=wff_mcss, bf=wbf_mcss)

    ax = axes[1, 0]
    ax.set_title("ITCF")
    ax.plot(tau_array, F_tot_inel, label="xDave inel", ls="dashed", c="magenta")
    ax.plot(tau_array, F_tot_inel_mcss / mcss_norm, label="MCSS inel", ls="dotted", c="purple")
    ax.legend()

    ax = axes[1, 1]
    ax.set_title("FF ITCF")
    ax.plot(tau_array, F_wff, label="xDave ff", ls="dashed", c="dodgerblue")
    ax.plot(tau_array, F_wff_mcss / mcss_norm, label="MCSS ff", ls="dotted", c="navy")
    ax.legend()

    ax = axes[1, 2]
    ax.set_title("BF ITCF")
    ax.plot(tau_array, F_wbf, label="xDave bf", ls="dashed", c="orange")
    ax.plot(tau_array, F_wbf_mcss / mcss_norm, label="MCSS bf", ls="dotted", c="brown")
    ax.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(f"be_test_T={T*K_TO_eV:.1f}_rho={rho*kg_per_m3_TO_g_per_cm3:.1f}_Z={Z}_q={q:.2f}_wrong.pdf")


def compare_mcss_xdave_c():
    plt.style.use("~/Desktop/resources/plotting/my_style.mplstyle")
    T = 80.0  # eV
    rho = 4.0  # g/cc
    Z = 0.5
    angle = 90  # degrees
    beam_energy = 20.0e3  # eV
    q = calculate_q(angle=angle, energy=beam_energy)
    print(f"Running at q={q:.3f}")

    En_mcss, wff_mcss, wbf_mcss, ff_mcss, bf_mcss, el_mcss, WR_mcss = run_c_sr_mode(
        T=T, rho=rho, Z=Z, angle=angle, user_defined_ipd=0.0, user_defined_lfc=0.0, plot=False
    )

    Z_min, Z_max, x1, x2 = get_fractions_from_Z(Z)

    elements = np.array(["C", "C"])

    partial_densities = np.array([x1, x2])
    charge_states = np.array([Z_min, Z_max])
    user_defined_inputs = dict()

    models = ModelOptions(polarisation_model="NUMERICAL", bf_model="SCHUMACHER", lfc_model="NONE", ipd_model="NONE")

    k = q  # 1/aB

    omega_array = np.arange(-4000, 4000, 0.5)  # eV

    kernel = xDave(
        models=models,
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        partial_densities=partial_densities,
        charge_states=charge_states,
        elements=elements,
        user_defined_inputs=user_defined_inputs,
    )

    mcss_norm = kernel.overlord_state.atomic_number

    bf_tot, ff_tot, dsf, WR, ff_i, bf_i = kernel.run(k=k, w=omega_array)
    ff_tot[np.isnan(ff_tot)] = 0.0

    print(f"Calculated Rayleigh weight = {WR}")

    # plot results
    fig, axes = plt.subplots(2, 3, figsize=(16, 16))

    ax = axes[0, 0]
    ax.set_title("Total DSF")
    ax.plot(omega_array, dsf, label="Inel", ls="-.", c="magenta")
    ax.plot(En_mcss, (wbf_mcss + wff_mcss) / mcss_norm, ls=":", c="purple", label="MCSS / AN")
    ax.legend()

    ax = axes[0, 1]
    ax.set_title("FF DSF")
    ax.plot(omega_array, ff_tot, label="FF", ls="--", c="orange")
    ax.plot(En_mcss, wff_mcss / mcss_norm, c="brown", ls=":", label="MCSS: ff")
    ax.legend()

    ax = axes[0, 2]
    ax.set_title("BF DSF")
    ax.plot(omega_array, bf_tot, label="BF", ls="solid", c="dodgerblue")
    ax.plot(En_mcss, wbf_mcss / mcss_norm, c="navy", ls=":", label="MCSS: bf")
    ax.legend()

    tau_array, F_tot_inel, F_wff, F_wbf = kernel.get_itcf(w=omega_array, ff=ff_tot, bf=bf_tot)
    tau_array, F_tot_inel_mcss, F_wff_mcss, F_wbf_mcss = kernel.get_itcf(w=En_mcss, ff=wff_mcss, bf=wbf_mcss)

    ax = axes[1, 0]
    ax.set_title("ITCF")
    ax.plot(tau_array, F_tot_inel, label="xDave inel", ls="dashed", c="magenta")
    ax.plot(tau_array, F_tot_inel_mcss / mcss_norm, label="MCSS inel", ls="dotted", c="purple")
    ax.axhline(WR, label=f"WR", c="navy", ls="-.")
    ax.axhline(WR_mcss / mcss_norm, label=f"MCSS: WR", c="dodgerblue", ls=":")
    ax.legend()

    ax = axes[1, 1]
    ax.set_title("FF ITCF")
    ax.plot(tau_array, F_wff, label="xDave ff", ls="dashed", c="dodgerblue")
    ax.plot(tau_array, F_wff_mcss / mcss_norm, label="MCSS ff", ls="dotted", c="navy")
    ax.legend()

    ax = axes[1, 2]
    ax.set_title("BF ITCF")
    ax.plot(tau_array, F_wbf, label="xDave bf", ls="dashed", c="orange")
    ax.plot(tau_array, F_wbf_mcss / mcss_norm, label="MCSS bf", ls="dotted", c="brown")
    ax.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(f"c_test_T={T*K_TO_eV:.1f}_rho={rho*kg_per_m3_TO_g_per_cm3:.1f}_Z={Z}_q={q:.2f}.pdf")


def compare_mcss_xdave_ch():
    plt.style.use("~/Desktop/resources/plotting/my_style.mplstyle")
    T = 80.0  # eV
    rho = 3.5  # g/cc
    ZC = 3.0
    ZH = 1.0
    xH = 0.2
    # q = 4.0
    angle = 75  # degrees
    beam_energy = 20.0e3  # eV
    q = calculate_q(angle=angle, energy=beam_energy)
    print(f"Running at q={q:.3f}")

    En_mcss, wff_mcss, wbf_mcss, ff_mcss, bf_mcss, el_mcss, WR_mcss = run_ch_sr_mode(
        T=T, rho=rho, xH=xH, ZH=ZH, ZC=ZC, angle=angle, user_defined_ipd=0.0, user_defined_lfc=0.0, plot=False
    )

    elements = np.array(["H", "C", "C"])

    Zmin, Zmax, xmin, xmax = get_fractions_from_Z_partial(ZC, x0=xH)
    partial_densities = np.array([xH, xmin, xmax])
    charge_states = np.array([1.0, Zmin, Zmax])
    user_defined_inputs = dict()

    models = ModelOptions(
        polarisation_model="NUMERICAL",
        bf_model="SCHUMACHER",
        lfc_model="NONE",
        ipd_model="NONE",
        screening_model="DEBYE_HUCKEL",
    )

    k = q  # 1/aB

    omega_array = np.arange(-4000, 4000, 0.5)  # eV

    kernel = xDave(
        models=models,
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_states=charge_states,
        elements=elements,
        partial_densities=partial_densities,
        user_defined_inputs=None,
    )

    mcss_norm = kernel.overlord_state.atomic_number
    # print(f"MCSS: WR = {WR_mcss}, WR/AN = {WR_mcss / kernel.overlord_state.charge_state}")

    bf_tot, ff_tot, dsf, WR, ff_i, bf_i = kernel.run(k=k, w=omega_array)
    ff_tot[np.isnan(ff_tot)] = 0.0

    print(f"Calculated Rayleigh weight = {WR}")

    # plot results
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    ax = axes[0, 0]
    ax.set_title("Total DSF")
    # ax.set_yscale("log")
    ax.plot(omega_array, dsf, label="Inel", ls="-.", c="magenta")
    ax.plot(En_mcss, (wbf_mcss + wff_mcss) / mcss_norm, ls=":", c="purple", label="MCSS / AN")
    ax.legend()
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.set_ylabel(r"DSF [1/eV]")

    ax = axes[0, 1]
    # ax.set_yscale("log")
    ax.set_title("FF DSF")
    ax.plot(omega_array, ff_tot, label="FF", ls="--", c="orange")
    ax.plot(En_mcss, wff_mcss / mcss_norm, c="brown", ls=":", label="MCSS: ff")
    ax.legend()
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.set_ylabel(r"DSF [1/eV]")

    ax = axes[0, 2]
    ax.set_title("BF DSF")
    # ax.set_yscale("log")
    ax.plot(omega_array, bf_tot, label="BF", ls="solid", c="dodgerblue")
    ax.plot(omega_array, bf_i[0], label="BF: H", ls="-.", c="magenta")
    ax.plot(omega_array, bf_i[2], label="BF: C", ls="-.", c="purple")
    ax.plot(En_mcss, wbf_mcss / mcss_norm, c="navy", ls=":", label="MCSS: bf")
    ax.legend()
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.set_ylabel(r"DSF [1/eV]")

    tau_array, F_tot_inel, F_wff, F_wbf = kernel.get_itcf(w=omega_array, ff=ff_tot, bf=bf_tot)
    tau_array, F_tot_inel_mcss, F_wff_mcss, F_wbf_mcss = kernel.get_itcf(w=En_mcss, ff=wff_mcss, bf=wbf_mcss)

    ax = axes[1, 0]
    # ax.set_title("ITCF")
    ax.plot(tau_array, F_tot_inel, label="xDave inel", ls="dashed", c="magenta")
    ax.plot(tau_array, F_tot_inel_mcss / mcss_norm, label="MCSS inel", ls="dotted", c="purple")
    ax.axhline(WR_mcss, c="black", ls="dotted", label="MCSS WR")
    ax.axhline(WR, c="slategrey", ls="dashed", label="WR")
    ax.legend()
    ax.set_xlabel(r"$\tau$ [1/eV]")
    ax.set_ylabel(r"ITCF")

    ax = axes[1, 1]
    # ax.set_title("FF ITCF")
    ax.plot(tau_array, F_wff, label="xDave ff", ls="dashed", c="dodgerblue")
    ax.plot(tau_array, F_wff_mcss / mcss_norm, label="MCSS ff", ls="dotted", c="navy")
    ax.legend()
    ax.set_xlabel(r"$\tau$ [1/eV]")
    ax.set_ylabel(r"ITCF")

    ax = axes[1, 2]
    # ax.set_title("BF ITCF")
    ax.plot(tau_array, F_wbf, label="xDave bf", ls="dashed", c="orange")
    ax.plot(tau_array, F_wbf_mcss / mcss_norm, label="MCSS bf", ls="dotted", c="brown")
    ax.legend()
    ax.set_xlabel(r"$\tau$ [1/eV]")
    ax.set_ylabel(r"ITCF")
    plt.tight_layout()
    plt.show()
    fig.savefig(f"ch_test_T={T:.1f}_rho={rho:.1f}_ZC={ZC}_q={q:.2f}.pdf")

    fname = f"xdave_ch_T={T:.1f}_rho={rho:.1f}_ZC={ZC}"
    dirname = "/home/bellen85/code/dev/xdave/xdave_results"
    kernel.save_result(
        fname=fname,
        dirname=dirname,
        w=omega_array,
        tau=tau_array,
        bf=bf_tot,
        ff=ff_tot,
        dsf=dsf,
        F_inel=F_tot_inel,
        F_bf=F_wbf,
        F_ff=F_wff,
    )


def compare_mcss_xdave_ch_static():
    from scipy.interpolate import interp1d

    # plt.style.use("~/Desktop/resources/plotting/my_style.mplstyle")

    T = 100.0  # eV
    rho = 1.2  # g/cc
    ZC = 3
    ZH = 1.0
    xH = 0.5
    xC = 1 - xH
    # q = 4.0
    angle = 75  # degrees
    beam_energy = 20.0e3  # eV
    q = calculate_q(angle=angle, energy=beam_energy)
    print(f"Running at q={q:.3f}")

    k_mcss, WR_mcss, f1_mcss, f2_mcss, q1_mcss, q2_mcss, S11_mcss, S12_mcss, S22_mcss = run_ch_ar_mode(
        T=T, rho=rho, xH=xH, ZH=ZH, ZC=ZC, angle=angle, user_defined_ipd=0.0, user_defined_lfc=0.0, plot=False
    )

    elements = np.array(["H", "C"])

    partial_densities = np.array([xH, xC])
    charge_states = np.array([ZH, ZC])
    user_defined_inputs = dict()

    models = ModelOptions(
        polarisation_model="NUMERICAL",
        bf_model="SCHUMACHER",
        lfc_model="NONE",
        ipd_model="NONE",
        # screening_model="DEBYE_HUCKEL",
    )

    k = q  # 1/aB

    kernel = xDave(
        models=models,
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_states=charge_states,
        elements=elements,
        partial_densities=partial_densities,
        user_defined_inputs=user_defined_inputs,
    )

    mcss_norm = kernel.overlord_state.atomic_number

    k = np.linspace(0.1, 15, 10000)
    k, Sab, WR, qs, fs = kernel.run_static_mode(k=k)

    q1_interp = interp1d(k_mcss, q1_mcss, fill_value="extrapolate")
    q1_new = q1_interp(k)
    q2_interp = interp1d(k_mcss, q2_mcss, fill_value="extrapolate")
    q2_new = q2_interp(k)
    S11_interp = interp1d(k_mcss, S11_mcss, fill_value="extrapolate")
    S11_new = S11_interp(k)
    S12_interp = interp1d(k_mcss, S12_mcss, fill_value="extrapolate")
    S12_new = S12_interp(k)
    S22_interp = interp1d(k_mcss, S22_mcss, fill_value="extrapolate")
    S22_new = S22_interp(k)

    # WR_test = np.sqrt(xH * xC) * (
    #     (qs[0] + fs[0]) ** 2 * Sab[0, 0, :]
    #     + (qs[1] + fs[1]) ** 2 * Sab[1, 1, :]
    #     + 2 * (qs[0] + fs[0]) * (qs[1] + fs[1]) * Sab[0, 1, :]
    # )
    WR_test = np.sqrt(xH * xC) * (
        (q1_new + fs[0]) ** 2 * Sab[0, 0, :]
        + (q2_new + fs[1]) ** 2 * Sab[1, 1, :]
        + 2 * (q1_new + fs[0]) * (q2_new + fs[1]) * Sab[0, 1, :]
    )

    print(q1_new / qs[0])
    print(q2_new / qs[1])

    # plot result
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    ax = axes[0, 0]
    ax.plot(k, Sab[0, 0, :], label="HH", c="crimson", ls="-.")
    ax.plot(k, Sab[0, 1, :], label="CH", c="navy", ls="-.")
    ax.plot(k, Sab[1, 1, :], label="CC", c="forestgreen", ls="-.")
    ax.plot(k_mcss, S11_mcss, label="MCSS: HH", c="crimson", ls=":")
    ax.plot(k_mcss, S12_mcss, label="MCSS: CH", c="navy", ls=":")
    ax.plot(k_mcss, S22_mcss, label="MCSS: CC", c="forestgreen", ls=":")
    ax.legend()
    ax.set_xlabel(r"$k$ [$a_B^{-1}$]")
    ax.set_ylabel(r"$S_{ab}$ [ ]")

    ax = axes[0, 1]
    ax.plot(k, qs[0], label="H", c="crimson", ls="-.")
    ax.plot(k, qs[1], label="C", c="forestgreen", ls="-.")
    ax.plot(k_mcss, q1_mcss, label="MCSS: H", c="crimson", ls=":")
    ax.plot(k_mcss, q2_mcss, label="MCSS: C", c="forestgreen", ls=":")
    # ax.plot(k_mcss, q1_new, label="MCSS interp: H", c="crimson", ls="solid")
    # ax.plot(k_mcss, q2_new, label="MCSS interp: C", c="forestgreen", ls="solid")
    ax.set_xlabel(r"$k$ [$a_B^{-1}$]")
    ax.set_ylabel(r"$q_{a}$ [ ]")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(k, fs[0], label="H", c="crimson", ls="-.")
    ax.plot(k, fs[1], label="C", c="forestgreen", ls="-.")
    ax.plot(k_mcss, f1_mcss, label="MCSS: H", c="crimson", ls=":")
    ax.plot(k_mcss, f2_mcss, label="MCSS: C", c="forestgreen", ls=":")
    ax.set_xlabel(r"$k$ [$a_B^{-1}$]")
    ax.set_ylabel(r"$f_{a}$ [ ]")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(k, WR, label=r"$W_R$", c="darkgreen", ls="-.")
    ax.plot(k_mcss, WR_mcss, label=r"MCSS: $W_R$", c="limegreen", ls=":")
    ax.plot(k, WR_test, label=r"Test: $W_R$", c="darkgreen", ls="solid")
    ax.set_xlabel(r"$k$ [$a_B^{-1}$]")
    ax.set_ylabel(r"$W_R$ [ ]")
    ax.legend()

    plt.tight_layout()
    plt.show()
    date = datetime.today().strftime("%Y-%m-%d")
    fig.savefig(f"/home/bellen85/code/dev/xdave/ch_test_T={T:.1f}_rho={rho:.1f}_ZC={ZC}_q={q:.2f}_static.pdf")

    # plt.figure()
    # plt.plot(k, np.abs(q1_new - qs[0]), label="H")
    # plt.plot(k, np.abs(q2_new - qs[1]), label="C")
    # plt.legend()
    # plt.xlabel(r"$k$ [$a_B^{-1}$]")
    # plt.ylabel(r"$q_{mcss} - q_{xdave}$")
    # plt.xlim(0, 10.0)
    # # plt.ylim(0, 10)
    # plt.show()


if __name__ == "__main__":
    # compare_mcss_xdave_be()
    # compare_mcss_xdave_ch()
    # compare_mcss_xdave_c()
    compare_mcss_xdave_ch_static()

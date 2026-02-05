from xdave import *
from xdave.plasma_state import get_fractions_from_Z_partial, get_fractions_from_Z
from xdave.utils import (
    calculate_q,
    load_mcss_result,
    get_mcss_wr_from_status_file,
    load_mcss_result_ar,
    load_mcss_result_ar_3species,
)
from xdave.unit_conversions import K_TO_eV, kg_per_m3_TO_g_per_cm3

import numpy as np
import matplotlib.pyplot as plt
import os

from datetime import datetime


THIS_DIR = os.path.dirname(__file__)


def compare_mcss_xdave_be():
    T = 155.5  # eV
    rho = 30.0  # g/cc
    Z = 3.5  #
    angle = 75  # degrees, also can run 120
    beam_energy = 20.0e3  # eV
    q = calculate_q(angle=angle, energy=beam_energy)
    print(f"Running at q={q:.3f}")

    fn = os.path.join(
        os.path.dirname(__file__),
        f"comparison_data/mcss_comparisons/be_runs_T={T:.2f}_rho={rho:.2f}/mcss_run_be_T={T:.2f}_rho={rho:.2f}_Z={Z}_angle={angle:.0f}.csv",
    )
    En_mcss, wff_mcss, wbf_mcss, ff_mcss, bf_mcss, el_mcss = load_mcss_result(filename=fn)
    # WR_mcss =

    mcss_norm = 1

    # hard-coded for Z=3.5
    elements = np.array(["Be", "Be"])
    partial_densities = np.array([0.5, 0.5])
    charge_states = np.array([3, 4])

    models = ModelOptions(polarisation_model="NUMERICAL", bf_model="SCHUMACHER", lfc_model="NONE", ipd_model="NONE")
    k = q  # 1/aB

    omega_array = np.arange(-2000, 2000, 0.05)  # eV
    kernel = xDave(
        models=models,
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        elements=elements,
        charge_states=charge_states,
        partial_densities=partial_densities,
        user_defined_inputs=None,
        enforce_fsum=False,
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
    T = 80.0  # eV
    rho = 4.0  # g/cc
    Z = 4.0  # also Z = 0.5, 1.5, 2.5, 3.5, 4.5
    angle = 90  # degrees
    beam_energy = 20.0e3  # eV
    q = calculate_q(angle=angle, energy=beam_energy)
    print(f"Running at q={q:.3f}")

    fn = os.path.join(
        os.path.dirname(__file__),
        f"comparison_data/mcss_comparisons/c_runs_T={T:.2f}_rho={rho:.2f}/mcss_run_c_T={T:.2f}_rho={rho:.2f}_Z={Z}_angle={angle:.0f}.csv",
    )
    status_fn = os.path.join(
        os.path.dirname(__file__),
        f"comparison_data/mcss_comparisons/c_runs_T={T:.2f}_rho={rho:.2f}/mcss_run_c_T={T:.2f}_rho={rho:.2f}_Z={Z}_angle={angle:.0f}_status.txt",
    )
    En_mcss, wff_mcss, wbf_mcss, ff_mcss, bf_mcss, el_mcss = load_mcss_result(filename=fn)
    WR_mcss = get_mcss_wr_from_status_file(status_file=status_fn)

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

    mcss_norm = 1  # kernel.overlord_state.atomic_number

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
    ax.axhline(WR_mcss, label=f"MCSS: WR", c="dodgerblue", ls=":")
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
    T = 80.0  # eV
    rho = 3.5  # g/cc
    ZC = 4.0  # also ZC=2.5, 3.5, 4.0, 4.5
    ZH = 1.0
    xH = 0.2
    angle = 75  # degrees
    beam_energy = 20.0e3  # eV
    q = calculate_q(angle=angle, energy=beam_energy)
    print(f"Running at q={q:.3f}")

    fn = os.path.join(
        os.path.dirname(__file__),
        f"comparison_data/mcss_comparisons/ch_runs_T={T:.2f}_rho={rho:.2f}/mcss_run_ch_T={T:.2f}_rho={rho:.2f}_ZC={ZC}_angle={angle:.0f}.csv",
    )
    status_fn = os.path.join(
        os.path.dirname(__file__),
        f"comparison_data/mcss_comparisons/ch_runs_T={T:.2f}_rho={rho:.2f}/mcss_run_ch_T={T:.2f}_rho={rho:.2f}_ZC={ZC}_angle={angle:.0f}_status.txt",
    )
    En_mcss, wff_mcss, wbf_mcss, ff_mcss, bf_mcss, el_mcss = load_mcss_result(filename=fn)
    WR_mcss = get_mcss_wr_from_status_file(status_file=status_fn)

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
        enforce_fsum=False,
    )

    mcss_norm = 1  # kernel.overlord_state.atomic_number

    bf_tot, ff_tot, dsf, WR, ff_i, bf_i = kernel.run(k=k, w=omega_array)
    ff_tot[np.isnan(ff_tot)] = 0.0

    print(f"Calculated Rayleigh weight = {WR}")

    # plot results
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    ax = axes[0, 0]
    ax.set_title("Total DSF")
    ax.plot(omega_array, dsf, label="Inel", ls="-.", c="magenta")
    ax.plot(En_mcss, (wbf_mcss + wff_mcss) / mcss_norm, ls=":", c="purple", label="MCSS / AN")
    ax.legend()
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.set_ylabel(r"DSF [1/eV]")

    ax = axes[0, 1]
    ax.set_title("FF DSF")
    ax.plot(omega_array, ff_tot, label="FF", ls="--", c="orange")
    ax.plot(En_mcss, wff_mcss / mcss_norm, c="brown", ls=":", label="MCSS: ff")
    ax.legend()
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.set_ylabel(r"DSF [1/eV]")

    ax = axes[0, 2]
    ax.set_title("BF DSF")
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
    ax.plot(tau_array, F_tot_inel, label="xDave inel", ls="dashed", c="magenta")
    ax.plot(tau_array, F_tot_inel_mcss / mcss_norm, label="MCSS inel", ls="dotted", c="purple")
    ax.axhline(WR_mcss, c="black", ls="dotted", label="MCSS WR")
    ax.axhline(WR, c="slategrey", ls="dashed", label="WR")
    ax.legend()
    ax.set_xlabel(r"$\tau$ [1/eV]")
    ax.set_ylabel(r"ITCF")

    ax = axes[1, 1]
    ax.plot(tau_array, F_wff, label="xDave ff", ls="dashed", c="dodgerblue")
    ax.plot(tau_array, F_wff_mcss / mcss_norm, label="MCSS ff", ls="dotted", c="navy")
    ax.legend()
    ax.set_xlabel(r"$\tau$ [1/eV]")
    ax.set_ylabel(r"ITCF")

    ax = axes[1, 2]
    ax.plot(tau_array, F_wbf, label="xDave bf", ls="dashed", c="orange")
    ax.plot(tau_array, F_wbf_mcss / mcss_norm, label="MCSS bf", ls="dotted", c="brown")
    ax.legend()
    ax.set_xlabel(r"$\tau$ [1/eV]")
    ax.set_ylabel(r"ITCF")
    plt.tight_layout()
    plt.show()
    fig.savefig(f"ch_test_T={T:.1f}_rho={rho:.1f}_ZC={ZC}_q={q:.2f}.pdf")


def compare_mcss_xdave_ch_static():

    T = 95.0  # eV
    rho = 1.2  # g/cc
    ZC = 4.0
    ZH = 1.0
    xH = 0.2
    xC = 1 - xH
    angle = 75  # degrees
    beam_energy = 9.0e3  # eV
    q = calculate_q(angle=angle, energy=beam_energy)

    fn = os.path.join(
        os.path.dirname(__file__),
        f"comparison_data/mcss_comparisons/ch_ar_runs_T={T:.2f}_rho={rho:.2f}/mcss_ar_run_ch_T={T:.2f}_rho={rho:.2f}_ZC={ZC}.csv",
    )
    k_mcss, WR_mcss, f1_mcss, f2_mcss, q1_mcss, q2_mcss, S11_mcss, S12_mcss, S22_mcss, _ = load_mcss_result_ar(
        filename=fn, use_lfc_model=False
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
        ee_potential="COULOMB",
        ei_potential="COULOMB",
        ii_potential="YUKAWA",
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

    k = np.linspace(0.1, 15, 10000)
    k, Sab, _, WR, qs, fs, lfc = kernel.run(k=k, w=0.0, mode="STATIC")
    print(qs[0])

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
    ax.set_xlabel(r"$k$ [$a_B^{-1}$]")
    ax.set_ylabel(r"$W_R$ [ ]")
    ax.legend()

    plt.tight_layout()
    plt.show()
    date = datetime.today().strftime("%Y-%m-%d")
    fig.savefig(f"ch_test_T={T:.1f}_rho={rho:.1f}_ZC={ZC}_q={q:.2f}_static.pdf")


def compare_mcss_xdave_ch_static_partialZC():

    T = 100.0  # eV
    rho = 1.2  # g/cc
    ZC = 4.5  # also Z=3.5
    ZH = 1.0
    xH = 0.2
    xC = 1 - xH
    # q = 4.0
    angle = 75  # degrees
    beam_energy = 9.0e3  # eV
    q = calculate_q(angle=angle, energy=beam_energy)
    ZC1, ZC2, xC1, xC2 = get_fractions_from_Z_partial(Z=ZC, x0=xH)

    fn = os.path.join(
        os.path.dirname(__file__),
        f"comparison_data/mcss_comparisons/ch_ar_runs_T={T:.2f}_rho={rho:.2f}/mcss_ar_run_ch_T={T:.2f}_rho={rho:.2f}_ZC={ZC}.csv",
    )
    (
        k_mcss,
        WR_mcss,
        f1_mcss,
        f2_mcss,
        f3_mcss,
        q1_mcss,
        q2_mcss,
        q3_mcss,
        S11_mcss,
        S13_mcss,
        S12_mcss,
        S22_mcss,
        S23_mcss,
        S33_mcss,
        lfc_mcss,
    ) = load_mcss_result_ar_3species(filename=fn, use_lfc_model=False)

    elements = np.array(["H", "C", "C"])

    partial_densities = np.array([xH, xC1, xC2])
    charge_states = np.array([ZH, ZC1, ZC2])
    user_defined_inputs = dict()

    models = ModelOptions(
        polarisation_model="NUMERICAL",
        bf_model="SCHUMACHER",
        lfc_model="NONE",
        ipd_model="NONE",
        ee_potential="COULOMB",
        ei_potential="COULOMB",
        ii_potential="YUKAWA",
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

    k = np.linspace(0.1, 15, 10000)
    k, Sab, _, WR, qs, fs, lfc = kernel.run(k=k, w=0.0, mode="STATIC")
    print(qs[0])

    ZC1 = int(ZC1)
    ZC2 = int(ZC2)
    # plot result
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    ax = axes[0, 0]
    ax.plot(k, Sab[0, 0, :], label="H-H", c="crimson", ls="-.")
    ax.plot(k, Sab[0, 1, :], label=f"C{ZC1}-H", c="magenta", ls="-.")
    ax.plot(k, Sab[0, 2, :], label=f"C{ZC2}-H", c="navy", ls="-.")
    ax.plot(k, Sab[1, 1, :], label=f"C{ZC1}-C{ZC1}", c="orange", ls="-.")
    ax.plot(k, Sab[1, 2, :], label=f"C{ZC1}-C{ZC2}", c="black", ls="-.")
    ax.plot(k, Sab[2, 2, :], label=f"C{ZC2}-C{ZC2}", c="forestgreen", ls="-.")
    ax.plot(k_mcss, S11_mcss, label=f"MCSS: H-H", c="crimson", ls=":")
    ax.plot(k_mcss, S12_mcss, label=f"MCSS: C{ZC1}-H", c="magenta", ls=":")
    ax.plot(k_mcss, S13_mcss, label=f"MCSS: C{ZC2}-H", c="navy", ls=":")
    ax.plot(k_mcss, S23_mcss, label=f"MCSS: C{ZC1}-C{ZC2}", c="black", ls=":")
    ax.plot(k_mcss, S22_mcss, label=f"MCSS: C{ZC1}-C{ZC2}", c="orange", ls=":")
    ax.plot(k_mcss, S33_mcss, label=f"MCSS: C{ZC2}-C{ZC2}", c="forestgreen", ls=":")
    ax.legend()
    ax.set_xlabel(r"$k$ [$a_B^{-1}$]")
    ax.set_ylabel(r"$S_{ab}$ [ ]")

    ax = axes[0, 1]
    ax.plot(k, qs[0], label="H", c="crimson", ls="-.")
    ax.plot(k, qs[1], label=f"C{ZC1}", c="forestgreen", ls="-.")
    ax.plot(k, qs[2], label=f"C{ZC2}", c="navy", ls="-.")
    ax.plot(k_mcss, q1_mcss, label=f"MCSS: H", c="crimson", ls=":")
    ax.plot(k_mcss, q2_mcss, label=f"MCSS: C{ZC1}", c="forestgreen", ls=":")
    ax.plot(k_mcss, q3_mcss, label=f"MCSS: C{ZC2}", c="navy", ls=":")
    ax.set_xlabel(r"$k$ [$a_B^{-1}$]")
    ax.set_ylabel(r"$q_{a}$ [ ]")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(k, fs[0], label="H", c="crimson", ls="-.")
    ax.plot(k, fs[1], label=f"C{ZC1}", c="forestgreen", ls="-.")
    ax.plot(k, fs[2], label=f"C{ZC2}", c="navy", ls="-.")
    ax.plot(k_mcss, f1_mcss, label=f"MCSS: H", c="crimson", ls=":")
    ax.plot(k_mcss, f2_mcss, label=f"MCSS: C{ZC1}", c="forestgreen", ls=":")
    ax.plot(k_mcss, f3_mcss, label=f"MCSS: C{ZC2}", c="navy", ls=":")
    ax.set_xlabel(r"$k$ [$a_B^{-1}$]")
    ax.set_ylabel(r"$f_{a}$ [ ]")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(k, WR, label=r"$W_R$", c="darkgreen", ls="-.")
    ax.plot(k_mcss, WR_mcss, label=r"MCSS: $W_R$", c="limegreen", ls=":")
    ax.set_xlabel(r"$k$ [$a_B^{-1}$]")
    ax.set_ylabel(r"$W_R$ [ ]")
    ax.legend()

    plt.tight_layout()
    plt.show()
    date = datetime.today().strftime("%Y-%m-%d")
    fig.savefig(f"ch_test_T={T:.1f}_rho={rho:.1f}_ZC={ZC}_q={q:.2f}_static.pdf")


if __name__ == "__main__":
    # compare_mcss_xdave_be()
    # compare_mcss_xdave_ch()
    # compare_mcss_xdave_c()
    # compare_mcss_xdave_ch_static()
    compare_mcss_xdave_ch_static_partialZC()

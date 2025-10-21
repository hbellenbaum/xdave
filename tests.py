import sys

sys.path.insert(1, "/home/bellen85/code/dev/xdave/xdave")
sys.path.insert(1, "/home/bellen85/code/dev/xdave/mcss_tests")


from plasma_state import PlasmaState, get_rho_T_from_rs_theta, get_fractions_from_Z, get_rho_T_from_rs_theta_SI
from models import ModelOptions
from unit_conversions import *
from constants import BOHR_RADIUS, PLANCK_CONSTANT, ELECTRON_MASS
from freefree_dsf import FreeFreeDSF
from boundfree_dsf import BoundFreeDSF
from utils import calculate_angle, calculate_q, load_itcf_from_file, load_mcss_result
from xdave import xDave

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


def test_chemical_potential():
    T = 1.0 * eV_TO_K
    rho = 0.01 * g_per_cm3_TO_kg_per_m3
    Z = 1.0
    AN = 1
    atomic_mass = 1.0 * amu_TO_kg
    state = PlasmaState(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_state=Z,
        atomic_mass=atomic_mass,
        atomic_number=AN,
    )
    n = state.total_electron_number_density

    TF_erg = state.fermi_temperature(mass=ELECTRON_MASS, number_density=n) * K_TO_erg
    EF = state.fermi_energy(mass=ELECTRON_MASS, number_density=n)
    EF_erg = EF * J_TO_erg
    Thetas = np.linspace(0.01, 4, 100)
    mus = np.zeros_like(Thetas)
    mus_fit = np.zeros_like(Thetas)
    mus_high = np.zeros_like(Thetas)
    mus_low = np.zeros_like(Thetas)
    mus_classical = np.zeros_like(Thetas)

    for i in range(0, len(Thetas)):
        theta = Thetas[i]
        T = theta * TF_erg * erg_TO_K
        state.electron_temperature = T
        mu_young, mu_high, mu_low = state.chemical_potential(temperature=T, number_density=n, mass=ELECTRON_MASS)
        mus_fit[i] = state.chemical_potential_ichimaru(temperature=T, number_density=n, mass=ELECTRON_MASS)
        mus_classical[i] = state.chemical_potential_classical(temperature=T, number_density=n, mass=ELECTRON_MASS)
        mus[i] = mu_young
        mus_high[i] = mu_high
        mus_low[i] = mu_low

    mus /= EF_erg
    mus_low /= EF_erg
    mus_high /= EF_erg
    mus_classical /= EF_erg
    mus_fit /= EF

    print(mus_fit)
    print(mus)
    plt.figure(figsize=(8, 13))
    plt.xlabel(r"$T/T_F$")
    plt.ylabel(r"$\mu / E_F$")
    plt.plot(Thetas, mus, label="numerical", ls="-.", lw=3)
    plt.plot(Thetas, mus_fit, label="fit", ls="--", lw=3)
    plt.plot(Thetas, mus_low, label="low T", ls=":", lw=3)
    plt.plot(Thetas, mus_high, label="high T", ls=":", lw=3)
    plt.plot(Thetas, mus_classical, label="classical", ls=":", lw=3)
    plt.ylim(-30.0, 1.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"chemical_potential_models_compared_rho={rho*kg_per_m3_TO_g_per_cm3:.2f}.pdf")


def test_ff_mcss():
    rs = 2
    theta = 1
    rho, Te = get_rho_T_from_rs_theta(rs=rs, theta=theta)
    ks = np.array((0.5,)) / BOHR_RADIUS
    rho *= g_per_cm3_TO_kg_per_m3
    Te *= eV_TO_K
    charge_state = 1.0
    atomic_mass = 1.0
    atomic_number = 1.0
    lfc = 0.0

    omega_array = np.linspace(-150, 300, 5000) * eV_TO_J
    state = PlasmaState(
        electron_temperature=Te,
        ion_temperature=Te,
        mass_density=rho,
        charge_state=charge_state,
        atomic_mass=atomic_mass,
        atomic_number=atomic_number,
    )
    model = "DANDREA_FIT"
    if model == "LINDHARD":
        mcss_model = "LINDHARD_RPA"
    elif model == "DANDREA_FIT":
        mcss_model = "DANDREA_RPA_FIT"
    elif model == "NUMERICAL_RPA":
        mcss_model = "NUMERICAL_RPA"
    models = ModelOptions(polarisation_model=model)
    colors = ["magenta", "crimson", "orange", "dodgerblue", "lightgreen", "lightgray", "yellow", "cyan"]
    fig, ax0 = plt.subplots(figsize=(14, 10))
    i = 0
    Hz_TO_eV = 4.1357e-15  # eV

    norm_factor = PLANCK_CONSTANT

    print(f"\nNormalised using factor = {norm_factor}\n")

    for k, c in zip(ks, colors):
        q = k * BOHR_RADIUS
        angle = calculate_angle(q=q, energy=8.0e3)
        angle = int(np.round(angle, 0))
        dsfs = []
        dsfs = np.zeros_like(omega_array)
        for i in range(0, len(omega_array)):
            omega = omega_array[i]
            kernel = FreeFreeDSF(state=state, models=models)
            dsf = kernel.get_dsf(k=k, w=omega, lfc=lfc)
            dsfs[i] = dsf

        # Run MCSS
        mcss_fn = f"mcss_tests/mcss_outputs_model={mcss_model}/mcss_ff_test_angle={angle}.csv"
        En, Es, _, wff, wbf, Pff, Pbf, Pel, tot = np.genfromtxt(mcss_fn, unpack=True, delimiter=",", skip_header=1)

        # Compare results
        dsfs *= norm_factor
        ax0.plot(
            omega_array[::-1] * J_TO_eV,
            np.array(dsfs[::-1]),
            label=f"$q$={q}",
            c=c,
        )

        fname = f"validation/ff_dsf/4hannah_rs_{int(rs)}_theta_{int(theta)}_{q}.txt"
        dat_j = np.genfromtxt(fname=fname, skip_header=22)
        norm_Jan = 1 / (RYDBERG_TO_eV * eV_TO_J)
        norm_mcss = 1 / (eV_TO_J)
        print(wff)
        twinx = ax0.twinx()
        twinx.plot(En[::-1], wff[::-1] * norm_mcss, label="MCSS", c=c, ls="dotted")
        twinx.plot(dat_j[:, 0] * RYDBERG_TO_eV, dat_j[:, 4] * norm_Jan, c=c, ls="dashed", label=f"RPA: q={q}")

        print(
            f"Maxima:\n"
            f"Jan: {np.max(dat_j[:, 4] * norm_Jan)} 1/J[?] ---> MCSS: {np.max(wff) * norm_mcss} [1/J] ---> me: {np.max(dsfs)} [wrong]\n"
            f"Ratio:  {np.max(dat_j[:, 4] * norm_Jan) / np.max(dsfs)}\n"
            f"1/ratio: {np.max(dsfs) / np.max(dat_j[:, 4] * norm_Jan)}\n"
        )

    ax0.legend()
    ax0.set_xlabel(r"$\omega$ [eV]")
    ax0.set_ylabel(r"$S_{ff}$ [mystery]")
    twinx.set_ylabel(r"$S_{ff}$ [1/J]")
    twinx.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
    fig.savefig(f"ff_comparison_rs={rs}_theta={theta}.pdf", dpi=200)
    plt.close()


def test_lindhard():
    rs = 2
    theta = 1
    rho, Te = get_rho_T_from_rs_theta(rs=rs, theta=theta)
    ks = np.array((0.5, 1.0, 2.0, 4.0)) / BOHR_RADIUS
    rho *= g_per_cm3_TO_kg_per_m3
    Te *= eV_TO_K
    charge_state = 1.0
    atomic_mass = 1.0
    atomic_number = 1.0
    lfc = 0.0

    models = ModelOptions(polarisation_model="NUMERICAL")
    models2 = ModelOptions(polarisation_model="LINDHARD")

    omega_array = np.linspace(-100, 100, 500) * eV_TO_J
    state = PlasmaState(
        electron_temperature=Te,
        ion_temperature=Te,
        mass_density=rho,
        charge_state=charge_state,
        atomic_mass=atomic_mass,
        atomic_number=atomic_number,
    )

    fig, axes = plt.subplots(1, 1, figsize=(14, 8))
    colors = ["magenta", "crimson", "orange", "dodgerblue", "lightgreen", "lightgray", "yellow", "cyan"]

    for k, cs in zip(ks, colors):
        dsfs = np.zeros_like(omega_array)
        dsfs2 = np.zeros_like(omega_array)
        q = k * BOHR_RADIUS

        for i in range(0, len(omega_array)):
            w = omega_array[i]
            kernel = FreeFreeDSF(state=state)
            # kernel2 = FreeFreeDSF(state=state, models=models2)

            dsf = kernel.get_dsf(k=k, w=w, lfc=lfc, model="NUMERICAL_RPA")
            dsf2 = kernel.get_dsf(k=k, w=w, lfc=lfc, model="LINDHARD")

            dsfs[i] = dsf
            dsfs2[i] = dsf2

        idx = np.argwhere(np.isnan(dsfs))
        dsfs_new = np.delete(dsfs, idx)
        dsfs2_new = np.delete(dsfs2, idx)
        omega_new = np.delete(omega_array, idx)

        axes.plot(omega_new * J_TO_eV, dsfs_new / J_TO_eV, label=f"q={q} 1/aB", c=cs)
        axes.plot(omega_new * J_TO_eV, dsfs2_new / J_TO_eV, label=f"Lindhard: q={q}", c=cs, ls="-.")

    axes.set_xlabel(r"$\omega$ [eV]")
    axes.set_ylabel(r"DSF [1/eV]")
    axes.legend()

    plt.tight_layout()
    plt.show()
    fig.savefig("ff_dsf_test_lindhard.pdf", dpi=200)


def test_full_spectrum():
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    Z_mean = 0.51

    rs = 3
    theta = 1
    atomic_mass = 1.00784
    rho, T = get_rho_T_from_rs_theta(rs=rs, theta=theta, atomic_mass=atomic_mass)
    rho *= g_per_cm3_TO_kg_per_m3
    T *= eV_TO_K

    models = ModelOptions(polarisation_model="NUMERICAL_RPA", bf_model="SCHUMACHER")

    state_bf = PlasmaState(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_state=0.0,
        atomic_mass=atomic_mass,
        atomic_number=1,
    )

    beam_energy = 9.0e3
    angles = np.array([13, 30, 45, 60, 80, 100, 120, 140, 160])
    ks = calculate_q(angle=angles, energy=beam_energy) / BOHR_RADIUS
    omega_array = np.linspace(-100, 100, 500) * eV_TO_J

    binding_energies = (
        np.array(
            [
                -13.6,
            ]
        )
        * eV_TO_J
    )

    k = ks[0]

    bf_kernel = BoundFreeDSF(state=state_bf)
    bf_dsf = bf_kernel.get_dsf(ZA=1.0, Zb=state_bf.Zb, k=k, w=omega_array, Eb=binding_energies, model=models.bf_model)

    state_ff = PlasmaState(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_state=1.0,
        atomic_mass=atomic_mass,
        atomic_number=1,
    )
    ff_kernel = FreeFreeDSF(state=state_ff)
    ff_dsf = ff_kernel.get_dsf(k=k, w=omega_array, lfc=0.0, model=models.polarisation_model)

    tot_dsf = 0.5 * ff_dsf + 0.5 * bf_dsf

    WR = 1.2

    sif = stats.norm.pdf(omega_array, 0, 2 * eV_TO_J)
    sif /= np.max(sif)
    # plt.figure()
    # plt.plot(omega_array, sif)
    # plt.show()

    # bf_sif = np.convolve(sif, bf_dsf, mode="same")
    # spectrum = np.convolve(tot_dsf, sif, mode="same")
    inelastic = fftconvolve(tot_dsf, sif, mode="same")  # + WR * sif
    elastic = WR * sif * J_TO_eV
    spectrum = inelastic + elastic

    # print(bf_dsf)
    # plt.figure()
    fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    ax.plot(omega_array * J_TO_eV, 0.5 * bf_dsf / J_TO_eV, label="BF", c="crimson", ls="-.")
    ax.plot(omega_array * J_TO_eV, 0.5 * ff_dsf / J_TO_eV, label="FF", c="navy", ls="-.")
    ax.plot(omega_array * J_TO_eV, tot_dsf / J_TO_eV, label="TOT", c="darkgreen", ls="-.")
    # ax.plot(omega_array * J_TO_eV, sif / J_TO_eV, label="SIF", c="black", ls="-.")
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.set_ylabel(r"DSF [1/eV]")
    ax.legend()

    ax = axes[1]
    ax.plot(omega_array * J_TO_eV, sif / np.max(sif), label="SIF", ls="-.")
    ax.plot(omega_array * J_TO_eV, inelastic / np.max(spectrum), label="Inelastic", ls="-.")
    ax.plot(omega_array * J_TO_eV, elastic / np.max(spectrum), label="Elastic", ls="-.")
    ax.plot(omega_array * J_TO_eV, spectrum / np.max(spectrum), label="Spectrum", ls="-.")
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.set_ylabel(r"Signal [arb. units]")
    # ax.plot(omega_array, bf_sif / np.max(bf_sif), label="BF", ls="solid")
    ax.legend()
    fig.suptitle(f"Hydrogen at rs={rs}, theta={theta}, Z=0.5")
    plt.show()

    # return


def compare_hydrogen_against_pimc():
    import scipy.stats as stats

    rs = 3
    theta = 1
    atomic_mass = 1.00784
    Z_mean = 0.51
    ipd_best_fit = -3.43  # eV
    rho, T = get_rho_T_from_rs_theta(rs=rs, theta=theta, atomic_mass=atomic_mass)
    rho *= g_per_cm3_TO_kg_per_m3
    T *= eV_TO_K

    N = 14
    pimc_data_dir = f"/home/bellen85/code/dev/itcf_fitting/data/N{N}_rs{rs}_theta{theta:.0f}"
    # pimc_data_dir = "/home/bellen85/code/dev/itcf_fitting/data/N14_rs3_theta1"
    q_value, tau_array, itcf_array, itcf_errors, S_ei, S_ii, WR = load_itcf_from_file(
        N=N, q_index=10, data_path=pimc_data_dir
    )

    state_H0 = PlasmaState(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_state=0.0,
        atomic_mass=atomic_mass,
        atomic_number=1,
    )

    state_H1 = PlasmaState(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_state=1.0,
        atomic_mass=atomic_mass,
        atomic_number=1,
    )

    # beam_energy = 9.0e3  # * eV_TO_J
    # angles = np.array([13, 30, 45, 60, 80, 100, 120, 140, 160])
    # ks = calculate_q(angle=angles, energy=beam_energy) / BOHR_RADIUS
    omega_array = np.linspace(-200, 200, 1000) * eV_TO_J

    # WR = 1.2

    models = ModelOptions(
        polarisation_model="NUMERICAL", bf_model="SCHUMACHER", lfc_model="DORNHEIM_ESA", ipd_model="NONE"
    )
    x1, x2 = get_fractions_from_Z(Z=Z_mean)
    xs = np.array([x1, x2])

    print(
        f"Running fractions: {x1} for charge {state_H0.charge_state}\n"
        f"and {x2} for charge {state_H1.charge_state}\n"
    )
    xdave = xDave(
        models=models,
        states=np.array([state_H0, state_H1]),
        fractions=xs,
        binding_energies=np.array([-13.6]) * eV_TO_J,
        rayleigh_weight=WR,
        sif=np.zeros_like(omega_array),
        ipd=ipd_best_fit,
    )

    q = q_value
    k = q / BOHR_RADIUS

    bf_tot, ff_tot, dsf, Wr = xdave.run(k=k, w=omega_array)
    tau_array, F_tot_inel, F_wff, F_wbf = xdave.get_itcf(
        tau=tau_array, w=omega_array * J_TO_eV, ff=ff_tot / J_TO_eV, bf=bf_tot / J_TO_eV
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 10))

    # plt.figure()
    ax = axes[0]
    ax.plot(omega_array * J_TO_eV, bf_tot / J_TO_eV, label="BF")
    ax.plot(omega_array * J_TO_eV, ff_tot / J_TO_eV, label="FF")
    ax.plot(omega_array * J_TO_eV, dsf / J_TO_eV, label="Tot")
    ax.legend()
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.set_ylabel(r"DSF [1/eV]")
    # plt.show()

    sif = stats.norm.pdf(omega_array, 0, 2 * eV_TO_J)
    sif /= np.max(sif)
    WR *= J_TO_eV

    inelastic, elastic, spectrum = xdave.convolve_with_sif(sif=sif, bf=bf_tot, ff=ff_tot, WR=WR)

    ax = axes[1]
    ax.plot(omega_array * J_TO_eV, inelastic / np.max(spectrum), label="inel", ls="-.")
    ax.plot(omega_array * J_TO_eV, elastic / np.max(spectrum), label="el", ls="-.")
    ax.plot(omega_array * J_TO_eV, spectrum / np.max(spectrum), label="tot", ls="-.")
    ax.legend()
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.set_ylabel(r"Intensity []")

    ax = axes[2]
    # ax.plot(tau_array, itcf_array, label="PIMC tot", ls="solid", c="black")
    ax.plot(tau_array, itcf_array - Wr, label="PIMC inel", ls="dashed", c="crimson")
    ax.plot(tau_array, F_tot_inel, label="xDave inel", ls="dashed", c="magenta")
    ax.plot(tau_array, F_wff, label="xDave ff", ls="dashed", c="navy")
    ax.plot(tau_array, F_wbf, label="xDave bf", ls="dashed", c="orange")
    ax.axhline(Wr, ls="solid", c="gray", label="WR")
    ax.legend()
    ax.set_xlabel(r"$\tau$ [1/eV]")
    ax.set_ylabel(r"ITCF []")

    plt.suptitle(f"Hydrogen at q={q:.2f} rs={rs}, theta={theta}, Z={Z_mean}, ipd={ipd_best_fit}")
    plt.tight_layout()
    plt.show()

    fig.savefig(f"hydrogen_test_rs={rs}_theta={theta}_Z={Z_mean}_q={q:.2f}.pdf", dpi=200)


def compare_hydrogen_against_pimc_and_mcss():
    import os
    import scipy.stats as stats

    q_index = 0

    rs = 3
    theta = 1
    atomic_mass = 1.00784  # * 1.6605e-27
    Z_mean = 0.51
    ipd_best_fit = -3.43  # eV
    rho, T = get_rho_T_from_rs_theta(rs=rs, theta=theta, atomic_mass=atomic_mass)
    # rho *= g_per_cm3_TO_kg_per_m3
    # T *= eV_TO_K

    N = 14
    pimc_data_dir = f"/home/bellen85/code/dev/itcf_fitting/data/N{N}_rs{rs}_theta{theta:.0f}"
    # pimc_data_dir = "/home/bellen85/code/dev/itcf_fitting/data/N14_rs3_theta1"
    q_value, tau_array, itcf_array, itcf_errors, S_ei, S_ii, WR_pimc = load_itcf_from_file(
        N=N, q_index=q_index, data_path=pimc_data_dir
    )

    mcss_data_dir = f"/home/bellen85/code/dev/itcf_fitting/results/processing/"
    mcss_fn = os.path.join(mcss_data_dir, f"mcss_production_run_N{N}_rs{rs}_theta{theta:.0f}_index={q_index}.csv")
    # rayleigh_weight = get_mcss_wr_from_status_file(mcss_fn + "_status.txt")
    mcss_En, mcss_wff, mcss_wbf, mcss_ff, mcss_bf, mcss_el = load_mcss_result(mcss_fn)

    binding_energies = np.array([-13.6 * eV_TO_J])

    omega_array = np.linspace(-70, 200, 9000)  # * eV_TO_J

    # WR = 1.2

    models = ModelOptions(
        polarisation_model="NUMERICAL", bf_model="SCHUMACHER", lfc_model="DORNHEIM_ESA", ipd_model="NONE"
    )
    Z1, Z2, x1, x2 = get_fractions_from_Z(Z=Z_mean)
    xs = np.array([x1, x2])

    elements = np.array(["H", "H"])
    partial_densities = np.array([0.485, 0.515])
    charge_states = np.array([0.0, 1.0])
    user_defined_inputs = {"ipd": ipd_best_fit}

    sif = stats.norm.pdf(omega_array, 0, 2 * eV_TO_J)
    sif /= np.max(sif)
    WR = WR_pimc * J_TO_eV

    xdave = xDave(
        models=models,
        elements=elements,
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_states=charge_states,
        partial_densities=partial_densities,
        user_defined_inputs=user_defined_inputs,
    )

    q = q_value
    k = q  # / BOHR_RADIUS

    bf_tot, ff_tot, dsf, Wr, iff, ibf = xdave.run(k=k, w=omega_array)
    _, F_tot_inel, F_wff, F_wbf = xdave.get_itcf(tau=tau_array, w=omega_array, ff=ff_tot, bf=bf_tot)

    _, F_tot_inel_mcss, F_wff_mcss, F_wbf_mcss = xdave.get_itcf(tau=tau_array, w=mcss_En, ff=mcss_wff, bf=mcss_wbf)

    fig, axes = plt.subplots(1, 2, figsize=(14, 10))

    # plt.figure()
    ax = axes[0]
    ax.plot(omega_array, bf_tot, label="BF", c="navy")
    ax.plot(omega_array, ff_tot, label="FF", c="crimson")
    ax.plot(omega_array, dsf, label="Tot", c="darkgreen")
    ax.plot(mcss_En, mcss_wbf, label="MCSS: BF", c="navy", ls="-.")
    ax.plot(mcss_En, mcss_wff, label="MCSS: FF", c="crimson", ls="-.")
    ax.plot(mcss_En, mcss_wbf + mcss_wff, label="MCSS", c="darkgreen", ls="-.")
    ax.set_xlim(-100, 200)
    ax.legend()
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.set_ylabel(r"DSF [1/eV]")
    # plt.show()

    inelastic, elastic, spectrum = xdave.convolve_with_sif(
        sif=sif, bf=bf_tot, ff=ff_tot, WR=WR, type="GAUSSIAN", fwhm=10, omega=omega_array
    )
    mcss_tot = mcss_ff + mcss_bf + mcss_el

    # ax = axes[1]
    # ax.plot(omega_array * J_TO_eV, inelastic / np.max(spectrum), label="inel", ls="-.")
    # ax.plot(omega_array * J_TO_eV, elastic / np.max(spectrum), label="el", ls="-.")
    # ax.plot(omega_array * J_TO_eV, spectrum / np.max(spectrum), label="tot", ls="-.")
    # # ax.plot(mcss_En, mcss_tot / np.max(mcss_el), label="MCSS tot", c="black", ls="-.")
    # ax.legend()
    # ax.set_xlabel(r"$\omega$ [eV]")
    # ax.set_ylabel(r"Intensity []")

    ax = axes[1]
    ax.plot(tau_array, itcf_array, label="PIMC tot", ls=":", c="black")
    ax.plot(tau_array, itcf_array - WR_pimc, label="PIMC inel", ls=":", c="crimson")
    ax.plot(tau_array, F_tot_inel, label="xDave inel", ls="solid", c="magenta")
    ax.plot(tau_array, F_tot_inel_mcss, label="MCSS inel", ls="dashed", c="magenta")
    ax.plot(tau_array, F_wff, label="xDave ff", ls="solid", c="navy")
    ax.plot(tau_array, F_wff_mcss, label="MCSS ff", ls="dashed", c="navy")
    ax.plot(tau_array, F_wbf, label="xDave bf", ls="solid", c="orange")
    ax.plot(tau_array, F_wbf_mcss, label="MCSS bf", ls="dashed", c="orange")
    ax.axhline(WR_pimc, ls=":", c="gray", label="WR")
    ax.legend()
    ax.set_xlabel(r"$\tau$ [1/eV]")
    ax.set_ylabel(r"ITCF []")

    plt.suptitle(f"Hydrogen at q={q:.2f} rs={rs}, theta={theta}, Z={Z_mean}, ipd={ipd_best_fit}")
    plt.tight_layout()
    plt.show()

    fig.savefig(f"hydrogen_test_rs={rs}_theta={theta}_Z={Z_mean}_q={q:.2f}.pdf", dpi=200)


if __name__ == "__main__":
    # test_chemical_potential()
    # test_ff_rpa()
    # test_lindhard()
    compare_hydrogen_against_pimc_and_mcss()

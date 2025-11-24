# import sys

# sys.path.insert(1, "./xdave")
# sys.path.insert(1, "./mcss_tests")


from xdave.plasma_state import PlasmaState, get_rho_T_from_rs_theta, get_fractions_from_Z
from xdave.unit_conversions import *
from xdave.utils import load_itcf_from_file, load_mcss_result
from xdave import xDave, ModelOptions

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

import os
import scipy.stats as stats


def compare_hydrogen_against_pimc():

    rs = 3
    theta = 1
    atomic_mass = 1.00784
    Z_mean = 0.51
    ipd_best_fit = -3.43  # eV
    rho, T = get_rho_T_from_rs_theta(rs=rs, theta=theta, atomic_mass=atomic_mass)
    # rho *= g_per_cm3_TO_kg_per_m3
    # T *= eV_TO_K

    N = 14
    pimc_data_dir = f"/home/bellen85/code/dev/itcf_fitting/data/N{N}_rs{rs}_theta{theta:.0f}"
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

    omega_array = np.linspace(-200, 200, 1000)

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
        binding_energies=np.array([-13.6]),
        rayleigh_weight=WR,
        sif=np.zeros_like(omega_array),
        ipd=ipd_best_fit,
    )

    q = q_value
    k = q  # / BOHR_RADIUS

    bf_tot, ff_tot, dsf, Wr = xdave.run(k=k, w=omega_array)
    tau_array, F_tot_inel, F_wff, F_wbf = xdave.get_itcf(tau=tau_array, w=omega_array, ff=ff_tot, bf=bf_tot)

    fig, axes = plt.subplots(1, 3, figsize=(14, 10))

    ax = axes[0]
    ax.plot(omega_array, bf_tot, label="BF")
    ax.plot(omega_array, ff_tot, label="FF")
    ax.plot(omega_array, dsf, label="Tot")
    ax.legend()
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.set_ylabel(r"DSF [1/eV]")

    # sif = stats.norm.pdf(omega_array, 0, 2 * eV_TO_J)
    # sif /= np.max(sif)
    # WR *= J_TO_eV

    spectrum_energy, inelastic, elastic, spectrum = xdave.convolve_with_sif(
        omega=omega_array, dsf=(bf_tot + ff_tot), Wr=Wr, beam_energy=9.0e3, type="GAUSSIAN", fwhm=10
    )

    ax = axes[1]
    ax.plot(spectrum_energy, inelastic / np.max(spectrum), label="inel", ls="-.")
    ax.plot(spectrum_energy, elastic / np.max(spectrum), label="el", ls="-.")
    ax.plot(spectrum_energy, spectrum / np.max(spectrum), label="tot", ls="-.")
    ax.legend()
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.set_ylabel(r"Intensity []")

    ax = axes[2]
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

    q_index = 0

    rs = 3
    theta = 1
    atomic_mass = 1.00784
    Z_mean = 0.51
    ipd_best_fit = -3.43  # eV
    rho, T = get_rho_T_from_rs_theta(rs=rs, theta=theta, atomic_mass=atomic_mass)
    N = 14
    pimc_data_dir = f"/home/bellen85/code/dev/itcf_fitting/data/N{N}_rs{rs}_theta{theta:.0f}"
    q_value, tau_array, itcf_array, itcf_errors, S_ei, S_ii, WR_pimc = load_itcf_from_file(
        N=N, q_index=q_index, data_path=pimc_data_dir
    )

    mcss_data_dir = f"/home/bellen85/code/dev/itcf_fitting/results/processing/"
    mcss_fn = os.path.join(mcss_data_dir, f"mcss_production_run_N{N}_rs{rs}_theta{theta:.0f}_index={q_index}.csv")
    mcss_En, mcss_wff, mcss_wbf, mcss_ff, mcss_bf, mcss_el = load_mcss_result(mcss_fn)

    binding_energies = np.array([-13.6 * eV_TO_J])

    omega_array = np.linspace(-70, 200, 9000)

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
    k = q

    bf_tot, ff_tot, dsf, Wr, iff, ibf = xdave.run(k=k, w=omega_array)
    _, F_tot_inel, F_wff, F_wbf = xdave.get_itcf(tau=tau_array, w=omega_array, ff=ff_tot, bf=bf_tot)

    _, F_tot_inel_mcss, F_wff_mcss, F_wbf_mcss = xdave.get_itcf(tau=tau_array, w=mcss_En, ff=mcss_wff, bf=mcss_wbf)

    fig, axes = plt.subplots(1, 2, figsize=(14, 10))

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

    inelastic, elastic, spectrum = xdave.convolve_with_sif(
        sif=sif, bf=bf_tot, ff=ff_tot, WR=WR, type="GAUSSIAN", fwhm=10, omega=omega_array
    )
    mcss_tot = mcss_ff + mcss_bf + mcss_el

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
    compare_hydrogen_against_pimc_and_mcss()

import sys

# sys.path.insert(1, "./xdave")
sys.path.insert(1, "./mcss_tests")


from xdave.plasma_state import PlasmaState, get_rho_T_from_rs_theta, get_fractions_from_Z, get_rho_T_from_rs_theta_SI
from xdave.models import ModelOptions
from xdave.unit_conversions import *
from xdave.constants import BOHR_RADIUS, PLANCK_CONSTANT, ELECTRON_MASS
from xdave.freefree_dsf import FreeFreeDSF
from xdave.boundfree_dsf import BoundFreeDSF
from xdave.utils import calculate_angle, calculate_q, load_itcf_from_file, load_mcss_result, get_mcss_wr_from_status_file
from xdave.xdave import xDave
import scipy.stats as stats

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


def test_full_spectrum():
    return


def test_setup():
    elements = np.array(["H", "H", "C", "C"])
    rho = 1  # * g_per_cm3_TO_kg_per_m3
    T = 70  # * eV_TO_K

    partial_densities = np.array([0.15, 0.15, 0.34, 0.36])
    charge_states = np.array([0.0, 1.0, 3, 4])
    user_defined_inputs = None

    models = ModelOptions(polarisation_model="NUMERICAL", bf_model="SCHUMACHER", lfc_model="NONE", ipd_model="NONE")

    omega_array = np.linspace(-1000, 1500, 1000) * eV_TO_J

    k_SI = 8 / ang_TO_m
    q = k_SI * BOHR_RADIUS
    k = 8 * per_A_TO_per_aB
    beam_energy = 9.0e3
    angle = calculate_angle(q=q, energy=beam_energy)
    print(f"Running at q={q}, E={beam_energy} -> angle={angle}")

    # Load values from MCSS output files
    mcss_fn = f"mcss_tests/mixed_species_tests/mcss_mixed_species_test_ch_angle={angle:.2f}"
    rayleigh_weight = get_mcss_wr_from_status_file(mcss_fn + "_status.txt")
    mcss_En, mcss_wff, mcss_wbf, mcss_ff, mcss_bf, mcss_el = load_mcss_result(mcss_fn + ".csv")
    mcss_ipd = -3.3087805e001  # eV
    sif = np.zeros_like(omega_array)

    kernel = xDave(
        models=models,
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        elements=elements,
        partial_densities=partial_densities,
        charge_states=charge_states,
        user_defined_inputs=user_defined_inputs,
    )

    bf_tot, ff_tot, dsf, WR, ff_i, bf_i = kernel.run(k=k, w=omega_array)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    mcss_norm = 0.3 * 1 + (1 - 0.3) * 6
    ax = axes[0, 0]
    ax.set_title("Total DSF")
    ax.plot(omega_array, bf_tot, label="BF")
    ax.plot(omega_array, ff_tot, label="FF")
    ax.plot(omega_array, dsf, label="Tot")
    ax.plot(mcss_En, (mcss_wbf + mcss_wff) / mcss_norm, lw=2, ls="dashed", c="black", label="MCSS")
    # ax.plot(mcss_En, (mcss_wbf + mcss_wff), lw=2, c="black", ls="dashed", label="MCSS")
    ax.legend()

    ax = axes[1, 0]
    ax.set_title("FF contributions")
    ax.plot(omega_array, ff_i[0], label="H0: FF")
    ax.plot(omega_array, ff_i[1], label="H1: FF")
    ax.plot(omega_array, ff_i[2], label="C3: FF")
    ax.plot(omega_array, ff_i[3], label="C4: FF")
    ax.plot(omega_array, ff_tot, label="Tot FF")
    ax.plot(mcss_En, mcss_wff / mcss_norm, lw=2, c="black", ls="dashed", label="MCSS")
    # ax.plot(mcss_En, mcss_wff, lw=2, c="black", ls="solid", label="MCSS")
    ax.legend()

    ax = axes[1, 1]
    ax.set_title("BF contributions")
    ax.plot(omega_array, bf_i[0], label="H0: BF")
    ax.plot(omega_array, bf_i[1], label="H1: BF")
    ax.plot(omega_array, bf_i[2], label="C3: BF")
    ax.plot(omega_array, bf_i[3], label="C4: BF")
    ax.plot(omega_array, bf_tot, label="Tot BF")
    ax.plot(mcss_En, mcss_wbf / mcss_norm, lw=2, c="black", ls="dashed", label="MCSS")
    # ax.plot(mcss_En, mcss_wbf, lw=2, c="black", ls="solid", label="MCSS")
    ax.legend()

    sif = stats.norm.pdf(omega_array, 0, 2 * eV_TO_J)
    sif /= np.max(sif)
    WR *= J_TO_eV

    spectrum_energy, inelastic, elastic, spectrum = kernel.convolve_with_sif(
        sif=sif, bf=bf_tot, ff=ff_tot, WR=WR, type="USER_INPUT"
    )

    ax = axes[0, 1]
    ax.set_title("Spectrum")
    ax.plot(spectrum_energy, inelastic / np.max(spectrum), label="inel", ls="-.")
    ax.plot(spectrum_energy, elastic / np.max(spectrum), label="el", ls="-.")
    ax.plot(spectrum_energy, spectrum / np.max(spectrum), label="tot", ls="-.")
    ax.legend()
    ax.set_xlim(-800, 900)

    plt.tight_layout()
    plt.show()
    fig.savefig(f"ch_test_T={T}_rho={rho}_Z={kernel.overlord_state.charge_state}.pdf", dpi=200)


def test_be():
    rs = 3
    theta = 1
    Z_mean = 3.73
    rho = 22.0  # g/cc
    T = 150  # eV

    beam_energy = 9.0e3  # eV
    angles = np.array([13, 30, 45, 60, 80, 100, 120, 140, 160])
    ks = calculate_q(angle=angles, energy=beam_energy) / BOHR_RADIUS  # 1/a_B
    omega_array = np.linspace(-800, 1400, 1000)  # eV

    # WR = 0.1

    models = ModelOptions(
        polarisation_model="NUMERICAL", bf_model="SCHUMACHER", lfc_model="DORNHEIM_ESA", ipd_model="STEWART_PYATT"
    )
    Z1, Z2, x1, x2 = get_fractions_from_Z(Z=Z_mean)
    xs = np.array([x1, x2])

    elements = np.array(["Be", "Be"])
    charge_states = np.array([Z1, Z2])

    user_defined_inputs = {"ipd": 0.0, "lfc": 1.0, "ion_core_radii": np.array([2.0, 2.0])}

    xdave = xDave(
        models=models,
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        elements=elements,
        partial_densities=xs,
        charge_states=charge_states,
        user_defined_inputs=user_defined_inputs,
    )

    k = 8 / A_TO_aB
    q = k  # 1/ aB

    bf_tot, ff_tot, dsf, WR, _, _ = xdave.run(k=k, w=omega_array)
    ff_tot[np.isnan(ff_tot)] = 0.0

    print(f"Calculate Rayleigh weight: {WR}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 10))

    ax = axes[0]
    ax.plot(omega_array, bf_tot, label="BF")
    ax.plot(omega_array, ff_tot, label="FF")
    ax.plot(omega_array, dsf, label="Tot")
    ax.legend()

    sif = stats.norm.pdf(omega_array, 0, 2)
    sif /= np.max(sif)

    spectrum_energy, spectrum = xdave.convolve_with_sif(
        # source=sif,
        # source_ene=omega_array,
        omega=omega_array,
        dsf=(bf_tot + ff_tot),
        Wr=WR,
        beam_energy=beam_energy,
        type="GAUSSIAN",
        fwhm=10,
    )

    ax = axes[1]
    # ax.plot(omega_array, inelastic / np.max(spectrum), label="inel", ls="-.")
    # ax.plot(omega_array, elastic / np.max(spectrum), label="el", ls="-.")
    ax.plot(spectrum_energy, spectrum / np.max(spectrum), label="tot", ls="-.")
    ax.legend()
    # ax.set_xlim(-800, 750)

    plt.show()
    fig.savefig(f"beryllium_test_rs={rs}_theta={theta}_Z={Z_mean}_q={q:.2f}.pdf", dpi=200)


if __name__ == "__main__":

    #     ##TODO(Hannah):
    #     ## check that the mass density and number of electrons is being handled correctly across all states
    #     ## compare against MCSS and PIMC for this set of conditions
    #     ## Add IPD model: DONE
    #     ## Clean up bf call (arguments are a bit messy): DONE
    #     ## Start calculating things like kF, EF, omega_p, etc. for the plasma state upon initialisation to avoid extra computation
    #     ## Start timing and looking at how much this scales with number of points
    #     ## I should move away from defining states by their mass density (problematic when you have mixed species) and just look at electron number density... probably a lot easier to split up: STILL THINKING ABOUT THIS
    # test_setup()
    test_be()

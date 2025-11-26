from xdave import xDave
from xdave.models import ModelOptions
from xdave.unit_conversions import *
from xdave.constants import BOHR_RADIUS
from xdave.utils import (
    calculate_angle,
    load_mcss_result,
    get_mcss_wr_from_status_file,
)
from xdave.xdave import xDave
import scipy.stats as stats

import numpy as np
import matplotlib.pyplot as plt


def test_setup():
    elements = np.array(["H", "H", "C", "C"])
    rho = 1  # g/cc
    T = 70  # eV

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
    ax.legend()

    ax = axes[1, 0]
    ax.set_title("FF contributions")
    ax.plot(omega_array, ff_i[0], label="H0: FF")
    ax.plot(omega_array, ff_i[1], label="H1: FF")
    ax.plot(omega_array, ff_i[2], label="C3: FF")
    ax.plot(omega_array, ff_i[3], label="C4: FF")
    ax.plot(omega_array, ff_tot, label="Tot FF")
    ax.plot(mcss_En, mcss_wff / mcss_norm, lw=2, c="black", ls="dashed", label="MCSS")
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


def test_mc_setup():
    # Random black Kapton example
    T = 45  # eV
    rho = 2 * 1.42  # g/cc

    partial_densities = np.array([0.026362, 0.691133, 0.073270, 0.209235])
    charge_states = np.array([1, 4, 4, 4])
    elements = np.array(["H", "C", "N", "O"])

    models = ModelOptions()

    user_defined_inputs = {
        "ipd": -10,
        "lfc": 0.1,
        "ion_core_radii": [1, 1, 1, 1],
        "csd_parameters": [1, 1, 1, 1],
        "csd_core_charges": [1, 6, 7, 8],
        "sec_core_power": 0.1,
        "srr_sigma_parameter": 1,
    }

    kernel = xDave(
        mass_density=rho,
        electron_temperature=T,
        ion_temperature=T,
        elements=elements,
        partial_densities=partial_densities,
        charge_states=charge_states,
        models=models,
        user_defined_inputs=user_defined_inputs,
    )
    models.print_default_options()
    print(f"Initialized kernel with states:\n")
    print(f"Overlord state: {vars(kernel.overlord_state)}")

    for i in range(0, len(kernel.states)):
        print(f"State {i}: {vars(kernel.states[i])}")


if __name__ == "__main__":
    # test_setup()
    # test_be()
    test_mc_setup()

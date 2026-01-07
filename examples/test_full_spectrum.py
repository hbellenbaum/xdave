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


def test_sif():
    T = 80.0  # eV
    rho = 3.5  # g/cc
    ZC = 4.0  # also ZC=2.5, 3.5, 4.0, 4.5
    ZH = 1.0
    xH = 0.2
    angle = 120  # degrees
    beam_energy = 8.5e3  # eV
    q = calculate_q(angle=angle, energy=beam_energy)
    print(f"Running at q={q:.3f}")

    elements = np.array(["H", "C", "C"])

    Zmin, Zmax, xmin, xmax = get_fractions_from_Z_partial(ZC, x0=xH)
    partial_densities = np.array([xH, xmin, xmax])
    charge_states = np.array([ZH, Zmin, Zmax])
    user_defined_inputs = dict()

    models = ModelOptions(
        polarisation_model="DANDREA_FIT",
        bf_model="SCHUMACHER",
        lfc_model="NONE",
        ipd_model="NONE",
        screening_model="DEBYE_HUCKEL",
    )

    k = q  # 1/aB

    omega_array = np.arange(-4000, 4000, 1.0)  # eV

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

    if False:
        plt.figure()
        plt.plot(omega_array, bf_tot, c="magenta", label="BF")
        plt.plot(omega_array, ff_tot, c="limegreen", label="FF")
        plt.plot(omega_array, dsf, c="orange", label="DSF")
        plt.legend()
        plt.show()

    energy1, inelastic1, elastic1, spectrum1 = kernel.convolve_with_sif(
        omega=omega_array, bf=bf_tot, ff=ff_tot, dsf=dsf, Wr=WR, beam_energy=beam_energy, type="GAUSSIAN", fwhm=26
    )

    def asym_gaussian(x, A, x0, sigma_left, sigma_right, baseline=0.0):
        """Credit: W. Martin :)"""
        x = np.asarray(x)
        xprime = x - x0
        sigma = np.where(x < x0, sigma_left, sigma_right)
        return xprime, baseline + A * np.exp(-0.5 * (xprime / sigma) ** 2)

    x = omega_array + beam_energy
    A = 0.025
    x0 = beam_energy
    sigma_left = 13
    sigma_right = 13
    ene_user_defined, sif_user_defined = asym_gaussian(x=x, A=A, x0=x0, sigma_left=sigma_left, sigma_right=sigma_right)
    energy2, inelastic2, elastic2, spectrum2 = kernel.convolve_with_sif(
        omega=omega_array,
        bf=bf_tot,
        ff=ff_tot,
        dsf=dsf,
        Wr=WR,
        beam_energy=beam_energy,
        type="USER_DEFINED",
        source_energy=ene_user_defined + beam_energy,
        source=sif_user_defined,
    )

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(energy1, spectrum1, label="GAUSSIAN", ls="-.", c="navy")
    axes[0].plot(energy2, spectrum2, label="USER_DEFINED", ls="--", c="crimson")
    axes[0].legend()
    axes[0].set_xlabel("Energy [eV]")
    axes[0].set_ylabel("Spectrum [ ]")

    axes[1].plot(energy1, elastic1, label="GAUSSIAN", ls="-.", c="navy")
    axes[1].plot(
        ene_user_defined + beam_energy,
        sif_user_defined,
        label="USER_DEFINED",
        ls="--",
        c="crimson",
    )
    axes[1].set_xlabel("Energy [eV]")
    axes[1].set_ylabel("SIF [ ]")
    axes[1].set_xlim(beam_energy - 100, beam_energy + 100)
    axes[1].legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_sif()

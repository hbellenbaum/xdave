## simple HDC example for comparison against MCSS
# import sys

# sys.path.insert(1, "./xdave")

from xdave.plasma_state import get_fractions_from_Z
from xdave.xdave import *

import matplotlib.pyplot as plt
from scipy.special import voigt_profile
import time


def hdc_example():
    T = 70.7  # eV
    rho = 10.0  # g/cc
    Z = 4.5  # 4.86

    Zmin, Zmax, xmin, xmax = get_fractions_from_Z(Z=Z)

    models = ModelOptions(
        ei_potential="YUKAWA",
        ii_potential="YUKAWA",
        ee_potential="COULOMB",
        polarisation_model="NUMERICAL",
        sf_model="HNC",
        lfc_model="DORNHEIM_ESA",
        ipd_model="STEWART_PYATT",
        bf_model="SCHUMACHER",
        screening_model="FINITE_WAVELENGTH",
    )

    elements = np.array(["C", "C"])
    partial_densities = np.array([xmin, xmax])
    charge_states = np.array([Zmin, Zmax])

    kernel = xDave(
        mass_density=rho,
        electron_temperature=T,
        ion_temperature=T,
        elements=elements,
        charge_states=charge_states,
        partial_densities=partial_densities,
        models=models,
        enforce_fsum=False,
        verbose=True,
    )

    w = np.linspace(-1000, 1400, 2000)
    angle = 160
    beam_energy = 9.0e3

    bf_tot, ff_tot, dsf, rayleigh_weight, ff_i, bf_i = kernel.run(
        w=w, angle=angle, beam_energy=beam_energy, mode="DYNAMIC"
    )

    Zn_He_alpha = np.array([8.893, 8.932, 8.949, 8.970, 8.998]) * 1.0e3
    Zn_He_amp = np.array([0.0, 0.182, 0.82, 0.21, 1.09]) * 1.0e3

    sigmaG, sigmaL = 8e-3 * 1.0e3, 13.5e-3 * 1.0e3  # in eV

    dE = 0.1e-3 * 1.0e3
    source_energy = np.arange(7.5e3, 10.5e3 + dE, dE)
    # source_energy = w  # + beam_energy
    source = np.zeros_like(source_energy)

    for ii in range(len(Zn_He_alpha)):
        source += Zn_He_amp[ii] * voigt_profile(source_energy - Zn_He_alpha[ii], sigmaG, sigmaL)

    plt.figure()
    plt.plot(source_energy, source)
    plt.show()

    spec_energy, inelastic, elastic, spectrum = kernel.convolve_with_sif(
        omega=w,
        bf=bf_tot,
        ff=ff_tot,
        dsf=(bf_tot + ff_tot),
        Wr=rayleigh_weight,
        beam_energy=beam_energy,
        type="USER_DEFINED",
        source_energy=source_energy,
        source=source,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 10))

    ax = axes[0]
    ax.plot(w, dsf, label="dsf", c="navy")
    ax.plot(w, bf_tot, label="bf_tot", c="crimson")
    ax.plot(w, ff_tot, label="ff_tot", c="green")
    ax.legend()

    ax = axes[1]
    ax.plot(spec_energy, spectrum, label="xDave", c="magenta")
    ax.plot(spec_energy, inelastic, label="Inelastic", c="lightgreen")
    ax.plot(spec_energy, elastic, label="Elastic", c="dodgerblue")
    # ax.plot(source_energy, source / np.max(source), label="Source", c="crimson")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    hdc_example()

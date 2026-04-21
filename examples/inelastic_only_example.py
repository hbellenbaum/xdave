from xdave.plasma_state import get_fractions_from_Z_partial
from xdave.utils import calculate_q
from xdave.constants import BOHR_RADIUS
from xdave import xDave, ModelOptions

import numpy as np
import matplotlib.pyplot as plt
import time


def ch_example():
    start_time = time.time()
    T = 100  # eV
    rho = 2 * 1.845  # two times solid density [g/cc]
    Z_C = 4.5

    xH = 0.5
    ZH = 1.0

    Zmin, Zmax, xmin, xmax = get_fractions_from_Z_partial(Z=Z_C, x0=xH)

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

    kernel = xDave(
        mass_density=rho,
        electron_temperature=T,
        ion_temperature=T,
        elements=np.array(["H", "C", "C"]),
        charge_states=np.array([ZH, Zmin, Zmax]),
        partial_densities=np.array([xH, xmin, xmax]),
        models=models,
        enforce_fsum=False,
        user_defined_inputs=None,
        verbose=True,
        hnc_max_iterations=10000,
        hnc_mix_fraction=0.99,
        hnc_delta=1.0e-7,
    )

    w = np.linspace(-2000, 2000, 10000)
    k = calculate_q(angle=60, energy=8.0e3)  # / BOHR_RADIUS
    # bf_tot, ff_tot, dsf, rayleigh_weight, ff_i, bf_i = kernel.run(w=w, angle=130, beam_energy=9.0e3, mode="DYNAMIC")
    bf_tot, ff_tot, dsf, ff_i, bf_i = kernel.run_inelastic(w=w, k=k)

    # this will convolve the dsf with a Gaussian sif of 10 eV fwhm
    # if you want to use your own, you can add it as an input to the sif input option
    # note that for now this will have to be centered around 0
    spec_energy, inelastic, elastic, spectrum = kernel.convolve_with_sif(
        omega=w,
        bf=bf_tot,
        ff=ff_tot,
        dsf=(bf_tot + ff_tot),
        Wr=0.0,
        beam_energy=9.0e3,
        type="GAUSSIAN",
        fwhm=10,
    )

    end_time = time.time()
    print(f"Code took {end_time - start_time} s to run.")
    # plot results
    fig, axes = plt.subplots(1, 3, figsize=(16, 16))

    ax = axes[0]
    ax.set_title("Total DSF")
    ax.plot(w, dsf, label="Inel", ls="-.", c="magenta")
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.legend()

    ax = axes[1]
    ax.set_title("FF DSF")
    ax.plot(w, ff_tot, label="FF", ls="--", c="orange")
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.legend()

    ax = axes[2]
    ax.set_title("BF DSF")
    ax.plot(w, bf_tot, label="BF", ls="solid", c="dodgerblue")
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.legend()

    plt.figure()
    # plt.plot(w, inelastic, label="inel")
    # plt.plot(w, elastic, label="inel")
    plt.plot(spec_energy, spectrum, label="inel")
    plt.legend()
    plt.xlabel(r"$\omega$ [eV]")
    plt.xlabel("Intensity [a.u.]")
    plt.show()


if __name__ == "__main__":
    ch_example()

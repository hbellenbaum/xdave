## simple Be example for comparison against MCSS
## also CH example
# import sys

# sys.path.insert(1, "./xdave")
from xdave.xdave import xDave
from xdave.models import ModelOptions

from xdave.plasma_state import get_fractions_from_Z
from xdave import *

import matplotlib.pyplot as plt
import time
import numpy as np
import os


def be_example():
    T = 20  # eV
    rho = 2 * 1.845  # two times solid density [g/cc]
    Z_Be = 3.5

    Zmin, Zmax, xmin, xmax = get_fractions_from_Z(Z=Z_Be)

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
        elements=np.array(["Be", "Be"]),
        charge_states=np.array([Zmin, Zmax]),
        partial_densities=np.array([xmin, xmax]),
        models=models,
        enforce_fsum=False,
        user_defined_inputs=None,
    )

    w = np.linspace(-1000, 1000, 10000)
    bf_tot, ff_tot, dsf, rayleigh_weight, ff_i, bf_i = kernel.run(w=w, angle=75, beam_energy=8.0e3, mode="DYNAMIC")

    # this will convolve the dsf with a Gaussian sif of 10 eV fwhm
    # if you want to use your own, you can add it as an input to the sif input option
    # note that for now this will have to be centered around 0
    spec_energy, inelastic, elastic, spectrum = kernel.convolve_with_sif(
        omega=w,
        bf=bf_tot,
        ff=ff_tot,
        dsf=(bf_tot + ff_tot),
        Wr=rayleigh_weight,
        beam_energy=9.0e3,
        type="GAUSSIAN",
        fwhm=10,
    )

    tau_array, F_tot_inel, F_wff, F_wbf = kernel.get_itcf(w=w, ff=ff_tot, bf=bf_tot)
    # plot results
    fig, axes = plt.subplots(2, 3, figsize=(16, 16))

    ax = axes[0, 0]
    ax.set_title("Total DSF")
    ax.plot(w, dsf, label="Inel", ls="-.", c="magenta")
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.legend()

    ax = axes[0, 1]
    ax.set_title("FF DSF")
    ax.plot(w, ff_tot, label="FF", ls="--", c="orange")
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.legend()

    ax = axes[0, 2]
    ax.set_title("BF DSF")
    ax.plot(w, bf_tot, label="BF", ls="solid", c="dodgerblue")
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.legend()

    tau_array, F_tot_inel, F_wff, F_wbf = kernel.get_itcf(w=w, ff=ff_tot, bf=bf_tot)

    ax = axes[1, 0]
    ax.set_title("ITCF")
    ax.plot(tau_array, F_tot_inel, label="xDave inel", ls="dashed", c="magenta")
    ax.set_xlabel(r"$\tau$ [1/eV]")
    ax.legend()

    ax = axes[1, 1]
    ax.set_title("FF ITCF")
    ax.plot(tau_array, F_wff, label="xDave ff", ls="dashed", c="dodgerblue")
    ax.set_xlabel(r"$\tau$ [1/eV]")
    ax.legend()

    ax = axes[1, 2]
    ax.set_title("BF ITCF")
    ax.plot(tau_array, F_wbf, label="xDave bf", ls="dashed", c="orange")
    ax.set_xlabel(r"$\tau$ [1/eV]")
    ax.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(spec_energy, spectrum, label="inel")
    plt.legend()
    plt.xlabel(r"$\omega$ [eV]")
    plt.xlabel("Intensity [a.u.]")
    plt.show()

    start_time = time.time()
    k = np.linspace(0.5, 10, 1000)
    k, Sab, _, Wr, qs, fs, lfc = kernel.run(w=w, k=k, beam_energy=8.0e3, mode="STATIC")
    data = dict({"k": k, "Sab": Sab, "Wr": Wr, "qs": qs, "fs": fs, "lfc": lfc})
    kernel.save_result(fname="be_test_static", dirname=os.path.dirname(__file__), data=data, run_mode="STATIC")
    end_time = time.time()
    print(f"Run took {end_time - start_time} s")

    fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    ax.plot(k, Wr, c="navy", label="WR")
    ax.set_xlabel(r"$k$ [$a_B^{-1}$]")
    ax.set_ylabel(r"$W_R$ [#]")
    ax.legend()

    ax = axes[1]
    ax.plot(k, Sab[0, 0, :], label=r"$S_{11}(k)$")
    ax.plot(k, Sab[0, 1, :], label=r"$S_{12}(k)$")
    ax.plot(k, Sab[1, 1, :], label=r"$S_{22}(k)$")
    ax.set_xlabel(r"$k$ [$a_B^{-1}$]")
    ax.set_ylabel(r"$S_{ii}(k)$ [#]")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    be_example()

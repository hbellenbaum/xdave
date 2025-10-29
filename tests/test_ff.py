import sys

sys.path.insert(1, "./xdave")

from constants import BOHR_RADIUS
from unit_conversions import g_per_cm3_TO_kg_per_m3, eV_TO_K, eV_TO_J, RYDBERG_TO_eV, J_TO_eV
from plasma_state import PlasmaState, get_rho_T_from_rs_theta
from models import ModelOptions
from freefree_dsf import FreeFreeDSF

import numpy as np
import matplotlib.pyplot as plt


def test_ff():

    rs = 2
    theta = 1
    rho, Te = get_rho_T_from_rs_theta(rs=rs, theta=theta)
    ks = np.array((0.5, 1.0, 2.0, 4.0)) / BOHR_RADIUS
    rho *= g_per_cm3_TO_kg_per_m3
    Te *= eV_TO_K
    # Te = 200  #
    charge_state = 1.0
    atomic_mass = 1.0
    atomic_number = 1.0
    lfc = 0.0

    models = ModelOptions(polarisation_model="NUMERICAL")
    models2 = ModelOptions(polarisation_model="DANDREA_FIT")

    omega_array = np.linspace(-100, 100, 100) * eV_TO_J
    state = PlasmaState(
        electron_temperature=Te,
        ion_temperature=Te,
        mass_density=rho,
        charge_state=charge_state,
        binding_energies=None,
        atomic_mass=atomic_mass,
        atomic_number=atomic_number,
    )

    fig, axes = plt.subplots(1, 1, figsize=(14, 8))
    colors = ["magenta", "crimson", "orange", "dodgerblue", "lightgreen", "lightgray", "yellow", "cyan"]

    for k, cs in zip(ks, colors):
        dsfs = np.zeros_like(omega_array)
        dsfs2 = np.zeros_like(omega_array)
        q = k * BOHR_RADIUS
        w = omega_array
        kernel = FreeFreeDSF(state=state)
        dsfs = kernel.get_dsf(k=k, w=w, lfc=lfc, model="NUMERICAL")
        dsfs2 = kernel.get_dsf(k=k, w=w, lfc=lfc, model="DANDREA_FIT")

        idx = np.argwhere(np.isnan(dsfs))
        dsfs_new = np.delete(dsfs, idx)
        dsfs2_new = np.delete(dsfs2, idx)
        omega_new = np.delete(omega_array, idx)

        fname = f"tests/comparison_data/ff_dsf/4hannah_rs_{int(rs)}_theta_{int(theta)}_{q}.txt"
        dat_j = np.genfromtxt(fname=fname, skip_header=22)
        axes.plot(
            dat_j[:, 0] * RYDBERG_TO_eV,
            dat_j[:, 4] / RYDBERG_TO_eV,
            ls=":",
            label=f"Jan: q={q}",
            marker="*",
            markevery=50,
            c=cs,
        )

        axes.plot(omega_new * J_TO_eV, dsfs2_new / J_TO_eV, label=f"Fit: q={q}", c=cs, ls="-.")

    axes.set_xlabel(r"$\omega$ [eV]")
    axes.set_ylabel(r"DSF [1/eV]")
    axes.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_ff()

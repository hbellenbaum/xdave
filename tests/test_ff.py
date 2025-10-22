import sys

sys.path.insert(1, "/home/bellen85/code/dev/xdave/xdave")

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
    ks = np.array((0.5, 1.0, 2.0, 4.0)) / BOHR_RADIUS  #  0.5, 1.0, 2.0, 4.0
    rho *= g_per_cm3_TO_kg_per_m3
    Te *= eV_TO_K
    # Te = 200  #
    charge_state = 1.0
    atomic_mass = 1.0
    atomic_number = 1.0
    lfc = 0.0

    models = ModelOptions(polarisation_model="NUMERICAL")
    models2 = ModelOptions(polarisation_model="DANDREA_FIT")

    omega_array = np.linspace(-100, 100, 100) * eV_TO_J  # + 8.5e3 * eV_TO_J
    state = PlasmaState(
        electron_temperature=Te,
        ion_temperature=Te,
        mass_density=rho,
        charge_state=charge_state,
        binding_energies=None,
        # frequency=omega,
        atomic_mass=atomic_mass,
        atomic_number=atomic_number,
    )

    # models = ModelOptions(polarisation_model=model)
    fig, axes = plt.subplots(1, 1, figsize=(14, 8))
    colors = ["magenta", "crimson", "orange", "dodgerblue", "lightgreen", "lightgray", "yellow", "cyan"]

    for k, cs in zip(ks, colors):
        # int_terms = []
        # real_dielectrics = np.zeros_like(omega_array)
        # im_dielectric = np.zeros_like(omega_array)
        dsfs = np.zeros_like(omega_array)
        dsfs2 = np.zeros_like(omega_array)
        q = k * BOHR_RADIUS

        # ims_dandrea = np.zeros_like(omega_array)
        # ims_rpa = np.zeros_like(omega_array)
        # reals_dandrea = np.zeros_like(omega_array)
        # reals_rpa = np.zeros_like(omega_array)
        # for i in range(0, len(omega_array)):
        w = omega_array  # [i]
        kernel = FreeFreeDSF(state=state)
        # int_term = kernel._real_dielectric_rpa(k=k, w=w)
        dsfs = kernel.get_dsf(k=k, w=w, lfc=lfc, model="NUMERICAL")
        dsfs2 = kernel.get_dsf(k=k, w=w, lfc=lfc, model="DANDREA_FIT")
        # dielectric_func, im_part_rpa, real_part_rpa = kernel.rpa_numerical_dielectric_func(k, w)
        # real_dielectrics[i] = np.real(dielectric_func)
        # im_dielectric[i] = np.imag(dielectric_func)
        # dsfs[i] = dsf
        # dsfs2[i] = dsf2

        idx = np.argwhere(np.isnan(dsfs))
        dsfs_new = np.delete(dsfs, idx)
        dsfs2_new = np.delete(dsfs2, idx)
        omega_new = np.delete(omega_array, idx)
        # dsfs_new *= 1 / J_TO_eV  # DIRAC_CONSTANT
        # twinx = axes.twinx()
        # axes.plot(omega_new * J_TO_eV, dsfs_new / J_TO_eV, label=f"q={q} 1/aB", c=cs)  #  /  np.max(dsfs_new)

        # axes[1].plot(omega_array * J_TO_eV, real_dielectrics, label=f"k={k}", c=cs)
        # axes[2].plot(omega_array * J_TO_eV, im_dielectric, label=f"k={k}", c=cs)

        fname = f"tests/comparison_data/ff_dsf/4hannah_rs_{int(rs)}_theta_{int(theta)}_{q}.txt"
        dat_j = np.genfromtxt(fname=fname, skip_header=22)
        # print(dat_j)
        axes.plot(
            dat_j[:, 0] * RYDBERG_TO_eV,
            dat_j[:, 4] / RYDBERG_TO_eV,
            ls=":",
            label=f"Jan: q={q}",
            marker="*",
            markevery=50,
            c=cs,
        )  # / np.max(dat_j[:, 4])
        # ax0.plot(dat_j[:, 0] * RYBBERG_TO_eV, dat_j[:, 5] / np.max(dat_j[:, 5]), c=c, ls="dotted", label=f"LFC: q={q}")
        axes.plot(omega_new * J_TO_eV, dsfs2_new / J_TO_eV, label=f"Fit: q={q}", c=cs, ls="-.")  #  /  np.max(dsfs_new)

    axes.set_xlabel(r"$\omega$ [eV]")
    axes.set_ylabel(r"DSF [1/eV]")
    # axes[1].set_xlabel(r"$\omega$ [eV]")
    # axes[1].set_ylabel(r"$Re\{\epsilon^{RPA}\}$")
    # axes[2].set_xlabel(r"$\omega$ [eV]")
    # axes[2].set_ylabel(r"$Im\{\epsilon^{RPA}\}$")
    axes.legend()

    # axes2[0].legend()
    plt.tight_layout()
    plt.show()
    # fig.savefig("ff_dsf_test3.pdf", dpi=200)


if __name__ == "__main__":
    test_ff()

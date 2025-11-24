import sys

# sys.path.insert(1, "./xdave")

from xdave.constants import BOHR_RADIUS, ELECTRON_MASS, DIRAC_CONSTANT
from xdave.unit_conversions import g_per_cm3_TO_kg_per_m3, eV_TO_K, eV_TO_J, RYDBERG_TO_eV, J_TO_eV
from xdave.plasma_state import PlasmaState, get_rho_T_from_rs_theta
from xdave.models import ModelOptions
from xdave.freefree_dsf import FreeFreeDSF

import numpy as np
import matplotlib.pyplot as plt


def test_ff():

    rs = 2
    theta = 1
    rho, Te = get_rho_T_from_rs_theta(rs=rs, theta=theta)
    ks = np.array((1.0,)) / BOHR_RADIUS  # 0.5, 1.0, 2.0, 4.0
    rho *= g_per_cm3_TO_kg_per_m3
    Te *= eV_TO_K
    # Te = 200  #
    charge_state = 1.0
    atomic_mass = 1.0
    atomic_number = 1.0
    lfc = 0.0

    omega_array = np.linspace(-100, 100, 5000) * eV_TO_J
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

        axes.plot(omega_new * J_TO_eV, dsfs2_new / J_TO_eV, label=f"RPA: q={q}", c=cs, ls="-.")

    axes.set_xlabel(r"$\omega$ [eV]")
    axes.set_ylabel(r"DSF [1/eV]")
    axes.legend()
    plt.tight_layout()
    plt.show()


def test_mermin_ff():
    rs = 2
    theta = 1
    rho, Te = get_rho_T_from_rs_theta(rs=rs, theta=theta)
    ks = np.array((2.0,)) / BOHR_RADIUS  # 0.5, 1.0, 2.0, 4.0
    rho *= g_per_cm3_TO_kg_per_m3
    Te *= eV_TO_K
    # Te = 200  #
    charge_state = 1.0
    atomic_mass = 1.0
    atomic_number = 1.0
    lfc = 0.0

    omega_array = np.linspace(-100, 150, 5000) * eV_TO_J
    state = PlasmaState(
        electron_temperature=Te,
        ion_temperature=Te,
        mass_density=rho,
        charge_state=charge_state,
        binding_energies=None,
        atomic_mass=atomic_mass,
        atomic_number=atomic_number,
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 8))
    colors = ["magenta", "crimson", "orange", "dodgerblue", "lightgreen", "lightgray", "yellow", "cyan"]

    for k, cs in zip(ks, colors):
        omega_p = state.plasma_frequency(-1, state.free_electron_number_density, ELECTRON_MASS) * DIRAC_CONSTANT
        mu_ei = 0.5 * omega_p * (1.0 + 1.0j)
        q = k * BOHR_RADIUS
        w = omega_array
        kF = state.fermi_wave_number(state.free_electron_number_density)  # 1/m
        EF = state.fermi_energy(state.free_electron_number_density, ELECTRON_MASS)  # J
        vF = DIRAC_CONSTANT * kF / ELECTRON_MASS  # m/s
        u = w / (k * vF * DIRAC_CONSTANT)  # dimensionless
        u_mermin = (w + 1.0j * mu_ei) / (k * vF * DIRAC_CONSTANT)
        z = k / (2 * kF)  # dimensionless
        kernel = FreeFreeDSF(state=state)
        dsfs_rpa = kernel.get_dsf(k=k, w=w, lfc=lfc, model="NUMERICAL")
        dsfs_mermin = kernel.get_dsf(k=k, w=w, lfc=lfc, model="MERMIN")
        # print(dsfs_mermin)
        dielectric_rpa = kernel.dielectric_function(k=k, w=w, model="NUMERICAL")
        dielectric_mermin = kernel.dielectric_function(k=k, w=w, model="MERMIN")

        axes[0].plot(omega_array * J_TO_eV, dsfs_mermin / J_TO_eV, label=f"Mermin: q={q}", c=cs, ls=":")
        axes[0].plot(omega_array * J_TO_eV, dsfs_rpa / J_TO_eV, label=f"RPA: q={q}", c=cs, ls="--")
        axes[1].plot(omega_array * J_TO_eV, dielectric_mermin.real, label=f"Re[Mermin]: q={q}", c=cs, ls=":")
        axes[1].plot(omega_array * J_TO_eV, dielectric_rpa.real, label=f"Re[RPA]: q={q}", c=cs, ls="--")
        axes[1].plot(omega_array * J_TO_eV, dielectric_mermin.imag, label=f"Im[Mermin]: q={q}", c="navy", ls=":")
        axes[1].plot(omega_array * J_TO_eV, dielectric_rpa.imag, label=f"Im[RPA]: q={q}", c="navy", ls="--")

        axes[2].plot(u, dielectric_mermin.real, label=f"Re[Mermin]: q={q}", c=cs, ls=":")
        axes[2].plot(u, dielectric_rpa.real, label=f"Re[RPA]: q={q}", c=cs, ls="--")
        axes[2].plot(u_mermin, dielectric_mermin.imag, label=f"Im[Mermin]: q={q}", c="navy", ls=":")
        axes[2].plot(u_mermin, dielectric_rpa.imag, label=f"Im[RPA]: q={q}", c="navy", ls="--")

    axes[0].set_xlabel(r"$\omega$ [eV]")
    axes[0].set_ylabel(r"DSF [1/eV]")
    axes[0].legend()
    # axes[0].set_ylim(-0.005, 0.02)
    axes[1].set_xlabel(r"$\omega$ [eV]")
    axes[1].set_ylabel(r"$\epsilon$")
    axes[1].legend()

    axes[2].set_xlabel(r"$u$")
    axes[2].set_ylabel(r"$\epsilon$")
    axes[2].legend()
    # axes[1].set_ylim(-0.005, 0.02)
    plt.tight_layout()
    plt.show()
    # fig.savefig(f"ff_test_mermin.pdf", dpi=200)


if __name__ == "__main__":
    test_ff()
    # test_mermin_ff()

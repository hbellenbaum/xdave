from xdave.constants import BOHR_RADIUS, ELECTRON_MASS, DIRAC_CONSTANT
from xdave.utils import calculate_angle, calculate_q
from xdave.unit_conversions import g_per_cm3_TO_kg_per_m3, eV_TO_K, eV_TO_J, RYDBERG_TO_eV, J_TO_eV
from xdave.plasma_state import PlasmaState, get_rho_T_from_rs_theta
from xdave.freefree_dsf import FreeFreeDSF

import numpy as np
import matplotlib.pyplot as plt

import os


def test_ff():

    rs = 2
    theta = 1
    rho, Te = get_rho_T_from_rs_theta(rs=rs, theta=theta)
    ks = np.array((0.5, 1.0, 2.0, 4.0)) / BOHR_RADIUS  # 0.5, 1.0, 2.0, 4.0
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

        axes.plot(omega_new * J_TO_eV, dsfs2_new / J_TO_eV, label=f"RPA Fit: q={q}", c=cs, ls="-.")
        axes.plot(omega_new * J_TO_eV, dsfs_new / J_TO_eV, label=f"RPA: q={q}", c=cs, ls="solid")

    axes.set_xlabel(r"$\omega$ [eV]")
    axes.set_ylabel(r"DSF [1/eV]")
    axes.legend()
    plt.tight_layout()
    plt.show()

    rtol = 1.0e-2

    if not np.isclose(
        dsfs_new / J_TO_eV,
        np.interp(x=omega_array * J_TO_eV, xp=dat_j[:, 0] * RYDBERG_TO_eV, fp=dat_j[:, 4] / RYDBERG_TO_eV),
        rtol=rtol,
    ).all():
        print(f"RPA test has failed.")
    if not np.isclose(
        dsfs2_new / J_TO_eV,
        np.interp(x=omega_array * J_TO_eV, xp=dat_j[:, 0] * RYDBERG_TO_eV, fp=dat_j[:, 4] / RYDBERG_TO_eV),
        rtol=rtol,
    ).all():
        print(f"Dandrea test has failed.")


def test_mermin_ff():
    # rs = 2
    # theta = 1
    # rho, Te = get_rho_T_from_rs_theta(rs=rs, theta=theta)
    angles = np.array([40, 120])
    qs = calculate_q(angle=angles, energy=3e3)
    ks = qs / BOHR_RADIUS  # np.array((2.0,)) / BOHR_RADIUS  # 0.5, 1.0, 2.0, 4.0
    # rho *= g_per_cm3_TO_kg_per_m3
    # Te *= eV_TO_K
    # Te = 200  #
    # charge_state = 1.0
    atomic_mass = 13.0
    atomic_number = 13.0
    lfc = 0.0
    rho = 1.85 * g_per_cm3_TO_kg_per_m3
    Te = 12 * eV_TO_K
    charge_state = 2.45

    omega_array = np.linspace(-50, 50, 500) * eV_TO_J
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
        # mu_ei = 1.5 * omega_p * (1.0 + 1.0j)
        input_collision_frequency = 0.5
        q = k * BOHR_RADIUS
        w = omega_array
        # kF = state.fermi_wave_number(state.free_electron_number_density)  # 1/m
        # EF = state.fermi_energy(state.free_electron_number_density, ELECTRON_MASS)  # J
        # vF = DIRAC_CONSTANT * kF / ELECTRON_MASS  # m/s
        # u = w / (k * vF * DIRAC_CONSTANT)  # dimensionless
        # u_mermin = (w + 1.0j * mu_ei) / (k * vF * DIRAC_CONSTANT)
        # z = k / (2 * kF)  # dimensionless
        kernel = FreeFreeDSF(state=state)

        nu_ei = 0.5  # [w_p
        dsfs_rpa = kernel.get_dsf(k=k, w=w, lfc=lfc, model="NUMERICAL")
        dsfs_born_mermin = kernel.get_dsf(k=k, w=w, lfc=lfc, model="MERMIN")
        dsfs_static_mermin = kernel.get_dsf(
            k=k, w=w, lfc=lfc, model="MERMIN", input_collision_frequency=input_collision_frequency
        )
        # print(dsfs_mermin)
        dielectric_rpa = kernel.dielectric_function(k=k, w=w, model="NUMERICAL")
        dielectric_born_mermin = kernel.dielectric_function(k=k, w=w, model="MERMIN")
        dielectric_static_mermin = kernel.dielectric_function(
            k=k, w=w, model="MERMIN", input_collision_frequency=input_collision_frequency
        )

        mu_ei_born = kernel.get_collision_frequency(k=k, w=w, model="BORN", lfc=0.0)

        axes[0].plot(omega_array * J_TO_eV, dsfs_born_mermin / J_TO_eV, label=f"BM: q={q}", c=cs, ls=":")
        axes[0].plot(omega_array * J_TO_eV, dsfs_rpa / J_TO_eV, label=f"RPA: q={q}", c=cs, ls="--")
        axes[0].plot(
            omega_array * J_TO_eV, dsfs_static_mermin / J_TO_eV, label=f"Static M: q={q}", c=cs, ls="-.", alpha=0.7
        )

        if cs == "magenta":
            axes[1].plot(omega_array * J_TO_eV, dielectric_rpa.real, label=f"Re[RPA]: q={q}", c=cs, ls="solid")
            axes[1].plot(omega_array * J_TO_eV, dielectric_rpa.imag, label=f"Im[RPA]: q={q}", c=cs, ls="-.")
            axes[1].plot(
                omega_array * J_TO_eV, dielectric_born_mermin.real, label=f"Re[BM]: q={q}", c="navy", ls="solid"
            )
            axes[1].plot(omega_array * J_TO_eV, dielectric_born_mermin.imag, label=f"Im[BM]: q={q}", c="navy", ls="-.")
            axes[1].plot(
                omega_array * J_TO_eV,
                dielectric_static_mermin.real,
                label=f"Re[M]: q={q}",
                c="lightgreen",
                ls="solid",
                alpha=0.7,
            )
            axes[1].plot(
                omega_array * J_TO_eV,
                dielectric_static_mermin.imag,
                label=f"Im[M]: q={q}",
                c="lightgreen",
                ls="-.",
                alpha=0.7,
            )
        axes[2].plot(omega_array * J_TO_eV, mu_ei_born.real, label=f"Re[mu]: q={q}", c=cs, ls="-.")
        axes[2].plot(omega_array * J_TO_eV, mu_ei_born.imag, label=f"Im[mu]: q={q}", c=cs, ls="-.")

    axes[0].set_xlabel(r"$\omega$ [eV]")
    axes[0].set_ylabel(r"DSF [1/eV]")
    axes[0].legend()
    axes[1].set_xlabel(r"$\omega$ [eV]")
    axes[1].set_ylabel(r"$\epsilon$")
    axes[1].legend()
    axes[2].set_xlabel(r"$\omega$ [eV]")
    axes[2].set_ylabel(r"$\mu$")
    axes[2].legend()

    plt.tight_layout()
    plt.show()
    # fig.savefig(f"ff_test_mermin.pdf", dpi=200)


if __name__ == "__main__":
    # test_ff()
    test_mermin_ff()

from xdave.constants import BOHR_RADIUS, ELECTRON_MASS, DIRAC_CONSTANT
from xdave.utils import calculate_angle, calculate_q
from xdave.unit_conversions import g_per_cm3_TO_kg_per_m3, eV_TO_K, eV_TO_J, RYDBERG_TO_eV, J_TO_eV
from xdave.plasma_state import PlasmaState, get_rho_T_from_rs_theta
from xdave.models import ModelOptions
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
        axes.plot(omega_new * J_TO_eV, dsfs_new / J_TO_eV, label=f"RPA Fit: q={q}", c=cs, ls="solid")

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

    omega_array = np.linspace(-200, 250, 5000) * eV_TO_J
    state = PlasmaState(
        electron_temperature=Te,
        ion_temperature=Te,
        mass_density=rho,
        charge_state=charge_state,
        binding_energies=None,
        atomic_mass=atomic_mass,
        atomic_number=atomic_number,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
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
        # twinx0 = axes[0].twinx()
        # twinx1 = axes[1].twinx()
        # twinx2 = axes[2].twinx()

        axes[0].plot(omega_array * J_TO_eV, dsfs_mermin / J_TO_eV, label=f"Mermin: q={q}", c=cs, ls=":")
        axes[0].plot(omega_array * J_TO_eV, dsfs_rpa / J_TO_eV, label=f"RPA: q={q}", c=cs, ls="--")
        axes[1].plot(omega_array * J_TO_eV, dielectric_mermin.real, label=f"Re[Mermin]: q={q}", c=cs, ls=":")
        axes[1].plot(omega_array * J_TO_eV, dielectric_rpa.real, label=f"Re[RPA]: q={q}", c=cs, ls="--")
        axes[1].plot(omega_array * J_TO_eV, dielectric_mermin.imag, label=f"Im[Mermin]: q={q}", c="navy", ls=":")
        axes[1].plot(omega_array * J_TO_eV, dielectric_rpa.imag, label=f"Im[RPA]: q={q}", c="navy", ls="--")

        # twinx2.plot(u, dielectric_mermin.real, label=f"Re[Mermin]: q={q}", c=cs, ls=":")
        # axes[2].plot(u, dielectric_rpa.real, label=f"Re[RPA]: q={q}", c=cs, ls="--")
        # twinx2.plot(u_mermin, dielectric_mermin.imag, label=f"Im[Mermin]: q={q}", c="navy", ls=":")
        # axes[2].plot(u_mermin, dielectric_rpa.imag, label=f"Im[RPA]: q={q}", c="navy", ls="--")

    axes[0].set_xlabel(r"$\omega$ [eV]")
    axes[0].set_ylabel(r"DSF [1/eV]")
    axes[0].legend()
    # axes[0].set_ylim(-0.005, 0.02)
    axes[1].set_xlabel(r"$\omega$ [eV]")
    axes[1].set_ylabel(r"$\epsilon$")
    axes[1].legend()

    # axes[2].set_xlabel(r"$u$")
    # axes[2].set_ylabel(r"$\epsilon$")
    # axes[2].legend()
    # axes[1].set_ylim(-0.005, 0.02)
    plt.tight_layout()
    plt.show()
    # fig.savefig(f"ff_test_mermin.pdf", dpi=200)


def update_ff_results(model, w, k_bohr, dsf, fn):
    file = fn + f"_k={k_bohr:.1f}_model={model}.csv"
    np.savetxt(file, np.array([w, dsf]).T)
    print(f"Updating FF test for model = {model}: \nfile = {file}")


def test_version():
    Te = 50 * eV_TO_K
    rho = 10.0 * g_per_cm3_TO_kg_per_m3
    atomic_number = 4
    atomic_mass = 1.0
    beam_energy = 9.0e3
    charge_state = 3.0

    ZA = atomic_number
    Zb = atomic_number - charge_state

    angles = np.array([13, 30, 45, 60, 80, 100, 120, 140, 160])
    ks = calculate_q(angle=angles, energy=beam_energy) / BOHR_RADIUS
    omega_array = np.linspace(-450, 800, 1000) * eV_TO_J
    binding_energies = np.array([-111.5, -111.5, -111.5]) * eV_TO_J
    state = PlasmaState(
        electron_temperature=Te,
        ion_temperature=Te,
        mass_density=rho,
        charge_state=charge_state,
        atomic_mass=atomic_mass,
        atomic_number=atomic_number,
        binding_energies=binding_energies,
    )

    output_dir = os.path.join(os.path.dirname(__file__), "xdave_results/ff/")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    fn = output_dir + f"dsf_check_be_T={Te/eV_TO_K:.0f}_rho={rho/g_per_cm3_TO_kg_per_m3:.0f}_Z={charge_state}"

    for i in range(len(ks)):
        k = ks[i]
        k_bohr = k * BOHR_RADIUS
        angle = np.round(calculate_angle(q=k * BOHR_RADIUS, energy=beam_energy))  # angles[i]
        print(f"Running for k={k * BOHR_RADIUS} 1/aB and angle={angle}")

        kernel = FreeFreeDSF(state=state)
        dsf_lindhard = kernel.get_dsf(k=k, w=omega_array, lfc=0.0, model="LINDHARD")
        # update_ff_results(model="LINDHARD", w=omega_array, k_bohr=k_bohr, dsf=dsf_lindhard, fn=fn)
        dsf_lindhard_save = np.genfromtxt(fn + f"_k={k_bohr:.1f}_model=LINDHARD.csv", delimiter=" ")

        dsf_rpa = kernel.get_dsf(k=k, w=omega_array, lfc=0.0, model="NUMERICAL")
        # update_ff_results(model="RPA", w=omega_array, k_bohr=k_bohr, dsf=dsf_lindhard, fn=fn)
        dsf_rpa_save = np.genfromtxt(fn + f"_k={k_bohr:.1f}_model=RPA.csv", delimiter=" ")

        dsf_fit = kernel.get_dsf(k=k, w=omega_array, lfc=0.0, model="DANDREA_FIT")
        # update_ff_results(model="Fit", w=omega_array, k_bohr=k_bohr, dsf=dsf_lindhard, fn=fn)
        dsf_fit_save = np.genfromtxt(fn + f"_k={k_bohr:.1f}_model=Fit.csv", delimiter=" ")

        if False:
            # I am currently ignoring Mermin because it is not working yet
            dsf_mermin = kernel.get_dsf(k=k, w=omega_array, lfc=0.0, model="MERMIN")
            update_ff_results(model="MERMIN", w=omega_array, k_bohr=k_bohr, dsf=dsf_lindhard, fn=fn)
            dsf_mermin_save = np.genfromtxt(fn + f"_k={k_bohr:.1f}_model=MERMIN.csv", delimiter=" ")

        if not np.isclose(dsf_lindhard, dsf_lindhard_save[:, 1], rtol=1.0e-2).all(axis=-1):
            print(f"Lindhard FF model has failed the test at k={k_bohr:.1f} 1/aB.")
        if not np.isclose(dsf_rpa, dsf_rpa_save[:, 1], rtol=1.0e-2).all(axis=-1):
            print(f"Numerical RPA FF model has failed the test at k={k_bohr:.1f} 1/aB.")
        if np.isclose(dsf_fit, dsf_fit_save[:, 1], rtol=1.0e-2).all(axis=-1):
            print(f"Dandrea RPA Fit FF model has failed the test at k={k_bohr:.1f} 1/aB.")
        # if np.isclose(dsf_mermin, dsf_mermin_save[:, 1], rtol=1.0e-2).all(axis=-1):
        #     print(f"Mermin FF model has failed the test at k={k_bohr:.1f} 1/aB.")


if __name__ == "__main__":
    # test_ff()
    # test_mermin_ff()
    test_version()

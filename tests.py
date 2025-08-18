from plasma_state import PlasmaState, get_rho_T_from_rs_theta
from models import ModelOptions
from unit_conversions import *
from constants import BOHR_RADIUS, PLANCK_CONSTANT
from freefree_dsf import FreeFreeDSF
from utils import calculate_angle

import numpy as np
import matplotlib.pyplot as plt


def test_chemical_potential():
    T = 1.0 * eV_TO_K
    rho = 0.01 * g_per_cm3_TO_kg_per_m3
    Z = 1.0
    AN = 1
    atomic_mass = 1.0 * amu_TO_kg
    state = PlasmaState(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_state=Z,
        atomic_mass=atomic_mass,
        atomic_number=AN,
    )
    n = state.electron_number_density

    TF_erg = state.fermi_temperature(mass=ELECTRON_MASS, number_density=n) * K_TO_erg
    EF = state.fermi_energy(mass=ELECTRON_MASS, number_density=n)
    EF_erg = EF * J_TO_erg
    Thetas = np.linspace(0.01, 4, 100)
    mus = np.zeros_like(Thetas)
    mus_fit = np.zeros_like(Thetas)
    mus_high = np.zeros_like(Thetas)
    mus_low = np.zeros_like(Thetas)
    mus_classical = np.zeros_like(Thetas)

    for i in range(0, len(Thetas)):
        theta = Thetas[i]
        T = theta * TF_erg * erg_TO_K
        state.electron_temperature = T
        mu_young, mu_high, mu_low = state.chemical_potential(temperature=T, number_density=n, mass=ELECTRON_MASS)
        mus_fit[i] = state.chemical_potential_ichimaru(temperature=T, number_density=n, mass=ELECTRON_MASS)
        mus_classical[i] = state.chemical_potential_classical(temperature=T, number_density=n, mass=ELECTRON_MASS)
        mus[i] = mu_young
        mus_high[i] = mu_high
        mus_low[i] = mu_low

    mus /= EF_erg
    mus_low /= EF_erg
    mus_high /= EF_erg
    mus_classical /= EF_erg
    mus_fit /= EF

    print(mus_fit)
    print(mus)
    plt.figure(figsize=(8, 13))
    plt.xlabel(r"$T/T_F$")
    plt.ylabel(r"$\mu / E_F$")
    plt.plot(Thetas, mus, label="numerical", ls="-.", lw=3)
    plt.plot(Thetas, mus_fit, label="fit", ls="--", lw=3)
    plt.plot(Thetas, mus_low, label="low T", ls=":", lw=3)
    plt.plot(Thetas, mus_high, label="high T", ls=":", lw=3)
    plt.plot(Thetas, mus_classical, label="classical", ls=":", lw=3)
    plt.ylim(-30.0, 1.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"chemical_potential_models_compared_rho={rho*kg_per_m3_TO_g_per_cm3:.2f}.pdf")


def test_ff_rpa():

    rs = 2
    theta = 1
    rho, Te = get_rho_T_from_rs_theta(rs=rs, theta=theta)
    ks = np.array((0.5, 1.0, 2.0, 4.0)) / BOHR_RADIUS
    rho *= g_per_cm3_TO_kg_per_m3
    Te *= eV_TO_K
    charge_state = 1.0
    atomic_mass = 1.0
    atomic_number = 1.0
    lfc = 0.0

    models = ModelOptions(polarisation_model="NUMERICAL")
    models2 = ModelOptions(polarisation_model="DANDREA_FIT")

    omega_array = np.linspace(-100, 100, 500) * eV_TO_J
    state = PlasmaState(
        electron_temperature=Te,
        ion_temperature=Te,
        mass_density=rho,
        charge_state=charge_state,
        atomic_mass=atomic_mass,
        atomic_number=atomic_number,
    )

    fig, axes = plt.subplots(1, 1, figsize=(14, 8))
    colors = ["magenta", "crimson", "orange", "dodgerblue", "lightgreen", "lightgray", "yellow", "cyan"]

    for k, cs in zip(ks, colors):
        dsfs = np.zeros_like(omega_array)
        dsfs2 = np.zeros_like(omega_array)
        q = k * BOHR_RADIUS

        for i in range(0, len(omega_array)):
            w = omega_array[i]
            kernel = FreeFreeDSF(state=state, models=models)
            kernel2 = FreeFreeDSF(state=state, models=models2)

            dsf = kernel.get_dsf(k=k, w=w, lfc=lfc)
            dsf2 = kernel2.get_dsf(k=k, w=w, lfc=lfc)

            dsfs[i] = dsf
            dsfs2[i] = dsf2

        idx = np.argwhere(np.isnan(dsfs))
        dsfs_new = np.delete(dsfs, idx)
        dsfs2_new = np.delete(dsfs2, idx)
        omega_new = np.delete(omega_array, idx)

        axes.plot(omega_new * J_TO_eV, dsfs_new / J_TO_eV, label=f"q={q} 1/aB", c=cs)

        fname = f"validation/ff_dsf/4hannah_rs_{int(rs)}_theta_{int(theta)}_{q}.txt"
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
    fig.savefig("ff_dsf_test3.pdf", dpi=200)


def test_ff_mcss():
    rs = 2
    theta = 1
    rho, Te = get_rho_T_from_rs_theta(rs=rs, theta=theta)
    ks = np.array((0.5,)) / BOHR_RADIUS
    rho *= g_per_cm3_TO_kg_per_m3
    Te *= eV_TO_K
    charge_state = 1.0
    atomic_mass = 1.0
    atomic_number = 1.0
    lfc = 0.0

    omega_array = np.linspace(-150, 300, 5000) * eV_TO_J
    state = PlasmaState(
        electron_temperature=Te,
        ion_temperature=Te,
        mass_density=rho,
        charge_state=charge_state,
        atomic_mass=atomic_mass,
        atomic_number=atomic_number,
    )
    model = "DANDREA_FIT"
    if model == "LINDHARD":
        mcss_model = "LINDHARD_RPA"
    elif model == "DANDREA_FIT":
        mcss_model = "DANDREA_RPA_FIT"
    elif model == "NUMERICAL_RPA":
        mcss_model = "NUMERICAL_RPA"
    models = ModelOptions(polarisation_model=model)
    colors = ["magenta", "crimson", "orange", "dodgerblue", "lightgreen", "lightgray", "yellow", "cyan"]
    fig, ax0 = plt.subplots(figsize=(14, 10))
    i = 0
    Hz_TO_eV = 4.1357e-15  # eV

    norm_factor = PLANCK_CONSTANT

    print(f"\nNormalised using factor = {norm_factor}\n")

    for k, c in zip(ks, colors):
        q = k * BOHR_RADIUS
        angle = calculate_angle(q=q, energy=8.0e3)
        angle = int(np.round(angle, 0))
        dsfs = []
        dsfs = np.zeros_like(omega_array)
        for i in range(0, len(omega_array)):
            omega = omega_array[i]
            kernel = FreeFreeDSF(state=state, models=models)
            dsf = kernel.get_dsf(k=k, w=omega, lfc=lfc)
            dsfs[i] = dsf

        # Run MCSS
        mcss_fn = f"mcss_tests/mcss_outputs_model={mcss_model}/mcss_ff_test_angle={angle}.csv"
        En, Es, _, wff, wbf, Pff, Pbf, Pel, tot = np.genfromtxt(mcss_fn, unpack=True, delimiter=",", skip_header=1)

        # Compare results
        dsfs *= norm_factor
        ax0.plot(
            omega_array[::-1] * J_TO_eV,
            np.array(dsfs[::-1]),
            label=f"$q$={q}",
            c=c,
        )

        fname = f"validation/ff_dsf/4hannah_rs_{int(rs)}_theta_{int(theta)}_{q}.txt"
        dat_j = np.genfromtxt(fname=fname, skip_header=22)
        norm_Jan = 1 / (RYDBERG_TO_eV * eV_TO_J)
        norm_mcss = 1 / (eV_TO_J)
        print(wff)
        twinx = ax0.twinx()
        twinx.plot(En[::-1], wff[::-1] * norm_mcss, label="MCSS", c=c, ls="dotted")
        twinx.plot(dat_j[:, 0] * RYDBERG_TO_eV, dat_j[:, 4] * norm_Jan, c=c, ls="dashed", label=f"RPA: q={q}")

        print(
            f"Maxima:\n"
            f"Jan: {np.max(dat_j[:, 4] * norm_Jan)} 1/J[?] ---> MCSS: {np.max(wff) * norm_mcss} [1/J] ---> me: {np.max(dsfs)} [wrong]\n"
            f"Ratio:  {np.max(dat_j[:, 4] * norm_Jan) / np.max(dsfs)}\n"
            f"1/ratio: {np.max(dsfs) / np.max(dat_j[:, 4] * norm_Jan)}\n"
        )

    ax0.legend()
    ax0.set_xlabel(r"$\omega$ [eV]")
    ax0.set_ylabel(r"$S_{ff}$ [mystery]")
    twinx.set_ylabel(r"$S_{ff}$ [1/J]")
    twinx.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
    fig.savefig(f"ff_comparison_rs={rs}_theta={theta}.pdf", dpi=200)
    plt.close()


if __name__ == "__main__":
    # test_chemical_potential()
    test_ff_rpa()

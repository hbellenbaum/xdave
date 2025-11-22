import sys

sys.path.insert(1, "./xdave")

from unit_conversions import (
    amu_TO_kg,
    eV_TO_K,
    K_TO_erg,
    erg_TO_K,
    J_TO_erg,
    g_per_cm3_TO_kg_per_m3,
    kg_per_m3_TO_g_per_cm3,
)
from constants import ELECTRON_MASS
from plasma_state import PlasmaState

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
        binding_energies=None,
    )
    n = state.total_electron_number_density

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


if __name__ == "__main__":
    test_chemical_potential()

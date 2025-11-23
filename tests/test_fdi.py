import sys

sys.path.insert(1, "./xdave")
sys.path.insert(1, "./mcss_tests")

from constants import ELECTRON_MASS
from unit_conversions import eV_TO_K, g_per_cm3_TO_kg_per_m3, J_TO_eV
from plasma_state import PlasmaState
from fermi_integrals import fdi as xdave_fdi
from plasmapy.formulary.mathematics import Fermi_integral as fdi

from scipy.special import gamma
import numpy as np
import matplotlib.pyplot as plt


def test():
    # T = 10  # eV
    rho = 5  # g/cc

    js = np.array([-1.5, -0.5, 0.5, 1.5, 2.5])
    temperatues = np.linspace(1, 40, 50)  # eV

    fig, axes = plt.subplots(2, 1, figsize=(10, 14), sharex=True)

    colors = ["dodgerblue", "magenta", "lightgreen", "navy", "crimson"]

    for j, c in zip(js, colors):
        etas = []
        fs_plasmapy = []
        fs_xdave = []
        for T in temperatues:

            state = PlasmaState(
                electron_temperature=T * eV_TO_K,
                ion_temperature=T * eV_TO_K,
                mass_density=rho * g_per_cm3_TO_kg_per_m3,
                charge_state=4,
                atomic_mass=12.011,
                atomic_number=6,
                binding_energies=None,
            )

            mue_ichimaru = state.chemical_potential_ichimaru(
                temperature=state.electron_temperature,
                number_density=state.free_electron_number_density,
                mass=ELECTRON_MASS,
            )

            EF = state.fermi_energy(number_density=state.free_electron_number_density, mass=ELECTRON_MASS)

            eta = mue_ichimaru / EF
            etas.append(eta)
            # j = -0.5

            normalize = True
            f_plasmapy = fdi(j=j, x=eta).real
            f_xdave = xdave_fdi(j=j, eta=eta, normalize=normalize)
            fs_plasmapy.append(f_plasmapy)
            fs_xdave.append(f_xdave)

            # print(f"\n T = {T} eV")
            # print(f"Plasmapy FDI for j={j}: {f_plasmapy}")
            # print(f"xDave FDI for j={j}: {f_xdave} with norm={normalize}")
            # print(f"Diff = {abs(f_plasmapy - f_xdave)}")
        axes[0].plot(etas, fs_plasmapy, label=f"Plasmapy j = {j}", ls="-.", c=c)
        axes[0].plot(etas, fs_xdave, label=f"xDave j = {j}", ls=":", c=c)
        axes[1].plot(etas, abs(np.array(fs_plasmapy) - np.array(fs_xdave)), c=c, ls="solid", label="Diff")

    axes[0].set_title(f"Checking FDI at constant rho = {rho} g/cc, normalize = {normalize}")
    axes[0].legend()
    axes[0].set_ylabel(r"$F_j(\eta)$")
    axes[1].legend()
    axes[1].set_xlabel(r"$\mu_e / E_F$ [ ]")
    axes[1].set_ylabel("|Diff|")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test()

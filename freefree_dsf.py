from constants import *
from unit_conversions import *
from plasma_state import PlasmaState
from models import ModelOptions
import numpy as np


def effective_coulomb_potential(ionisation, wave_number):
    r"""
    Effective Coulomb potential
    """
    V_aa = -4 * PI * ionisation * ELEMENTARY_CHARGE_SQR * UNIT_COULOMB_POTENTIAL / (wave_number * wave_number)
    return V_aa


def lindhard_rpa(state: PlasmaState, wave_number):
    """
    Limiting case for full degeneracy
    """
    z = state.frequency / (4 * state.fermi_frequency())
    q = (wave_number / 2) / state.fermi_wave_number()
    x_pos = z / q + q
    x_neg = z / q - q

    def func(x):
        return x + 0.5 * (1 - x**2) * np.log((x + 1) / (x - 1))

    pol_func = 3 * state.electron_number_density / (4 * state.fermi_energy() * q) * (func(x_pos) - func(x_neg))
    return pol_func


class FreeFreeDSF:

    def __init__(self, state: PlasmaState) -> None:
        self.state = state

    def get_dsf(self, k, w, lfc):
        pol_func = self.polarisation_function(k, w, lfc)
        beta = 1 / (self.state.electron_temperature * BOLTZMANN_CONSTANT)
        potential_func = 4 * np.pi * COULOMB_CONSTANT * ELEMENTARY_CHARGE**2 / k**2
        dielectric_func = 1 - potential_func * pol_func
        L_ee = 2 / (np.exp(-beta * DIRAC_CONSTANT * w) - 1) * np.imag(pol_func / dielectric_func)
        S_EG = DIRAC_CONSTANT / (2 * PI * self.state.electron_number_density) * L_ee
        return S_EG

    def polarisation_function(self, k, w, lfc, model="LINDHARD"):
        if model == "LINDHARD":
            return self.lindhard_pol_func(k, w, lfc)
        else:
            raise NotImplementedError(f"Model {model} not recognized. Try LINDHARD")

    def lindhard_pol_func(self, k, w, lfc):

        k_Fe = self.state.fermi_wave_number(self.state.electron_number_density)
        E_Fe = self.state.fermi_energy(self.state.electron_number_density)
        q0 = 0.5 * k / k_Fe
        w0 = 0.25 * DIRAC_CONSTANT * w / (E_Fe * q0)

        z = 0.25 * w / self.state.fermi_energy(self.state.electron_number_density)

        def lindhard_func(x):
            log_arg = abs(((x + 1.0) ** 2 + EPSILON) / ((x - 1.0) ** 2 + EPSILON))
            log_term = 0.5 * np.log(log_arg)
            Re_G = -(x - 0.5 * (x**2 - 1.0) * log_term)
            Im_G = HALF_PI * (1.0 - x**2) * np.heaviside(1.0 - x**2, 1.0)
            return Re_G + 1.0j * Im_G

        G_p = lindhard_func(w0 + q0)
        G_m = lindhard_func(w0 - q0)

        Pi_aa_0_TF = 1.5 * self.state.electron_number_density / E_Fe

        pol_func = Pi_aa_0_TF * (G_p - G_m) / (4.0 * q0)
        return pol_func


def test():
    import matplotlib.pyplot as plt

    lfc = 0
    Te = 20 * eV_TO_K
    rho = 1.0 * g_per_cm3_TO_kg_per_m3
    charge_state = 0.9
    atomic_number = 1
    atomic_mass = 1.0
    k = 1.02e11  # 1/m
    scattering_angle = 30
    E0 = 2.96 * eV_TO_J
    k = 2 * E0 / (DIRAC_CONSTANT * SPEED_OF_LIGHT) * np.sin(scattering_angle / 2)

    omega_array = np.linspace(0, 100, 100) * eV_TO_J  # + 8.5 * eV_TO_J

    dsfs = []
    state = PlasmaState(
        electron_temperature=Te,
        ion_temperature=Te,
        mass_density=rho,
        charge_state=charge_state,
        # frequency=omega,
        atomic_mass=atomic_mass,
        atomic_number=atomic_number,
    )

    for omega in omega_array:
        kernel = FreeFreeDSF(state=state)  # , models=ModelOptions)
        dsf = kernel.get_dsf(k=k, w=omega, lfc=lfc)
        print(dsf)
        dsfs.append(dsf)

    plt.figure()
    plt.plot(omega_array * J_TO_eV, dsfs)
    plt.show()


if __name__ == "__main__":
    test()

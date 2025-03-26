from constants import *
from plasma_state import PlasmaState
from models import ModelOptions
import numpy as np


def effective_coulomb_potential(ionisation, wave_number):
    r"""
    Effective Coulomb potential
    """
    V_aa = (
        -4
        * PI
        * ionisation
        * ELEMENTARY_CHARGE_SQR
        * UNIT_COULOMB_POTENTIAL
        / (wave_number * wave_number)
    )
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
    pol_func = (
        3
        * state.electron_number_density
        / (4 * state.fermi_energy() * q)
        * (func(x_pos) - func(x_neg))
    )
    return pol_func


class FreeFreeDSF:

    def __init__(self, state: PlasmaState, wave_number, models: ModelOptions) -> None:
        self.state = state
        self.wave_number = wave_number

    def numerical_rpa(self):
        beta = 1.0 / (self.state.electron_temperature * BOLTZMANN_CONSTANT)
        prefactor = (
            DIRAC_CONSTANT
            / (2 * PI * self.state.electron_number_density)
            * 2
            / (np.exp(-beta * DIRAC_CONSTANT * self.state.frequency) - 1)
        )
        lfc = 0.0
        effective_potential = effective_coulomb_potential(
            ionisation=self.state.charge_state, wave_number=self.state.wave_number
        )

        polarisation_function = lindhard_rpa(self.state, self.wave_number)
        sf = prefactor * np.imag(
            polarisation_function
            / (1 - polarisation_function * (1 - lfc) * effective_potential)
        )
        return sf

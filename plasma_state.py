import numpy as np
from constants import *


class PlasmaState:

    def __init__(
        self, electron_temperature, mass_density, charge_state, atomic_mass, frequency
    ) -> None:
        self.electron_temperature = electron_temperature
        self.mass_density = mass_density
        self.atomic_mass = atomic_mass * ATOMIC_MASS_UNIT
        self.charge_state = charge_state
        mi = self.atomic_mass
        self.ion_number_density = mass_density / mi
        self.electron_number_density = charge_state * self.ion_number_density
        self.frequency = frequency
        self.wave_number = None

    def initiliase():
        return

    def mean_sphere_radius(self):
        return 1.0 / np.cbrt(FOUR_THIRDS_PI * self.electron_number_density)

    def fermi_energy(self):
        return (
            0.5
            * DIRAC_CONSTANT_SQR
            * np.cbrt(3.0 * PI_SQR * self.electron_number_density) ** 2
            / self.atomic_mass
        )

    def fermi_frequency(self):
        return self.fermi_energy() / DIRAC_CONSTANT

    def fermi_wave_number(self):
        return np.cbrt(3.0 * PI_SQR * self.electron_number_density)

    def fermi_momentum(self):
        return DIRAC_CONSTANT * self.fermi_wave_number()

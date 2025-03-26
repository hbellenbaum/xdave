import numpy as np
from constants import *
# from fermi_integrals import 


class PlasmaState:

    def __init__(
        self,
        electron_temperature,
        ion_temperature,
        mass_density,
        charge_state,
        atomic_mass,
        frequency,
        atomic_number,
        wave_number,
    ) -> None:
        self.electron_temperature = electron_temperature
        self.ion_temperature = ion_temperature
        self.mass_density = mass_density
        self.atomic_mass = atomic_mass * ATOMIC_MASS_UNIT
        self.charge_state = charge_state
        mi = self.atomic_mass
        self.ion_number_density = mass_density / mi
        self.electron_number_density = charge_state * self.ion_number_density
        self.frequency = frequency
        self.wave_number = wave_number
        self.AN = atomic_number
        # self.scattering_number = scattering_number

    def initiliase():
        return

    def mean_sphere_radius(self, number_density):
        return 1.0 / np.cbrt(FOUR_THIRDS_PI * number_density)
    
    def degeneracy_parameter(self, number_density, temperature, mass):
        return number_density * self.thermal_de_broglie_wavelength(temperature, mass) ** 3

    def fermi_energy(self, number_density):
        return (
            0.5
            * DIRAC_CONSTANT_SQR
            * np.cbrt(3.0 * PI_SQR * number_density) ** 2
            / self.atomic_mass
        )

    def fermi_frequency(self):
        return self.fermi_energy() / DIRAC_CONSTANT

    def fermi_wave_number(self, number_density):
        return np.cbrt(3.0 * PI_SQR * number_density)

    def fermi_momentum(self):
        return DIRAC_CONSTANT * self.fermi_wave_number()
    
    def compton_frequency(self, mass):
        return mass * SPEED_OF_LIGHT_SQR / DIRAC_CONSTANT
    
    def debye_screening_length(self, charge, number_density, temperature):
        return np.sqrt(ELECTRIC_CONSTANT * BOLTZMANN_CONSTANT * temperature / number_density) / abs(charge * ELEMENTARY_CHARGE)
    
    def thomas_fermi_screening_length(self, charge, number_density, mass):
        return np.sqrt(ELECTRIC_CONSTANT * self.fermi_energy(number_density) / (1.5 * number_density)) \
           / abs(charge * ELEMENTARY_CHARGE)
    
    def thermal_speed(self, temperature, mass):
        return np.sqrt(BOLTZMANN_CONSTANT * temperature / mass)
    
    def thermal_de_broglie_wavelength(self, temperature, mass):
        return SQRT_TWO_PI * DIRAC_CONSTANT / np.sqrt(mass * BOLTZMANN_CONSTANT * temperature)
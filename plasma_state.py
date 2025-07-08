import numpy as np
from constants import *
from unit_conversions import *

# from fermi_integrals import


def get_rho_T_from_rs_theta(rs, theta, atomic_mass=1.00784):
    """
    Calculates the mass density and electron temperature in cgs for a given rs and theta.
    Inputs:
        - rs: units of Bohr radius
        - theta: non-dimensional temperature
        - atomic_mass: in amu, note that the default is Hydrogen
    Outputs:
        - mass density rho: g/cc
        - temperature T: eV
    """
    fermi_energy = DIRAC_CONSTANT**2 / (2 * ELECTRON_MASS) * (9 * np.pi / (4 * (rs * BOHR_RADIUS) ** 3)) ** (2 / 3)
    Te = fermi_energy * theta / BOLTZMANN_CONSTANT
    T = Te * K_TO_eV

    m_atm = atomic_mass * 1.6605e-27
    rho = 3 / (4 * np.pi * (rs * BOHR_RADIUS) ** 3) * m_atm / 1000

    return rho, T


def get_rs_theta_from_rho_T(rho, T, atomic_mass=1.00784):
    m_atm = atomic_mass * 1.6605e-27
    rs = (m_atm / 1000 * 3 / (4 * np.pi * rho)) ** (1 / 3) * 1 / BOHR_RADIUS

    fermi_energy = DIRAC_CONSTANT**2 / (2 * ELECTRON_MASS) * (9 * np.pi / (4 * (rs * BOHR_RADIUS) ** 3)) ** (2 / 3)
    Te_K = T * eV_TO_K
    theta = Te_K * BOLTZMANN_CONSTANT / fermi_energy
    return rs, theta


class PlasmaState:

    def __init__(
        self,
        electron_temperature,
        ion_temperature,
        mass_density,
        charge_state,
        atomic_mass,
        atomic_number,
    ) -> None:

        # This should all be in SI units
        self.electron_temperature = electron_temperature
        self.ion_temperature = ion_temperature
        if electron_temperature == ion_temperature:
            self.temperature = electron_temperature

        self.mass_density = mass_density
        self.atomic_mass = atomic_mass * ATOMIC_MASS_UNIT
        self.charge_state = charge_state
        mi = self.atomic_mass
        self.ion_number_density = mass_density / mi
        self.electron_number_density = charge_state * self.ion_number_density
        self.atomic_number = atomic_number
        self.free_electron_number_density = charge_state * self.ion_number_density
        self.bound_electron_number_density = (atomic_number - charge_state) * self.ion_number_density
        self.total_electron_number_density = self.free_electron_number_density + self.bound_electron_number_density

    def initiliase():
        return

    def plasma_frequency(self, mass_density, atomic_mass):
        return np.sqrt(4 * np.pi * mass_density * ELEMENTARY_CHARGE_SQR / atomic_mass)

    def mean_sphere_radius(self, number_density):
        return 1.0 / np.cbrt(FOUR_THIRDS_PI * number_density)

    def degeneracy_parameter(self, number_density, temperature, mass):
        return number_density * self.thermal_de_broglie_wavelength(temperature, mass) ** 3

    def fermi_energy(self, number_density, mass):
        return 0.5 * DIRAC_CONSTANT_SQR * np.cbrt(3.0 * PI_SQR * number_density) ** 2 / mass

    def fermi_frequency(self, number_density, mass):
        return self.fermi_energy(number_density, mass) / DIRAC_CONSTANT

    def fermi_wave_number(self, number_density):
        return np.cbrt(3.0 * PI_SQR * number_density)

    def fermi_momentum(self):
        return DIRAC_CONSTANT * self.fermi_wave_number()

    def compton_frequency(self, mass):
        return mass * SPEED_OF_LIGHT_SQR / DIRAC_CONSTANT

    def debye_screening_length(self, charge, number_density, temperature):
        return np.sqrt(ELECTRIC_CONSTANT * BOLTZMANN_CONSTANT * temperature / number_density) / abs(
            charge * ELEMENTARY_CHARGE
        )

    def thomas_fermi_screening_length(self, charge, number_density, mass):
        return np.sqrt(ELECTRIC_CONSTANT * self.fermi_energy(number_density) / (1.5 * number_density)) / abs(
            charge * ELEMENTARY_CHARGE
        )

    def thermal_speed(self, temperature, mass):
        return np.sqrt(BOLTZMANN_CONSTANT * temperature / mass)

    def thermal_de_broglie_wavelength(self, temperature, mass):
        return SQRT_TWO_PI * DIRAC_CONSTANT / np.sqrt(mass * BOLTZMANN_CONSTANT * temperature)

    def chemical_potential(self, temperature, number_density, mass):
        """
        Fit from Ichimaru (2018)

        """
        ne = number_density * per_m3_TO_per_cm3
        Te_eV = temperature * K_TO_eV
        beta = 1.0 / Te_eV

        # Define the constants in the appropiate units
        hbar = 6.582e-16  # eVs
        me = 5.685e-16  # s^2eV/(cm^2)
        # Calculate Ef in eV
        ef = hbar**2 / (2 * me) * (3 * np.pi**2 * ne) ** (2 / 3)

        # ef = calc_ef(ne=ne)
        theta = Te_eV / ef  # calc_theta(ef=ef, Te=Te_eV)

        # Calculate the chemical potential using Ichimaru (2018)
        # Compute the sums individually
        A = 0.25954
        B = 0.072
        c = 0.858
        s1 = -3 / 2 * np.log(theta)
        s2 = np.log(4 / (3 * np.sqrt(np.pi)))
        s3 = (A * theta ** -(c + 1) + B * theta ** -((c + 1) / 2)) / (1 + A * theta**-c)
        bmu = s1 + s2 + s3
        # return the sum and we get \beta * \mu
        mu = bmu / beta
        return mu * eV_TO_J

    def alt_degeneracy_parameter(self, number_density, temperature, mass):
        return BOLTZMANN_CONSTANT * temperature / self.fermi_energy(number_density, mass)

from .constants import *
from .unit_conversions import *

import numpy as np

# from fermi_integrals import
from .fermi_integrals import fdi as xdave_fdi

from scipy.integrate import quad
from scipy.optimize import root_scalar

from dataclasses import dataclass


def get_Z(Z):
    Z_min = np.floor(Z)
    Z_max = np.ceil(Z)
    return Z_min, Z_max


def get_frac(Z, Z_min, Z_max):
    frac_min = 0
    frac_max = 0

    if Z_min != Z_max:
        frac_max = (Z - Z_min) / (Z_max - Z_min)
        frac_min = 1 - frac_max
    else:
        frac_max = 1.0
        frac_min = 0.0
    return frac_min, frac_max


def get_frac_partial(Z, Z_min, Z_max, xlim):
    frac_min = 0
    frac_max = 0
    tot = 1.0 - xlim

    if Z_min != Z_max:
        frac_max = tot * (Z - Z_min) / (Z_max - Z_min)
        frac_min = tot - frac_max
    else:
        frac_max = tot
        frac_min = 0.0
    return frac_min, frac_max


def get_fractions_from_Z(Z):
    Z_min, Z_max = get_Z(Z)
    frac_min, frac_max = get_frac(Z=Z, Z_min=Z_min, Z_max=Z_max)
    return Z_min, Z_max, frac_min, frac_max


def get_fractions_from_Z_partial(Z, x0=0):
    Z_min, Z_max = get_Z(Z)
    frac_min, frac_max = get_frac_partial(Z=Z, Z_min=Z_min, Z_max=Z_max, xlim=x0)
    return Z_min, Z_max, frac_min, frac_max


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


def get_rho_T_from_rs_theta_SI(rs, theta, atomic_mass):
    fermi_energy = DIRAC_CONSTANT**2 / (2 * ELECTRON_MASS) * (9 * np.pi / (4 * (rs * BOHR_RADIUS) ** 3)) ** (2 / 3)
    Te = fermi_energy * theta / BOLTZMANN_CONSTANT
    T = Te  # * K_TO_eV

    m_atm = atomic_mass  #  * 1.6605e-27
    rho = 3 / (4 * np.pi * (rs * BOHR_RADIUS) ** 3) * m_atm
    return rho, T


def get_rs_theta_from_rho_T(rho, T, atomic_mass=1.00784):
    m_atm = atomic_mass * 1.6605e-27
    rs = (m_atm / 1000 * 3 / (4 * np.pi * rho)) ** (1 / 3) * 1 / BOHR_RADIUS

    fermi_energy = DIRAC_CONSTANT**2 / (2 * ELECTRON_MASS) * (9 * np.pi / (4 * (rs * BOHR_RADIUS) ** 3)) ** (2 / 3)
    Te_K = T * eV_TO_K
    theta = Te_K * BOLTZMANN_CONSTANT / fermi_energy
    return rs, theta


def get_rs_theta_from_rho_T_SI(rho, T, atomic_mass):

    if rho == 0:
        print(f"Density is set to zero. This function should not be called.")
    m_atm = atomic_mass  # * 1.6605e-27
    rs = (m_atm * 3 / (4 * np.pi * rho)) ** (1 / 3) * 1 / BOHR_RADIUS

    fermi_energy = DIRAC_CONSTANT**2 / (2 * ELECTRON_MASS) * (9 * np.pi / (4 * (rs * BOHR_RADIUS) ** 3)) ** (2 / 3)
    # Te_K = T * eV_TO_K
    theta = T * BOLTZMANN_CONSTANT / fermi_energy
    return rs, theta


# @dataclass(slots=True)
class PlasmaState:

    def __init__(
        self,
        electron_temperature: float,
        ion_temperature: float,
        mass_density: float,
        charge_state: float,
        atomic_mass: float,
        atomic_number: float,
        binding_energies: np.array,
        electron_number_density: float = None,
        ion_number_density: float = None,
        ion_core_radius: float = BOHR_RADIUS,
        sec_power: float = 2.0,
        csd_core_charge: float = None,
        csd_parameter: float = None,
        srr_sigma: float = None,
    ) -> None:
        # TODO(Hannah): also add option to initialize using rs and theta

        self.initiliased = False

        # This should all be in SI units
        self.electron_temperature = electron_temperature
        self.ion_temperature = ion_temperature
        if electron_temperature == ion_temperature:
            self.temperature = electron_temperature

        self.mass_density = mass_density

        # this is currently the only input that is not in SI units. This needs to change
        # TODO(Hannah)
        self.atomic_mass = atomic_mass * ATOMIC_MASS_UNIT

        self.atomic_number = atomic_number
        self.charge_state = charge_state
        self.ion_charge = charge_state
        mi = self.atomic_mass
        if ion_number_density is not None:
            # assert electron_number_density is not None
            self.ion_number_density = ion_number_density
        else:
            self.ion_number_density = mass_density / mi
        if electron_number_density is not None:
            assert ion_number_density is not None
            self.total_electron_number_density = electron_number_density
            self.free_electron_number_density = charge_state * self.ion_number_density
        else:
            self.free_electron_number_density = charge_state * self.ion_number_density
            self.total_electron_number_density = self.atomic_number * self.ion_number_density

        self.bound_electron_number_density = self.total_electron_number_density - self.free_electron_number_density
        self.binding_energies = binding_energies

        # print(
        #     f"Initializing state at:\n rho={mass_density * kg_per_m3_TO_g_per_cm3} g/cc\n T={electron_temperature * K_TO_eV}eV\n Z={charge_state}\n"
        # )

        self.rs, self.theta = get_rs_theta_from_rho_T_SI(
            rho=self.mass_density, T=self.electron_temperature, atomic_mass=self.atomic_mass
        )
        self.Zb = atomic_number - charge_state

        # some potential parameters that need to be set, these are not physical parameters but instead need to be user inputs
        self.ion_core_radius = ion_core_radius
        self.sec_power = sec_power
        self.csd_core_charge = csd_core_charge
        self.csd_parameter = csd_parameter
        self.srr_sigma = srr_sigma

    def initiliase(self):
        self.kF = self.fermi_wave_number(number_density=self.total_electron_number_density)
        self.EF = self.fermi_energy(self.total_electron_number_density, self.atomic_mass)
        self.mue = self.chemical_potential_ichimaru(
            temperature=self.electron_temperature,
            number_density=self.total_electron_number_density,
            mass=self.atomic_mass,
        )
        self.kappae = self.screening_length(
            ELECTRON_MASS, 1, self.electron_temperature, self.free_electron_number_density
        )
        self.omegaF = self.fermi_frequency(self.total_electron_number_density, self.atomic_mass)
        self.initiliased = True

    def fermi_temperature(self, mass, number_density):
        TF = DIRAC_CONSTANT**2 / (mass * BOLTZMANN_CONSTANT) * 0.5 * (3 * PI_SQR) ** (2 / 3) * number_density**1.5
        TF = self.fermi_energy(number_density, mass) / BOLTZMANN_CONSTANT
        return TF

    def plasma_frequency(self, charge, number_density, mass):
        # return np.sqrt(4 * np.pi * mass_density * ELEMENTARY_CHARGE_SQR / (atomic_mass * VACUUM_PERMITTIVITY))
        return np.sqrt(number_density / (mass * ELECTRIC_CONSTANT)) * abs(charge * ELEMENTARY_CHARGE)

    def mean_sphere_radius(self, number_density):
        return 1.0 / np.cbrt(FOUR_THIRDS_PI * number_density)

    def degeneracy_parameter(self, number_density, temperature, mass):
        return number_density * self.thermal_de_broglie_wavelength(temperature, mass) ** 3

    def fermi_energy(self, number_density, mass):
        return 0.5 * DIRAC_CONSTANT**2 * (3.0 * PI**2 * number_density) ** (2 / 3) / mass

    def fermi_frequency(self, number_density, mass):
        return self.fermi_energy(number_density, mass) / DIRAC_CONSTANT

    def fermi_wave_number(self, number_density):
        return np.cbrt(3.0 * PI_SQR * number_density)

    def fermi_momentum(self):
        return DIRAC_CONSTANT * self.fermi_wave_number()

    def compton_frequency(self, mass):
        return mass * SPEED_OF_LIGHT_SQR / DIRAC_CONSTANT

    def screening_length(self, mass, charge, temperature, number_density):
        eta = self.chemical_potential_ichimaru(temperature=temperature, number_density=number_density, mass=mass)
        beta = 1 / (BOLTZMANN_CONSTANT * temperature)
        f = xdave_fdi(j=-0.5, eta=eta * beta, normalize=True)
        degeneracy_parameter = (
            number_density * (TWO_PI * DIRAC_CONSTANT_SQR / (mass * BOLTZMANN_CONSTANT * temperature)) ** 1.5
        )
        kappa = np.sqrt(
            2.0
            * charge**2
            * number_density
            * ELEMENTARY_CHARGE**2
            * f
            / (VACUUM_PERMITTIVITY * BOLTZMANN_CONSTANT * temperature * degeneracy_parameter)
        )
        return kappa

    def debye_screening_length(self, charge, number_density, temperature):
        return np.sqrt(
            number_density * charge * ELEMENTARY_CHARGE**2 / (VACUUM_PERMITTIVITY * BOLTZMANN_CONSTANT * temperature)
        )
        # return np.sqrt(ELECTRIC_CONSTANT * BOLTZMANN_CONSTANT * temperature / number_density) / abs(
        #     charge * ELEMENTARY_CHARGE
        # )

    def thomas_fermi_screening_length(self, charge, number_density, mass):
        return np.sqrt(ELECTRIC_CONSTANT * self.fermi_energy(number_density) / (1.5 * number_density)) / abs(
            charge * ELEMENTARY_CHARGE
        )

    def thermal_speed(self, temperature, mass):
        return np.sqrt(BOLTZMANN_CONSTANT * temperature / mass)

    def thermal_de_broglie_wavelength(self, temperature, mass):
        return SQRT_TWO_PI * DIRAC_CONSTANT / np.sqrt(mass * BOLTZMANN_CONSTANT * temperature)

    def chemical_potential(self, temperature, number_density, mass):

        def f(mu_tilde, T_tilde):
            integrand = lambda x: x**0.5 / (np.exp((x - mu_tilde) / T_tilde) + 1)
            res, _ = quad(integrand, 0.0, np.inf, limit=100)
            return res - 2 / 3

        def solve_mu(T_tilde):
            # result = minimize()
            result = root_scalar(lambda mu: f(mu, T_tilde), bracket=[-10, 20], method="brentq")  # ,
            return result.root

        T_erg = temperature * K_TO_erg
        EF_erg = self.fermi_energy(number_density, mass) * J_TO_erg

        Theta = T_erg / EF_erg  # BOLTZMANN_CONSTANT *

        mu_erg = EF_erg * solve_mu(T_tilde=Theta)  # multiply by E_F to remove normalization

        # ideal fermi gas (theta < 0.2)
        mu_erg_Low = EF_erg * (1 - (PI * Theta) ** 2 / 12 - (PI * Theta) ** 4 / 80)
        # print("Low T", mu_erg_Low)

        # high temperature expansion
        mu_erg_High = T_erg * log(4 / 3 / sqrt(PI) / sqrt(Theta**3))
        # print("High T", mu_erg_High)

        return mu_erg, mu_erg_High, mu_erg_Low

    def chemical_potential_classical(self, temperature, number_density, mass):
        Tq = DIRAC_CONSTANT**2 / (mass * BOLTZMANN_CONSTANT) * 2 * PI * (number_density / 2) ** 2 / 3
        mu_class = -3 / 2 * BOLTZMANN_CONSTANT * temperature * np.log(temperature / Tq)
        return mu_class

    def reduced_chemical_potential_tobias(self, theta):
        mu = 0

        C = 0.752252778063675

        mu = theta * np.log(C / (theta**1.5)) + theta * np.log(1 + C / (theta**1.5) / (2**1.5))

        if theta < 1.36:
            a1 = 0.016
            a2 = -0.957
            a3 = -0.293
            a4 = 0.209
            mu = 1.0 + a1 * theta + a2 * (theta**2) + a3 * (theta**3) + a4 * (theta**4)

        # print("chem_pot=", mu)

        ############# chemical potential in units of k_BT ##############
        eta = mu / theta
        return eta

    def chemical_potential_ichimaru(self, temperature, number_density, mass):
        # from scipy.special import gamma

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
        ef = hbar**2 / (2 * me) * (3 * np.pi**2 * ne) ** (2 / 3)  # eV

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

    def electron_electron_coupling_parameter(self, number_density, temperature):
        radius = (3 / (4 * PI * number_density)) ** (1 / 3)
        Gamma_ee = ELEMENTARY_CHARGE**2 / (4 * PI * VACUUM_PERMITTIVITY * radius * BOLTZMANN_CONSTANT * temperature)
        return Gamma_ee

    def coupling_parameter(self, Za, beta, da):
        return Za**2 * ELEMENTARY_CHARGE**2 * beta / (4 * PI * VACUUM_PERMITTIVITY * da)

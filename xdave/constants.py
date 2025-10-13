r"""
Provides constants used in the code
"""

import sys
from numpy import log

from math import pi as PI
from numpy import cbrt, sqrt, log as ln

# from constants.maths import PI_SQR, FOUR_THIRDS_PI, TWO_PI, FOUR_PI, NINE_PI

####################
# CODE
####################

EPSILON = sys.float_info.epsilon
TINY = sys.float_info.min
HUGE = sys.float_info.max
LN_TINY = log(TINY)
LN_HUGE = log(HUGE)


####################
# MATHS
####################


THIRD_PI = PI / 3.0
HALF_PI = 0.5 * PI
FOUR_THIRDS_PI = 4.0 * THIRD_PI
TWO_PI = 2.0 * PI
THREE_PI = 3.0 * PI
FOUR_PI = 2.0 * TWO_PI
FIVE_PI = 5.0 * PI
NINE_PI = 3.0 * THREE_PI
PI_SQR = PI * PI
PI_CUB = PI_SQR * PI
TWO_PI_SQR = 2.0 * PI_SQR
FOUR_PI_SQR = 4.0 * PI_SQR
CBRT_PI = cbrt(PI)
SQRT_HALF_PI = sqrt(HALF_PI)
SQRT_PI = sqrt(PI)
SQRT_TWO_PI = sqrt(TWO_PI)

ZERO_COMPLEX = complex(0.0)
ONE_COMPLEX = complex(1.0)
I_COMPLEX = complex(0.0, 1.0)
I_PI = complex(0.0, PI)
I_TWO_PI = complex(0.0, TWO_PI)

SQRT_TWO = sqrt(2.0)
SQRT_THREE = sqrt(3.0)
SQRT_FIVE = sqrt(5.0)
SQRT_SIX = SQRT_TWO * SQRT_THREE
SQRT_SEVEN = sqrt(7.0)
SQRT_EIGHT = 2.0 * SQRT_TWO
SQRT_TEN = SQRT_TWO * SQRT_FIVE
CBRT_TWO = cbrt(2.0)
CBRT_THREE = cbrt(3.0)

LN_TWO = ln(2.0)
LN_TEN = ln(10.0)

EULER_MASCHERONI = 0.57721566490153286060651209008240243104215933593992
GOLDEN_RATIO = 0.5 * (1.0 + SQRT_FIVE)


####################
# PHYSICS
####################


# Speed of light in units of m/s.
SPEED_OF_LIGHT = 2.99792458e08

# Magnetic constant (permeability of vacuum) in units of H/m.
MAGNETIC_CONSTANT = FOUR_PI * 1.0e-07

# Planck's constant in units of J.s.
PLANCK_CONSTANT = 6.62607015e-34

# Elementary charge in units of C.
ELEMENTARY_CHARGE = 1.602176634e-19

# Boltzmann's constant in units of J/K.
BOLTZMANN_CONSTANT = 1.380649e-23

# Avogadro constant in units of 1/mol.
AVOGADRO_CONSTANT = 6.02214076e23

# Charges of electron and proton in units of the elementary charge e.
ELECTRON_CHARGE_OVER_E = -1.0e00
PROTON_CHARGE_OVER_E = +1.0e00

# Standard of temperature and pressure according to the NIST definition (20 degrees C and 1 atm) in units of K and Pa.
NORMAL_TEMPERATURE = 2.9315e02
NORMAL_PRESSURE = 1.01325e05

# Standard of temperature and pressure according to the IUPAC definition (0 degrees C and 1 bar) in units of K and Pa.
STANDARD_TEMPERATURE = 2.7315e02
STANDARD_PRESSURE = 1.0e05

# Mass of the electron in units of kg.
ELECTRON_MASS = 9.10938356e-31

# Mass of the proton in units of kg.
PROTON_MASS = 1.672621898e-27

# Atomic mass unit in units of kg.
ATOMIC_MASS_UNIT = 1.660539040e-27

# Vacuum permittivity in units of F/m
VACUUM_PERMITTIVITY = 8.8541878188e-12

# Squares of important unit_conversions, which get used in many places.
SPEED_OF_LIGHT_SQR = SPEED_OF_LIGHT * SPEED_OF_LIGHT
PLANCK_CONSTANT_SQR = PLANCK_CONSTANT * PLANCK_CONSTANT
ELEMENTARY_CHARGE_SQR = ELEMENTARY_CHARGE * ELEMENTARY_CHARGE
BOLTZMANN_CONSTANT_SQR = BOLTZMANN_CONSTANT * BOLTZMANN_CONSTANT
ELECTRON_MASS_SQR = ELECTRON_MASS * ELECTRON_MASS

# Electric constant (permittivity of vacuum) in units of F/m.
ELECTRIC_CONSTANT = 1.0 / (MAGNETIC_CONSTANT * SPEED_OF_LIGHT**2)

# Coulomb's constant in units of kg.m^3/s^4/A^2.
COULOMB_CONSTANT = 1.0 / (FOUR_PI * ELECTRIC_CONSTANT)
COULOMB_CONSTANT_SQR = COULOMB_CONSTANT * COULOMB_CONSTANT

# Dirac's constant (reduced Planck's constant) in units of J.s.
DIRAC_CONSTANT = PLANCK_CONSTANT / TWO_PI
DIRAC_CONSTANT_SQR = DIRAC_CONSTANT * DIRAC_CONSTANT
DIRAC_CONSTANT_CUB = DIRAC_CONSTANT * DIRAC_CONSTANT_SQR

# Universal gas constant in units of J/mol/K.
UNIVERSAL_GAS_CONSTANT = BOLTZMANN_CONSTANT * AVOGADRO_CONSTANT

# Fine-structure constant (dimensionless).
FINE_STRUCTURE_CONSTANT = ELEMENTARY_CHARGE_SQR * COULOMB_CONSTANT / (DIRAC_CONSTANT * SPEED_OF_LIGHT)
FINE_STRUCTURE_CONSTANT_SQR = FINE_STRUCTURE_CONSTANT * FINE_STRUCTURE_CONSTANT

# Bohr radius (of an electron) in units of m.
BOHR_RADIUS = DIRAC_CONSTANT / (FINE_STRUCTURE_CONSTANT * ELECTRON_MASS * SPEED_OF_LIGHT)

# Rdberg constant in units of 1/m.
RYDBERG_CONSTANT = 0.5 * FINE_STRUCTURE_CONSTANT_SQR * ELECTRON_MASS * SPEED_OF_LIGHT / PLANCK_CONSTANT

# Hartree energy in units of J.
RYDBERG_ENERGY = RYDBERG_CONSTANT * PLANCK_CONSTANT * SPEED_OF_LIGHT

# Hartree energy in units of J.
HARTREE_ENERGY = 2.0 * RYDBERG_ENERGY

# Classical electron radius in units of m.
CLASSICAL_ELECTRON_RADIUS = ELEMENTARY_CHARGE_SQR * COULOMB_CONSTANT / (ELECTRON_MASS * SPEED_OF_LIGHT_SQR)

# Thomson cross section in units of m^2.
THOMSON_CROSS_SECTION = 2.0 * FOUR_THIRDS_PI * CLASSICAL_ELECTRON_RADIUS * CLASSICAL_ELECTRON_RADIUS

# Stefan-Boltzmann constant in units of W/m^2/K^4.
STEFAN_BOLTZMANN_CONSTANT = (
    PI_SQR
    * BOLTZMANN_CONSTANT_SQR
    * BOLTZMANN_CONSTANT
    / (60.0 * DIRAC_CONSTANT * DIRAC_CONSTANT_SQR * SPEED_OF_LIGHT_SQR)
)


# Potential energy of a pair of elementary charges in units of J at a separation of 1 m.
UNIT_COULOMB_POTENTIAL = ELEMENTARY_CHARGE_SQR * COULOMB_CONSTANT
UNIT_COULOMB_POTENTIAL_SQR = UNIT_COULOMB_POTENTIAL * UNIT_COULOMB_POTENTIAL

# The constant \alpha = \cbrt(4/9\pi) is encountered often in function related to solid state physics.
SOLID_STATE_ALPHA = cbrt(4.0 / NINE_PI)


# COMPUTATIONAL CONSTANTS
EPSILON = sys.float_info.epsilon

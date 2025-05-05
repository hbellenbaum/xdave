from plasma_state import PlasmaState

from constants import *
from unit_conversions import *
from fermi_integrals import fermi_integral

import numpy as np


def get_ipd(state: PlasmaState, model):
    Zi = state.charge_state
    ne = state.electron_number_density
    ni = state.ion_number_density
    Te = state.electron_temperature
    Ti = state.ion_temperature

    if model == "STEWART_PYATT":
        return ipd_stewart_pyatt(Zi=Zi, ne=ne, ni=ni, Te=Te, Ti=Ti)
    elif model == "DEBYE_HUCKEL":
        return ipd_debye_hueckel(Zi=Zi, ne=ne, ni=ni, Te=Te, Ti=Ti)
    elif model == "ECKER_KROLL":
        return ipd_ecker_kroell(Zi=Zi, ne=ne, ni=ni, Te=Te, Ti=Ti)
    elif model == "ION_SPHERE":
        return ipd_ion_sphere(Zi=Zi, ne=ne, ni=ni)
    else:
        raise NotImplementedError(f"IPD model {model} is not recognised.")


def chem_potential_fit(T, n_e):
    """
    Fit for the chemical potential [Gregori (2003)]
    # TODO(Hannah): I should compare this to the Utsumi fit
    """
    A = 0.25945
    B = 0.072
    b = 0.858

    E_f = DIRAC_CONSTANT**2 / (2 * ELECTRON_MASS) * (3 * np.pi**2 * n_e) ** (2 / 3)
    Theta = BOLTZMANN_CONSTANT * T / E_f
    f = (
        (-3 / 2 * np.log(Theta))
        + (np.log(4 / (3 * np.sqrt(np.pi))))
        + ((A * Theta ** (-b - 1) + B * Theta ** (-(b + 1) / 2)) / (1 + A * Theta ** (-b)))
    )
    return f * BOLTZMANN_CONSTANT * T


def inverse_electron_screening_length_classical(ne, ni, Zi, Te):
    kappa_sqr = ELEMENTARY_CHARGE_SQR / (ELECTRIC_CONSTANT * BOLTZMANN_CONSTANT * Te) * (Zi**2 * ni + ne)
    return np.sqrt(kappa_sqr)


def inverse_electron_screening_length(ne, Te):
    """
    Inverse screening length for arbitrary degeneracy [Baggott (2017)]
    """

    chem_pot = chem_potential_fit(Te, ne)
    beta = 1 / (1 * BOLTZMANN_CONSTANT * Te)

    fermi_integral = fermi_integral(phi=(chem_pot * beta), j=-0.5, normalise=False)

    EF = 0.5 * DIRAC_CONSTANT_SQR * np.cbrt(3.0 * PI_SQR * ne) ** 2 / ELECTRON_MASS

    kappa_sqr = (
        12
        * PI ** (5 / 2)
        * ELEMENTARY_CHARGE_SQR
        / (4 * PI * ELECTRIC_CONSTANT)
        * ne
        * beta
        * fermi_integral
        / (beta * EF) ** 1.5
    )

    return np.sqrt(kappa_sqr)


def ipd_debye_hueckel(Zi, ne, ni, Te, Ti):

    kappa_C = inverse_electron_screening_length_classical(ne, ni, Zi, Te)

    delta_ipd = kappa_C * (Zi + 1) * ELEMENTARY_CHARGE_SQR / (4 * PI * ELECTRIC_CONSTANT)

    return delta_ipd * J_TO_eV


def ipd_ion_sphere(Zi, ne, ni):
    """
    Just the dumb ion shere radius, should include the correction by [Zimmerman (1980)] at some point
    """

    kappa = (4 * PI * ni / 3) ** (1 / 3)

    delta_ipd = -Zi * ELEMENTARY_CHARGE_SQR * COULOMB_CONSTANT * kappa

    return delta_ipd * J_TO_eV


def ipd_stewart_pyatt(Zi, ne, ni, Te, Ti):
    """
    Corrected Stewart-Pyatt model [Roepke (2019)]
    """

    r_IS = (3 * Zi / (4 * PI * ne)) ** (1 / 3)
    kappa = inverse_electron_screening_length(ne, Te)
    s = 1 / (r_IS * kappa)
    factor = (1 + s**3) ** (2 / 3) - s**2
    ipd_shift = 3 / 2 * (Zi + 1) * ELEMENTARY_CHARGE_SQR / (4 * PI * ELECTRIC_CONSTANT * r_IS) * factor

    return ipd_shift * J_TO_eV


def ipd_ecker_kroell(Zi, ne, ni, Te, Ti):
    """
    Original Ecker-Kroell model (does not appear to work correctly)
    """

    lambda_Di = np.sqrt(ELECTRIC_CONSTANT * BOLTZMANN_CONSTANT * Ti / (ne * ELEMENTARY_CHARGE**2))

    R_0 = (3 / (4 * np.pi * ni)) ** (1 / 3)

    # critical density

    n_c = (3 / (4 * np.pi)) * (4 * np.pi * 1 * ELECTRIC_CONSTANT * BOLTZMANN_CONSTANT * Te / ELEMENTARY_CHARGE**2) ** 3

    # Ecker-Kroells constant
    C = 2.2 * np.sqrt(ELEMENTARY_CHARGE**2 / (BOLTZMANN_CONSTANT * Te)) * n_c ** (1 / 6)

    ipd_c1 = -1 * ELEMENTARY_CHARGE**2 / (ELECTRIC_CONSTANT * lambda_Di) * Zi
    ipd_c2 = -C * ELEMENTARY_CHARGE**2 / (ELECTRIC_CONSTANT * R_0) * Zi

    # The ionization potential depression energy shift
    ipd_shift = np.where(ni <= n_c, ipd_c1, ipd_c2)

    return ipd_shift * J_TO_eV

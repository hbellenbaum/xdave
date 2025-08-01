from plasma_state import PlasmaState

from constants import *
from unit_conversions import *
# from fermi_integrals import fermi_integral

# NOTE(TG): Would recommend replacing this with Antia fits. At high densities fdi becomes
#           extremely slow, if it even is able to reach an answer.
from plasmapy.formulary.mathematics import Fermi_integral as fdi

import numpy as np


def get_ipd(state: PlasmaState, model):
    Zi = state.charge_state
    ne = state.electron_number_density
    ni = state.ion_number_density
    Te = state.electron_temperature
    Ti = state.ion_temperature
    Zn = state.atomic_number

    if model == "STEWART_PYATT":
        return ipd_stewart_pyatt(Zi=Zi, ne=ne, ni=ni, Te=Te, Ti=Ti)
    elif model == "DEBYE_HUCKEL":
        return ipd_debye_hueckel(Zi=Zi, ne=ne, ni=ni, Te=Te, Ti=Ti)
    elif model == "ECKER_KROLL":
        return ipd_ecker_kroell(Zi=Zi, ne=ne, ni=ni, Te=Te, Ti=Ti, Zn=Zn)
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


def inverse_electron_screening_length_sqr_classical(ne, Te):
    kappa_sqr = ELEMENTARY_CHARGE_SQR / (ELECTRIC_CONSTANT * BOLTZMANN_CONSTANT * Te) * ne
    return kappa_sqr


def inverse_electron_screening_length_sqr(ne, Te):
    """
    Thomas-Fermi electron inverse screening length squared
    """
    # NOTE(TG): Original formula looked like it came from [Roepke (2019)], but the final form of
    #           Eq. (19) is incorrect - going through the maths it contains an extra factor of 4pi.
    #           You can tell it's correct both by derivation and the fact that it doesn't produce
    #           the electron Debye length in the weakly coupled limit. Also, his Fermi-Dirac integral
    #           is a normalised one, not an unnorm'd one as was used here. Point is, I've corrected it.

    chem_pot = chem_potential_fit(Te, ne)
    beta = 1 / (BOLTZMANN_CONSTANT * Te)

    EF = 0.5 * DIRAC_CONSTANT_SQR * np.cbrt(3.0 * PI_SQR * ne) ** 2 / ELECTRON_MASS

    Ip0p5 = (beta * EF)**1.5 * 2/3
    Im0p5 = fdi(x=chem_pot * beta, j=-1/2).real * SQRT_PI # fdi gives norm'd FDIs.

    kappa_sqr = ELEMENTARY_CHARGE_SQR/ELECTRIC_CONSTANT * ne * beta * 0.5*Im0p5/Ip0p5

    return kappa_sqr


def ipd_debye_hueckel(Zi, ne, ni, Te, Ti):

    # NOTE(TG): Setting up some stuff here in case code eventually handles multiple charge states
    Zmean = ne/ni  # Zmean = ne/ni_tot
    Zstar = Zi**2 / Zmean  # Zstar = np.sum(Zi**2 * csd) / Zmean  # csd = charge state distribution

    kappa_C_sqr = inverse_electron_screening_length_sqr_classical(ne, Te)
    kappa_C_sqr *= (1 + Zstar)
    kappa_C = np.sqrt(kappa_C_sqr)

    delta_ipd = kappa_C * (Zi + 1) * ELEMENTARY_CHARGE_SQR / (4 * PI * ELECTRIC_CONSTANT)

    return delta_ipd * J_TO_eV


def ipd_ion_sphere(Zi, ne, ni):
    """
    Just the dumb ion shere radius, should include the correction by [Zimmerman (1980)] at some point
    """
    # NOTE(TG): Ion sphere model chosen should match the SP model, since it's what people will expect.
    #           Also added the missing (Z+1) dependence in the IPD and the ion sphere radius

    kappa = ((FOUR_PI * ne) / (3 * (Zi + 1)))**(1/3)
    delta_ipd = 1.5 * (Zi + 1) * ELEMENTARY_CHARGE_SQR * COULOMB_CONSTANT * kappa

    return delta_ipd * J_TO_eV


def ipd_stewart_pyatt(Zi, ne, ni, Te, Ti):
    """
    Corrected Stewart-Pyatt model [Roepke (2019)]
    """
    # NOTE(TG): Added in the ion contribution to the IPD here, which was missing and necessary for SP.
    #           Note that SP assumes the electrons and ions are in =ium. More advanced model like Crowley needed
    #           to treat two-temp system with arbitrary temperature (otherwise ion screening length explodes!).

    # Ion sphere radius depends on the charge state you're changing into
    r_IS = (3 * (Zi + 1) / (FOUR_PI * ne)) ** (1 / 3)

    # NOTE(TG): Setting up some stuff here in case code eventually handles multiple charge states
    Zmean = ne/ni  # Zmean = ne/ni_tot
    Zstar = Zi**2 / Zmean  # Zstar = np.sum(Zi**2 * csd) / Zmean  # csd = charge state distribution


    kappa_sqr = inverse_electron_screening_length_sqr(ne, Te)
    kappa_sqr *= 1 + Zstar
    kappa = np.sqrt(kappa_sqr)
    
    s = 1 / (r_IS * kappa)
    factor = (1 + s**3) ** (2 / 3) - s**2
    ipd_shift = 3 / 2 * (Zi + 1) * ELEMENTARY_CHARGE_SQR * COULOMB_CONSTANT / r_IS * factor

    return ipd_shift * J_TO_eV


def ipd_ecker_kroell(Zi, ne, ni, Te, Ti, Zn):
    """
    Original Ecker-Kroell model (does not appear to work correctly)
    """

    # NOTE(TG): Setting up some stuff here in case code eventually handles multiple charge states
    Zmean = ne/ni  # Zmean = ne/ni_tot
    Zstar = Zi**2 / Zmean  # Zstar = np.sum(Zi**2 * csd) / Zmean  # csd = charge state distribution

    kappa_C_sqr = inverse_electron_screening_length_sqr_classical(ne, Te)
    kappa_C_sqr *= (1 + Zstar)
    kappa_D = np.sqrt(kappa_C_sqr)

    # NOTE(TG): Missing the electron term in the EK radius, which was probably causing the issue
    inv_R_EK = ((FOUR_PI * (ne + ni)) / 3)**(1/3)

    # Critical density
    # NOTE(TG): if multiple species included, there needs to be a different nc for each species...
    n_c = 3/FOUR_PI * ( BOLTZMANN_CONSTANT * Te/COULOMB_CONSTANT/ELEMENTARY_CHARGE_SQR/Zn**2  )**3

    # Ecker-Kroells constant
    # C_EK = 2.2 * np.sqrt(ELEMENTARY_CHARGE**2 / (BOLTZMANN_CONSTANT * Te)) * n_c ** (1 / 6)
    C_EK = 1 # Just set to one since this tends to agree better with experiments

    common_const = (Zi + 1) * ELEMENTARY_CHARGE_SQR * COULOMB_CONSTANT

    # The ionization potential depression energy shift
    # NOTE(TG): n_c is tested against total particle density in the system, not just ions
    ipd_shift = common_const * np.where(ni + ne <= n_c, kappa_D, C_EK * inv_R_EK)

    return ipd_shift * J_TO_eV

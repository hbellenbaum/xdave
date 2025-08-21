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
    ne = state.free_electron_number_density
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
    elif model == "CROWLEY":
        return ipd_crowley(Zi=Zi, ne=ne, ni=ni, Te=Te, Ti=Ti, ForceConst=0.9)
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
    """
    Debye inverse screening length squared
    """
    kappa_sqr = ELEMENTARY_CHARGE_SQR / (ELECTRIC_CONSTANT * BOLTZMANN_CONSTANT * Te) * ne
    return kappa_sqr


def inverse_electron_screening_length_sqr(ne, Te):
    """
    Thomas-Fermi electron inverse screening length squared
    """
    # NOTE(TG): Original formula looked like it came from [Roepke (2019)], but the final form of
    #           Eq. (19) is incorrect - going through the maths it contains an extra factor of 4pi.
    #           You can tell it's incorrect both by derivation and the fact that it doesn't produce
    #           the electron Debye length in the weakly coupled limit. Also, his Fermi-Dirac integral
    #           is a normalised one, not an unnorm'd one as was used here. Point is, I've corrected it.

    chem_pot = chem_potential_fit(Te, ne)
    beta = 1 / (BOLTZMANN_CONSTANT * Te)

    EF = 0.5 * DIRAC_CONSTANT_SQR * np.cbrt(3.0 * PI_SQR * ne) ** 2 / ELECTRON_MASS

    Ip0p5 = (beta * EF) ** 1.5 * 2 / 3
    Im0p5 = fdi(x=chem_pot * beta, j=-1 / 2).real * SQRT_PI  # fdi gives norm'd FDIs.

    kappa_sqr = ELEMENTARY_CHARGE_SQR / ELECTRIC_CONSTANT * ne * beta * 0.5 * Im0p5 / Ip0p5

    return kappa_sqr


def ipd_debye_hueckel(Zi, ne, ni, Te, Ti):

    # NOTE(TG): Setting up some stuff here in case code eventually handles multiple charge states
    Zmean = ne / ni  # Zmean = ne/ni_tot
    Zstar = Zi**2 / Zmean  # Zstar = np.sum(Zi**2 * csd) / Zmean  # csd = charge state distribution

    kappa_C_sqr = inverse_electron_screening_length_sqr_classical(ne, Te)
    kappa_C_sqr *= 1 + Zstar
    kappa_C = np.sqrt(kappa_C_sqr)

    delta_ipd = kappa_C * (Zi + 1) * UNIT_COULOMB_POTENTIAL

    return delta_ipd


def ipd_ion_sphere(Zi, ne, ni):
    """
    Ion sphere IPD
    """
    # NOTE(TG): Ion sphere model chosen should match the SP model, since it's what people will expect.
    #           Also added the missing (Z+1) dependence in the IPD and the ion sphere radius

    kappa = ((FOUR_PI * ne) / (3 * (Zi + 1))) ** (1 / 3)
    delta_ipd = 1.5 * (Zi + 1) * UNIT_COULOMB_POTENTIAL * kappa

    return delta_ipd


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
    Zmean = ne / ni  # Zmean = ne/ni_tot
    Zstar = Zi**2 / Zmean  # Zstar = np.sum(Zi**2 * csd) / Zmean  # csd = charge state distribution

    kappa_sqr = inverse_electron_screening_length_sqr(ne, Te)
    kappa_sqr *= 1 + Zstar
    kappa = np.sqrt(kappa_sqr)

    s = 1 / (r_IS * kappa)
    factor = (1 + s**3) ** (2 / 3) - s**2
    ipd_shift = 3 / 2 * (Zi + 1) * UNIT_COULOMB_POTENTIAL / r_IS * factor

    return ipd_shift


def ipd_ecker_kroell(Zi, ne, ni, Te, Ti, Zn):
    """
    Original Ecker-Kroell model
    """

    # NOTE(TG): Setting up some stuff here in case code eventually handles multiple charge states
    Zmean = ne / ni  # Zmean = ne/ni_tot
    Zstar = Zi**2 / Zmean  # Zstar = np.sum(Zi**2 * csd) / Zmean  # csd = charge state distribution

    kappa_C_sqr = inverse_electron_screening_length_sqr_classical(ne, Te)
    kappa_C_sqr *= 1 + Zstar
    kappa_D = np.sqrt(kappa_C_sqr)

    # NOTE(TG): Missing the electron term in the EK radius, which was probably causing the issue
    inv_R_EK = ((FOUR_PI * (ne + ni)) / 3) ** (1 / 3)

    # Critical density
    # NOTE(TG): if multiple species included, there needs to be a different nc for each species...
    n_c = 3 / FOUR_PI * (BOLTZMANN_CONSTANT * Te / UNIT_COULOMB_POTENTIAL / Zn**2) ** 3

    # Ecker-Kroells constant
    # C_EK = 2.2 * np.sqrt(ELEMENTARY_CHARGE**2 / (BOLTZMANN_CONSTANT * Te)) * n_c ** (1 / 6)
    C_EK = 1  # Just set to one since this tends to agree better with experiments

    common_const = (Zi + 1) * UNIT_COULOMB_POTENTIAL

    # The ionization potential depression energy shift
    # NOTE(TG): n_c is tested against total particle density in the system, not just ions
    ipd_shift = common_const * np.where(ni + ne <= n_c, kappa_D, C_EK * inv_R_EK)

    return ipd_shift


# Here on a a bunch of functions to do the Crowley model.


###################################################################################
# Functions for solving the cubic polynomial
def f_term(a, Y):
    a2 = a * a
    a3 = a * a2
    Y3 = Y**3
    Y6 = Y3 * Y3

    # Need to make square root complex so that a solution always come out
    f = -2 * a3 * Y3 + np.sqrt(-4 * a3 * Y3 - 3 * a2 * Y6 + 6 * a * Y3 + 4 * Y6 + 1 + 0j) + 3 * a * Y3 + 1
    f *= 0.5
    return f ** (1 / 3)


def X_term(a, Y):
    f_aY = f_term(a, Y)

    aY = a * Y
    B = Y**2 * (1 - a**2) / f_aY

    sol1 = f_aY - B - aY
    if not (np.any(np.abs(sol1.imag) > 1e-12) or np.any(sol1.real < 0)):
        return sol1.real

    sol2 = 0.5 * ((1 - SQRT_THREE * 1j) * f_aY + (1 + SQRT_THREE * 1j) * B) - aY
    if not (np.any(np.abs(sol2.imag) > 1e-12) or np.any(sol2.real < 0)):
        return sol2.real

    sol3 = 0.5 * (-(1 + SQRT_THREE * 1j) * f_aY + (1 - SQRT_THREE * 1j) * B) - aY
    if not (np.any(np.abs(sol3.imag) > 1e-12) or np.any(sol3.real < 0)):
        return sol3.real

    print("Uhoh! No real, positive solutions for X found!")
    print("X1:", sol1)
    print("X2:", sol2)
    print("X3:", sol3)


# =============================================================================
# Static continuum lowering pieces
def h(L):
    return (1 + L) ** (2 / 3) - 1


def g(L):
    return (0.6 * ((1 + L) ** (5 / 3) - 1) - L) / L


def ipd_crowley(Zi, ne, ni, Te, Ti, ForceConst):
    """
    Crowley IPD model. For now, contains shift from the Pauli blocking term
    """

    # NOTE(TG): Pieces needed for charge state distributions
    # Normalise charge state distribution in case it isn't already
    # csd /= np.sum(csd)

    # # Get number densities for each ion species
    ni_tot = ni  # ni = ni_tot * csd

    # # Get electron number density
    # ne = np.sum(ni * zi)

    kTe = BOLTZMANN_CONSTANT * Te
    kTi = BOLTZMANN_CONSTANT * Ti
    beta_e = 1 / kTe

    e2_epsilon0 = ELEMENTARY_CHARGE_SQR / ELECTRIC_CONSTANT

    # Fermi-Dirac integrals and chemical potential (divided by kTe)

    chem_pot = chem_potential_fit(Te, ne)
    eta_e = chem_pot * beta_e
    EF = 0.5 * DIRAC_CONSTANT_SQR * np.cbrt(3.0 * PI_SQR * ne) ** 2 / ELECTRON_MASS
    Ip0p5 = (beta_e * EF) ** 1.5 * 2 / 3
    Im0p5 = fdi(x=eta_e, j=-1 / 2).real * SQRT_PI  # fdi gives norm'd FDIs.

    # Plasma (or perturber) effective charge
    Zmean = ne / ni  # Zmean = ne/ni_tot
    Zstar = Zi**2 / Zmean  # Zstar = np.sum(Zi**2 * csd) / Zmean  # csd = charge state distribution

    # Plasma screening length - ion's use Debye, electrons use Thomas-Fermi,
    inv_ion_screen_sq = e2_epsilon0 * ne * Zstar / kTi
    inv_ele_screen_sq = e2_epsilon0 * ne * beta_e * 0.5 * Im0p5 / Ip0p5
    inv_screen_len_sq = inv_ion_screen_sq + inv_ele_screen_sq

    # Electron screening effect
    alpha = sqrt(inv_screen_len_sq / inv_ion_screen_sq)

    # Wigner-Seitz Radius
    Rws = (3 / (FOUR_PI * ni_tot)) ** (1 / 3)

    # Coupling parameter
    Gamma = Zmean * Zstar * e2_epsilon0 / (FOUR_PI * kTi * Rws)

    # Lambda terms
    Lambda_00 = (3 * Gamma) ** 1.5 / Zmean
    Lambda_0 = Lambda_00 * (Zi)
    Lambda_p = Lambda_00 * (Zi + 0.5)

    # X terms(ratio of ion core radius to ion sphere radius)
    X_0 = X_term(alpha, Lambda_0 ** (-1 / 3))
    X_p = X_term(alpha, Lambda_p ** (-1 / 3))
    # X_0 = X_term(alpha, Lambda_0)
    # X_p = X_term(alpha, Lambda_p)

    # Electron polarization terms
    X_const = alpha**2 * (ForceConst / 0.9) ** 1.5 - 1
    pol_0 = alpha * (1 + X_const * X_0**3)
    pol_p = alpha * (1 + X_const * X_p**3)

    # Polarized Lambda terms
    Lambda_0 *= pol_0
    Lambda_p *= pol_p

    # h and g term
    h0_p_g0 = h(Lambda_0) - g(Lambda_0)
    # h0_p_g0[Zi==0] = 0 # uncomment for charge state distribution
    hp = h(Lambda_p)

    # Fermi surface adjustment term for Pauli blocking
    if eta_e > 2:
        kTe_w = kTe * (eta_e - 2)
    else:
        kTe_w = 0.0

    # Spectroscopic IPD
    # SIPD = kTi/(2*Zstar) * ( hp + np.sum(csd*zi * h0_p_g0)/(2 * Zmean) ) - kTe_w
    # NOTE(TG): While there is no charge state distribution, this should do the job.
    #           When there are, use commented out formula
    SIPD = kTi / (2 * Zstar) * (hp + (Zi * h0_p_g0) / (2 * Zmean)) - kTe_w

    return SIPD

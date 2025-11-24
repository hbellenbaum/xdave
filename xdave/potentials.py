from .constants import (
    VACUUM_PERMITTIVITY,
    COULOMB_CONSTANT,
    PI,
    ELEMENTARY_CHARGE,
    BOLTZMANN_CONSTANT,
    SQRT_PI,
    BOHR_RADIUS,
)
from .utils import forward_transform_fft

from scipy.special import erf
import numpy as np


## Ion potentials for the ion-ion HNC solver


# 'normal' Coulomb potential
def coulomb_r(Qa, Qb, r):
    return Qa * Qb * ELEMENTARY_CHARGE**2 / (4 * PI * VACUUM_PERMITTIVITY * r)


def coulomb_k(Qa, Qb, k):
    """
    Coulomb potential in k-space, note that the inputs Qa and Qb here are given in units of e.
    """
    return Qa * Qb * ELEMENTARY_CHARGE**2 / (VACUUM_PERMITTIVITY * k**2)


# Screened Coulomb potential - Yukawa potential
def yukawa_r(Qa, Qb, r, alpha):
    return Qa * Qb * ELEMENTARY_CHARGE**2 / (4 * PI * VACUUM_PERMITTIVITY) * np.exp(-alpha * r) / r


def yukawa_k(Qa, Qb, k, alpha):
    return Qa * Qb * ELEMENTARY_CHARGE**2 / VACUUM_PERMITTIVITY * alpha**2 / (k**2 * (alpha**2 + k**2))


# Deutsch potential
def deutsch_r(Qa, Qb, r, alpha):
    return Qa * Qb * ELEMENTARY_CHARGE**2 / (4 * PI * VACUUM_PERMITTIVITY * r) * (1 - np.exp(-alpha * r))


def deutsch_k(Qa, Qb, k, alpha):
    return Qa * Qb * ELEMENTARY_CHARGE**2 / (VACUUM_PERMITTIVITY * k**2) * (alpha**2 / (k**2 + alpha**2))


# Kelbg
def kelbg_r(Qa, Qb, r, alpha):
    x = alpha * r
    gauss_func = erf(x)
    Vab = (
        Qa
        * Qb
        * ELEMENTARY_CHARGE**2
        / (4 * PI * VACUUM_PERMITTIVITY * r)
        * (1 - np.exp(-(x**2)) + SQRT_PI * x * (1 - gauss_func))
    )
    return Vab


def kelbg_k(Qa, Qb, k, alpha):
    return Qa * Qb * ELEMENTARY_CHARGE**2 / (VACUUM_PERMITTIVITY * k**2) * alpha**2 / (k**2 + alpha**2)


def short_range_screening_r(Qa, Qb, r, Ti, srr_core_power, ion_core_radius, srr_sigma, kappa_e, alpha):
    """
    Corrected version of the SSR first published in K. Wunsch et al., PRE 79 (2009) doi: 10.1103/physreve.79.010201
    """
    Vab = (
        debye_huckel_r(Qa, Qb, r, alpha, kappa_e)
        + ELEMENTARY_CHARGE**2
        * COULOMB_CONSTANT
        / BOHR_RADIUS
        * np.exp(srr_sigma - r / ion_core_radius)
        * (ion_core_radius / r) ** srr_core_power
    )
    return Vab


def debye_huckel_r(Qa, Qb, r, alpha, kappa_e):
    return coulomb_r(Qa=Qa, Qb=Qb, r=r) * np.exp(-(kappa_e + alpha) * r)


def debye_huckel_k(Qa, Qb, k, alpha, kappa_e):
    return (
        coulomb_k(Qa=Qa, Qb=Qb, k=k)
        * k**2
        * (alpha**2 + 2 * alpha * kappa_e)
        / ((k**2 + kappa_e**2) * (k**2 + (kappa_e + alpha) ** 2))
    )


def charge_switching_debye_r(
    Qa, Qb, r, csd_parameter_a, csd_parameter_b, csd_core_charge_a, csd_core_charge_b, kappa_e
):
    """
    Charge-switching debye potential given in J. Vorberger et al., PRL 109 (2012)
    and J. Vorberger and D. Gericke, HEDP 9 (2013)
    Not this will have to be run for each species pairing ab
    """
    y = (csd_parameter_a + csd_parameter_b) / 2
    Vab = (
        coulomb_r(Qa=Qa, Qb=Qb, r=r)
        * (1 + (csd_core_charge_a * csd_core_charge_b / (Qa * Qb)) - 1)
        * np.exp(-y * r)
        * np.exp(-(kappa_e + y) * r)
    )
    return Vab


def charge_switching_debye_k(
    Qa, Qb, k, csd_parameter_a, csd_parameter_b, csd_core_charge_a, csd_core_charge_b, alpha, kappa_e
):
    """
    Charge-switching debye potential given in J. Vorberger et al., PRL 109 (2012)
    and J. Vorberger and D. Gericke, HEDP 9 (2013)
    Not this will have to be run for each species pairing ab separately and expressed in matrix form
    """
    y = (csd_parameter_a + csd_parameter_b) / 2
    Vab = k**2.0 * (
        csd_core_charge_a
        * csd_core_charge_b
        / (Qa * Qb)
        * alpha
        * (alpha + 2.0 * (y + kappa_e))
        / (k**2.0 + (y + kappa_e) ** 2.0)
        / (k**2.0 + (alpha + y + kappa_e) ** 2.0)
        + 1.0 / (k**2.0 + (alpha + y + kappa_e) ** 2.0)
        - 1.0 / (k**2.0 + (y + kappa_e) ** 2.0)
        - 1.0 / (k**2.0 + (alpha + kappa_e) ** 2.0)
        + 1.0 / (k**2.0 + kappa_e**2.0)
    )
    return Vab * coulomb_k(Qa=Qa, Qb=Qb, k=k)


## Electron-ion potentials for the screening cloud
def ei_coulomb_r(Qa, r):
    return coulomb_r(Qa=Qa, Qb=-1, r=r)


def ei_coulomb_k(Qa, k):
    return coulomb_k(Qa=Qa, Qb=-1, k=k)


def ei_yukawa_r(Qa, r, alpha):
    return yukawa_r(Qa=Qa, Qb=-1, r=r, alpha=alpha)


def ei_yukawa_k(Qa, k, alpha):
    return yukawa_k(Qa=Qa, Qb=-1, k=k, alpha=alpha)


# Only the k-space expression is required in the ei potential for the screening cloud, so only these are included
def soft_core_ei_k(Qa, k, rcore, n):
    r = np.linspace(1.0e-2 * BOHR_RADIUS, 1.0e2 * BOHR_RADIUS, 8192)
    U_eff_k = ei_coulomb_k(Qa=Qa, k=k)
    U_eff_r = ei_coulomb_r(Qa=Qa, r=r)
    return U_eff_k - forward_transform_fft(U_eff_r * np.exp(-((r / rcore) ** n)))


def hard_core_ei_k(Qa, Qb, k, sigma_c):
    return coulomb_k(Qa, Qb, k) * np.cos(k * sigma_c / 2)


# Klimontovich, Kraeft
def klimontovich_kraeft_r(Qa, r, T, lambda_ei):
    beta = 1 / (BOLTZMANN_CONSTANT * T)
    zi = Qa * ELEMENTARY_CHARGE**2 * beta / (lambda_ei * 4 * PI * VACUUM_PERMITTIVITY)
    return (
        -BOLTZMANN_CONSTANT
        * T
        * zi**2
        / 16
        / (1 + BOLTZMANN_CONSTANT * T * zi**2 * r * 4 * PI * VACUUM_PERMITTIVITY / (16 * Qa * ELEMENTARY_CHARGE**2))
    )


def klimontovich_kraeft_k(Qa, k, T, lambda_ei):
    return np.zeros_like(k)

from xdave.constants import VACUUM_PERMITTIVITY, PI, ELEMENTARY_CHARGE, COULOMB_CONSTANT
from xdave.utils import forward_transform_fft

import numpy as np


## Ion potentials for the ion-ion HNC solver


# 'normal' Coulomb potential
def coulomb_r(Qa, Qb, r):
    return Qa * Qb * ELEMENTARY_CHARGE**2 / (4 * PI * VACUUM_PERMITTIVITY * r)


def coulomb_k(Qa, Qb, k):
    """
    Coulomb potential in k-space, note that the inputs Qa and Qb here are given in units of C.
    """
    return Qa * Qb * ELEMENTARY_CHARGE**2 / (VACUUM_PERMITTIVITY * k**2)


# Screened Coulomb potential - Yukawa potential
def yukawa_r(Qa, Qb, r, alpha):
    return Qa * Qb * ELEMENTARY_CHARGE**2 / (4 * PI * VACUUM_PERMITTIVITY) * np.exp(-alpha * r) / r


def yukawa_k(Qa, Qb, k, alpha):
    return Qa * Qb * ELEMENTARY_CHARGE**2 / VACUUM_PERMITTIVITY * alpha**2 / (k**2 * (alpha**2 + k**2))


## Electron-ion potentials for the screening cloud


def ei_coulomb_r(Qa, r):
    return coulomb_r(Qa=Qa, Qb=1, r=r)


def ei_coulomb_k(Qa, k):
    return coulomb_k(Qa=Qa, Qb=1, k=k)


def effective_ei_coulomb_r(Qa, r):
    return -Qa * ELEMENTARY_CHARGE**2 * COULOMB_CONSTANT / r


def effective_ei_coulomb_k(Qa, k):
    return -4 * PI * Qa * ELEMENTARY_CHARGE**2 * COULOMB_CONSTANT / k**2


def soft_core_k(Qa, k, r, rcore, n):
    U_eff_k = effective_ei_coulomb_k(Qa=Qa, k=k)
    U_eff_r = effective_ei_coulomb_r(Qa=Qa, r=r)
    return U_eff_k - forward_transform_fft(U_eff_r * np.exp(-((r / rcore) ** n)))


def hard_core_k(Qa, Qb, k, sigma_c):
    return coulomb_k(Qa, Qb, k) * np.cos(k * sigma_c / 2)

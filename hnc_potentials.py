from constants import VACUUM_PERMITTIVITY, PI, ELEMENTARY_CHARGE

import numpy as np


## Ion potentials
# Screened Coulomb potential - Yukawa potential
def springer_short_range_rs(Qa, Qb, r, alpha):
    return Qa * Qb / (4 * PI * VACUUM_PERMITTIVITY) * np.exp(-alpha * r) / r


def springer_long_range_rs(Qa, Qb, r, alpha):
    return Qa * Qb / (4 * PI * VACUUM_PERMITTIVITY) * (1 - np.exp(-alpha * r)) / r


def springer_long_range_ks(Qa, Qb, k, alpha):
    return Qa * Qb / VACUUM_PERMITTIVITY * alpha**2 / (k**2 * (alpha**2 + k**2))


def effective_coulomb():
    return


def coulomb():
    return


def coulomb_k(Qa, Qb, k):
    """
    Coulomb potential in k-space, note that the inputs Qa and Qb here are given in units of C.
    """
    return Qa * Qb * ELEMENTARY_CHARGE**2 / (VACUUM_PERMITTIVITY * k**2)

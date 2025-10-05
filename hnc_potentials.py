from constants import VACUUM_PERMITTIVITY, PI

import numpy as np


# Screened Coulomb potential - Yukawa potential
def springer_short_range_rs(Qa, Qb, r, alpha):
    return Qa * Qb / (4 * PI * VACUUM_PERMITTIVITY) * np.exp(-alpha * r) / r


def springer_long_range_rs(Qa, Qb, r, alpha):
    return Qa * Qb / (4 * PI * VACUUM_PERMITTIVITY) * (1 - np.exp(-alpha * r)) / r


def springer_long_range_ks(Qa, Qb, k, alpha):
    return Qa * Qb / VACUUM_PERMITTIVITY * alpha**2 / (k**2 * (alpha**2 + k**2))

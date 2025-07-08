import numpy as np
from constants import DIRAC_CONSTANT, SPEED_OF_LIGHT, BOHR_RADIUS
from unit_conversions import eV_TO_J


def calculate_q(angle, energy):
    angle *= np.pi / 180
    E0 = energy * eV_TO_J
    q = 2 * E0 / (DIRAC_CONSTANT * SPEED_OF_LIGHT) * np.sin(angle / 2)
    q *= BOHR_RADIUS
    return q


def calculate_angle(q, energy):
    """
    Calculates an angle for the relevant q value.
    Input:
    - q: in Hartree units
    Returns:
    - angle: degrees
    """

    q_value = q / BOHR_RADIUS

    # convert energy from eV to J
    E0 = energy * eV_TO_J

    # small angle approximation: see Eqn. (9) in [2]
    K = DIRAC_CONSTANT * SPEED_OF_LIGHT * q_value / (2 * E0)
    angle = 2 * np.arcsin(K)

    # convert angle from radians to degrees
    angle *= 180 / np.pi

    return angle

from constants import ELEMENTARY_CHARGE
from plasma_state import PlasmaState


def hnc_ab_structure(state: PlasmaState):

    beta = 1.0 / (ELEMENTARY_CHARGE * state.ion_temperature)
    # Raa = mean_sphere_radius()

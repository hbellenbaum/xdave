# Ionic form factors
from models import ModelOptions
from plasma_state import PlasmaState

from constants import BOHR_RADIUS, ELEMENTARY_CHARGE, PI, BOLTZMANN_CONSTANT
import numpy as np


class StaticStructureFactor:

    def __init__(self, state: PlasmaState, models: ModelOptions):
        self.state = state
        self.ss_model = models.static_structure_factor_approximation

    @classmethod
    def get_static_structure_factor(self, k):
        if self.ss_model == "MSA":
            return self.mean_spherical_approximation_ss(k)
        else:
            raise NotImplementedError(
                f"Model {self.ss_model} for the static structure factor not yet implemented. Try MSA :)"
            )

    def mean_spherical_approximation_ss(self, k):
        # This should really be a user-defined input
        ion_particle_diameter = None  ## this needs to be moved to the plasma state

        eta = PI / 6 * self.state.ion_number_density * ion_particle_diameter**3
        gamma = (
            self.state.charge_state**2
            * ELEMENTARY_CHARGE**2
            / (ion_particle_diameter * BOLTZMANN_CONSTANT * self.state.ion_temperature)
        )
        xi = np.sqrt(24 * eta * gamma)

        sink = np.sin(k)
        ksink = k * sink
        cosk = np.cos(k)
        kcosk = k * cosk
        ksqr = k * k
        kcub = ksqr * k
        kquad = ksqr * ksqr
        q = 1 - self.state.charge_state

        sqrt_term = 1 + 2 * (1 - eta) ** 3 * xi / (1 + 2 * eta) ** 2
        h0 = (1 + 2 * eta) / (1 - eta) * (1 - np.sqrt(sqrt_term))
        h1 = h0**2 / (24 * eta) - (1 + eta / 2) / (1 - eta) ** 2
        h2 = -(1 + eta - eta**2 / 5) / (12 * eta) - (1 - eta) * h0 / (12 * eta * xi)

        y0 = (
            -((1 + 2 * eta) ** 2) / (1 - eta) ** 4
            + h0**2 / (4 * (1 - eta) ** 2)
            - (1 + eta) * h0 * xi / (12 * eta)
            - (5 + eta**2) * xi**2 / (60 * eta)
        )
        y1 = 6 * eta * h1**2
        y2 = xi**2 / 6
        y3 = eta / 2 * (y0 + xi**2 * h2)
        y4 = eta * xi**2 / 60

        c1 = y0 * kcub * (sink - kcosk) + y1 * ksqr * (2 * ksink - (q**2 - 2) * cosk - 2)
        c2 = y2 * k * ((3 * ksqr - 6) * sink - (ksqr - 6) * kcosk)
        c3 = y3 * ((4 * ksqr - 24) * ksink - (ksqr * ksqr - 12 * ksqr + 24) * cosk + 24)
        c4 = y4 / ksqr * (6 * (kquad - 20 * ksqr + 120) * ksink - (k**6 - 30 * kquad + 360 * ksqr - 720) * cosk - 720)
        c5 = -gamma * kquad * cosk

        c_ii = 24 * eta / k**6 * (c1 + c2 + c3 + c4 - c5)
        S_ii_OCP = 1 / (1 - c_ii)

        kappa_i = 1 / self.state.debye_screening_length(
            (1 - self.state.charge_state), self.state.ion_number_density, self.state.ion_temperature
        )  ## inverse ion screening length
        kappa_e = 1 / self.state.debye_screening_length(
            self.state.charge_state, self.state.electron_number_density, self.state.electron_temperature
        )  ## inverse electron screening length

        ## weakly coupled limit
        # for the more general case, replace the dielectric function of the electrons by something like RPA
        dielectric = 1 + (kappa_e**2 / k**2)
        screening_cloud = kappa_i**2 / k**2 * np.cos(k * ion_particle_diameter / 2) * (1 / dielectric - 1)
        S_ii = S_ii_OCP / (1 + screening_cloud * S_ii_OCP)
        return S_ii

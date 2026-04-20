from .freefree_dsf import FreeFreeDSF
from .plasma_state import PlasmaState
from .constants import ELEMENTARY_CHARGE, BOHR_RADIUS
from .potentials import *

from .unit_conversions import *
from .constants import PI_SQR, SQRT_HALF_PI, SQRT_PI

import numpy as np
import matplotlib.pyplot as plt

from .fermi_integrals import fdi

import warnings


class ScreeningCloud:
    """
    Class to describe the electronic screening around the ions.

    Attributes:
        state (PlasmaState): container for all plasma parameters
        overlord_state (PlasmaState): mean plasma state to describe average plasma parameters
    """

    def __init__(self, state: PlasmaState, overlord_state: PlasmaState):
        self.state = state
        self.overlord_state = overlord_state

    def get_screening_cloud(
        self,
        k,
        lfc=0.0,
        screening_model="FINITE_WAVELENGTH",
        ee_potential="COULOMB",
        ei_potential="COULOMB",
    ):
        """
        Main run function to obtain the screening cloud.

        Parameters:
            k (float/array): wave number in units of 1/m
            lfc (float/array): local field correction, dimensionless
            screening_model (str): option to control the screening model
            ee_potential(str): option to control the electron-electron potential model
            ei_potential (str): option to control the electron-ion potential model

        Returns:
            float/array: screening cloud depending on the k-input type
        """

        Zi = self.state.ion_charge

        alpha = 2 / self.state.mean_sphere_radius(number_density=self.state.ion_number_density)

        if ee_potential == "COULOMB":
            Uee = coulomb_k(Qa=1, Qb=1, k=k)
        elif ee_potential == "YUKAWA":
            Uee = yukawa_k(Qa=1, Qb=1, k=k, alpha=alpha)
        else:
            warnings.warn(
                f"Model {ee_potential} not recognized for the electron-electron potential. Overwriting using COULOMB."
            )
            Uee = coulomb_k(Qa=1, Qb=1, k=k)

        if ei_potential == "COULOMB":
            Uei = ei_coulomb_k(Qa=Zi, k=k)
        elif ei_potential == "YUKAWA":
            Uei = ei_yukawa_k(Qa=Zi, k=k, alpha=alpha)
        elif ei_potential == "HARD_CORE":
            Uei = hard_core_ei_k(Qa=-1, Qb=Zi, k=k, sigma_c=self.state.ion_core_radius)
        elif ei_potential == "SOFT_CORE":
            Uei = soft_core_ei_k(Qa=-1, Qb=Zi, k=k, rcore=self.state.ion_core_radius, n=self.state.sec_power)
        else:
            Uei = ei_yukawa_k(Qa=Zi, k=k, alpha=alpha)
            warnings.warn(
                f"Model {ei_potential} not recognized for the electron-ion potential. Overwriting using YUKAWA."
            )

        if screening_model == "DEBYE_HUCKEL":
            screening_cloud = self._debye_huckel_screening_full(k=k, lfc=lfc, Uee=Uee, Uei=Uei)
        elif screening_model == "FINITE_WAVELENGTH":
            screening_cloud = self._finite_wavelength_screening_full(k=k, lfc=lfc, Uee=Uee, Uei=Uei)
        elif screening_model == "NONE":
            screening_cloud = np.zeros_like(k)
        else:
            warnings.warn(
                f"Model {screening_model} for the screening cloud not recognized. Overwriting using FINITE_WAVELENGTH."
            )
            screening_cloud = self._finite_wavelength_screening_full(k=k, lfc=lfc, Uee=Uee, Uei=Uei)

        return screening_cloud

    def _debye_huckel_screening_full(self, k, lfc, Uee, Uei):
        kappa_e_full = self.overlord_state.screening_length(
            mass=ELECTRON_MASS,
            charge=1,
            temperature=self.overlord_state.electron_temperature,
            number_density=self.overlord_state.free_electron_number_density,
        )
        screening_length = kappa_e_full**2
        ratio = Uei / Uee
        screening_cloud = -ratio * screening_length / (k**2 + (1 - lfc) * screening_length)

        return screening_cloud

    def _debye_huckel_screening(self, k):
        """
        Small wavenumber limit of the screening cloud in the RPA

        Parameters:
            k (float/array): wave number in units of 1/m

        Returns:
            float/array: screening length depending on the k-input type
        """
        kappa_e = self.overlord_state.debye_screening_length(
            1, self.overlord_state.free_electron_number_density, self.overlord_state.electron_temperature
        )
        kappa_e = np.real(kappa_e)
        screening_length = kappa_e**2
        return screening_length

    def _finite_wavelength_screening_full(self, k, lfc, Uee, Uei):

        pol_func = FreeFreeDSF(state=self.overlord_state).dandrea_fit(k=k, w=0.0).real
        screening_length = -(k**2) * Uee * pol_func
        ratio = Uei / Uee
        screening_cloud = -ratio * screening_length / (k**2 + (1 - lfc) * screening_length)

        return screening_cloud

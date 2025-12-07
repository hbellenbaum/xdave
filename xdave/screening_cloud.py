from .freefree_dsf import FreeFreeDSF
from .plasma_state import PlasmaState
from .constants import ELEMENTARY_CHARGE, BOHR_RADIUS
from .potentials import *

from .unit_conversions import *
from .constants import PI_SQR, SQRT_HALF_PI, SQRT_PI

import numpy as np
import matplotlib.pyplot as plt

# from plasmapy.formulary.mathematics import Fermi_integral as fdi
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
            # raise NotImplementedError(f"Cannot recognize ee-potential {ee_potential} in the screening cloud.")
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
            # raise NotImplementedError(f"Cannot recognize ei-potential {ei_potential} in the screening cloud.")
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
                f"Model {screening_model} for the screening cloud not regocnized. Overwriting using FINITE_WAVELENGTH."
            )
            screening_cloud = self._finite_wavelength_screening_full(k=k, lfc=lfc, Uee=Uee, Uei=Uei)
            # raise NotImplementedError(f" Model for the screening cloud: {screening_model} not recognised.")

        return screening_cloud

        # if screening_model == "DEBYE_HUCKEL":
        #     screening_length = self._debye_huckel_screening(k)
        # elif screening_model == "FINITE_WAVELENGTH":
        #     screening_length = self._finite_wavelength_screening_short(k)
        # elif screening_model == "NONE":
        #     screening_length = np.zeros_like(k)
        # else:
        #     raise NotImplementedError(f" Model for the screening cloud: {screening_model} not recognised.")

        # ratio = Uei / Uee
        # screening_cloud = -ratio * screening_length / (k**2 + (1 - lfc) * screening_length)

        # return screening_cloud

    def _debye_huckel_screening_full(self, k, lfc, Uee, Uei):
        kappa_e_full = self.overlord_state.screening_length(
            mass=ELECTRON_MASS,
            charge=1,
            temperature=self.overlord_state.electron_temperature,
            number_density=self.overlord_state.free_electron_number_density,
        )
        kappa_e = self.overlord_state.debye_screening_length(
            1, self.overlord_state.free_electron_number_density, self.overlord_state.electron_temperature
        )
        # kappa_e = np.real(kappa_e)
        # kappa_e = 28809906697.517681
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
        # kappa_e = self.overlord_state.screening_length(
        #     mass=ELECTRON_MASS,
        #     charge=1,
        #     temperature=self.overlord_state.electron_temperature,
        #     number_density=self.overlord_state.total_electron_number_density,
        # )
        kappa_e = self.overlord_state.debye_screening_length(
            1, self.overlord_state.free_electron_number_density, self.overlord_state.electron_temperature
        )
        kappa_e = np.real(kappa_e)
        screening_length = kappa_e**2
        return screening_length

    def _finite_wavelength_screening_full(self, k, lfc, Uee, Uei):

        pol_func = FreeFreeDSF(state=self.overlord_state).dandrea_fit(k=k, omega=0.0).real
        screening_length = -(k**2) * Uee * pol_func
        ratio = Uei / Uee
        screening_cloud = -ratio * screening_length / (k**2 + (1 - lfc) * screening_length)

        return screening_cloud

    def _finite_wavelength_screening_short(self, k):
        r"""
        FWS screening uses the RPA at omega=0 to estimate electronic screening.
        Here, the Dandrea fit is used for a quick evaluation.

        Parameters:
            k (float/array): wave number in units of 1/m

        Returns:
            float/array: screening length depending on the k-input type
        """

        kF = self.overlord_state.fermi_wave_number(self.overlord_state.free_electron_number_density)
        EF = self.overlord_state.fermi_energy(
            mass=ELECTRON_MASS, number_density=self.overlord_state.free_electron_number_density
        )
        kappa_e = self.overlord_state.screening_length(
            ELECTRON_MASS,
            1,
            self.overlord_state.electron_temperature,
            self.overlord_state.free_electron_number_density,
        )
        # kappa_e = self.overlord_state.debye_screening_length(
        #     1, self.overlord_state.free_electron_number_density, self.overlord_state.electron_temperature
        # )
        theta = ELEMENTARY_CHARGE * self.overlord_state.electron_temperature / EF
        sqrt_theta = np.sqrt(theta)
        x = 0.5 * k / kF
        eta = self.overlord_state.chemical_potential_ichimaru(
            self.overlord_state.electron_temperature, self.overlord_state.free_electron_number_density, ELECTRON_MASS
        )
        eta /= EF

        Fm1p5 = fdi(j=-1.5, eta=eta, normalize=True)
        Fm0p5 = fdi(j=-0.5, eta=eta, normalize=True)
        Fp1p5 = fdi(j=1.5, eta=eta, normalize=True)
        Fp2p5 = fdi(j=2.5, eta=eta, normalize=True)

        c2 = np.array([-2.28e-1, 4.222e-1, -6.466e-1, 7.0572e-1, 5.882])
        c4 = np.array([-3.0375, 6.4646e1, 1.9608e1, -9.6978e1, 4.2366e2, -3.3101e2, 2.0833e1])
        c6 = np.array([-1.9e-1, 3.6538e-1, -2.2575, 2.2942e1, -4.3492e1, 1.064e2])
        c8 = np.array([-7.1316, 2.2725e1, 5.8092e1, -4.3602e1, -8.2651e2, 4.9129e3, 9.1e-1, -6.4453, 1.22324e1])

        a2 = (c2[1 - 1] + theta) / (c2[2 - 1] + c2[3 - 1] * theta ** c2[4 - 1] + c2[5 - 1] * theta**2.0)
        a4 = (1.0 + c4[1 - 1] * theta + c4[2 - 1] * theta**2.0) / (
            c4[3 - 1]
            + c4[4 - 1] * theta
            + c4[5 - 1] * theta**2.0
            + c4[6 - 1] * theta**3.0
            + c4[7 - 1] * c4[2 - 1] * theta**4.0
        )
        a6 = (c6[1 - 1] + theta) / (
            c6[2 - 1] + c6[3 - 1] * theta + c6[4 - 1] * theta**2.0 + c6[5 - 1] * theta**3.0 + c6[6 - 1] * theta**4.0
        )
        a8 = (c8[7 - 1] + c8[8 - 1] * theta + c8[9 - 1] * theta**2.0) / (
            1.0
            + c8[1 - 1] * theta
            + c8[2 - 1] * theta**2.0
            + c8[3 - 1] * theta**3.0
            + c8[4 - 1] * theta**4.0
            + c8[5 - 1] * theta**5.0
            + c8[6 - 1] * theta**6.0
        )

        cK = np.array([-4.878, 4.7325e2, -2.3375e3, 3.4831e2, 1.5173e3])
        Kfit = (1.0 + cK[0] * theta**2.0 + cK[1] * theta**4.0 + cK[2] * theta**7.0) / (
            1.0
            + (cK[0] - 0.75 * PI_SQR) * theta**2.0
            + cK[3] * theta**4.0
            + cK[4] * theta**7.0
            - 0.875 * cK[2] * theta**8.5 / SQRT_HALF_PI
            - 0.375 * cK[2] * theta**1.0e1
        )
        # Dandrea fit functions
        b10 = 1.5 * SQRT_PI * sqrt_theta * Fm0p5 * a8
        b8 = 1.5 * SQRT_PI * (sqrt_theta * Fm0p5 * a6 - theta**2.5 * Fp1p5 * b10 / 4.0)
        b6 = (
            1.5
            * SQRT_PI
            * (sqrt_theta * Fm0p5 * a4 - theta**2.5 * Fp1p5 * b8 / 4.0 - 3.0 * theta**3.5 * Fp2p5 * b10 / 8.0)
        )
        b2 = a2 + 2.0 * Fm1p5 / (3.0 * theta * Fm0p5)
        b4 = b2**2.0 - a2 * b2 + a4 + 2.0 * Kfit / (1.5e1 * SQRT_PI * sqrt_theta * Fm0p5)
        screening_length = kappa_e * np.sqrt(
            (1 + a2 * x**2 + a4 * x**4 + a6 * x**6 + a8 * x**8)
            / (1 + b2 * x**2 + b4 * x**4 + b6 * x**6 + b8 * x**8 + b10 * x**10)
        )
        return np.real(screening_length) ** 2

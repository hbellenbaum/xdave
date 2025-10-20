from models import ModelOptions
from plasma_state import PlasmaState
from constants import ELEMENTARY_CHARGE, BOHR_RADIUS
from potentials import *
from freefree_dsf import FreeFreeDSF

from unit_conversions import *
from constants import PI_SQR, SQRT_HALF_PI, SQRT_PI

import numpy as np
import matplotlib.pyplot as plt

from plasmapy.formulary.mathematics import Fermi_integral as fdi


class ScreeningCloud:

    def __init__(self, state: PlasmaState, overlord_state=PlasmaState):
        self.state = state
        self.overlord_state = overlord_state
        # self.screening_model = models.screening_model
        # self.ei_potential = models.ei_potential
        # self.kappa_e = 1 / self.state.debye_screening_length(
        #     ELEMENTARY_CHARGE, self.state.electron_number_density, self.state.electron_temperature
        # )

    def get_screening_cloud(
        self, k, ion_core_radius, lfc=0.0, screening_model="FWS", ee_potential="COULOMB", ei_potential="COULOMB"
    ):
        screening_length = 0.0
        Zi = self.state.ion_charge

        # Uee = coulomb_k(Qa=1, Qb=1, k=k)
        # Uei = coulomb_k(Qa=1, Qb=self.state.ion_charge, k=k)
        alpha = 2 / self.state.mean_sphere_radius(number_density=self.state.ion_number_density)

        if ee_potential == "COULOMB":
            Uee = coulomb_k(Qa=-1, Qb=-1, k=k)
        elif ee_potential == "YUKAWA":
            Uee = yukawa_k(Qa=-1, Qb=-1, k=k, alpha=alpha)
        else:
            raise NotImplementedError(f"Cannot recognize ee-potential {ee_potential}.")

        if ei_potential == "COULOMB":
            Uei = ei_coulomb_k(Qa=Zi, k=k)
        elif ei_potential == "YUKAWA":
            Uei = ei_yukawa_k(Qa=Zi, k=k, alpha=alpha)
        elif ei_potential == "HARD_CORE":
            Uei = hard_core_k(Qa=-1, Qb=Zi, k=k, sigma_c=ion_core_radius)
        else:
            raise NotImplementedError(f"Cannot recognize ei-potential {ei_potential}.")

        if screening_model == "DEBYE_HUCKEL":
            screening_length = self._debye_huckel_screening(k)
        elif screening_model == "FINITE_WAVELENGTH":
            screening_length = self._finite_wavelength_screening_short(k)
        else:
            raise NotImplementedError(f" Model for the screening cloud: {screening_model} not recognised.")

        G = lfc
        ratio = Uei / Uee
        screening_cloud = -ratio * screening_length / (k**2 + (1 - G) * screening_length)

        return screening_cloud

    def _debye_huckel_screening(self, k):
        """
        Small wavenumber limit of the screening cloud
        """
        kappa_e = self.overlord_state.screening_length(
            mass=ELECTRON_MASS,
            charge=1,
            temperature=self.overlord_state.electron_temperature,
            number_density=self.overlord_state.total_electron_number_density,
        )
        kappa_e = np.real(kappa_e)
        print(f"Inverse screening length = {kappa_e}")
        screening_length = kappa_e**2
        return screening_length

    def _finite_wavelength_screening_long(self, k, Uee, Uei):
        # kappa_e = self.state.screening_length(
        #     mass=ELECTRON_MASS,
        #     charge=1,
        #     temperature=self.state.electron_temperature,
        #     number_density=self.state.total_electron_number_density,
        # ).real
        kappa_e = self.overlord_state.debye_screening_length(
            1, self.overlord_state.free_electron_number_density, self.overlord_state.electron_temperature
        )
        lfc = 0.0
        # TODO(Hannah): use a simplified form for the dandrea fit here, for now, this works though

        RPA_pol = FreeFreeDSF(state=self.overlord_state).susceptibility_function(k=k, w=0, model="DANDREA_FIT")
        # screening_length = kappa_e * np.sqrt(-Uee * RPA_pol.real)
        screening_cloud = Uei * RPA_pol.real / (1 - (1 - lfc) * Uee * RPA_pol.real)
        # screening_cloud = RPA_pol * Uei / (1 - Uee * RPA_pol)
        return screening_cloud  # screening_length**2

    def _finite_wavelength_screening_short(self, k):

        kF = self.overlord_state.fermi_wave_number(self.overlord_state.free_electron_number_density)
        EF = self.overlord_state.fermi_energy(
            mass=ELECTRON_MASS, number_density=self.overlord_state.free_electron_number_density
        )
        kappa_e = self.overlord_state.debye_screening_length(
            1, self.overlord_state.free_electron_number_density, self.overlord_state.electron_temperature
        )
        theta = ELEMENTARY_CHARGE * self.state.electron_temperature / EF
        sqrt_theta = np.sqrt(theta)
        x = 0.5 * k / kF
        eta = self.overlord_state.chemical_potential_ichimaru(
            self.overlord_state.electron_temperature, self.overlord_state.free_electron_number_density, ELECTRON_MASS
        )
        eta /= EF

        Fm1p5 = fdi(j=-1.5, x=eta)
        Fm0p5 = fdi(j=-0.5, x=eta)
        Fp1p5 = fdi(j=1.5, x=eta)
        Fp2p5 = fdi(j=2.5, x=eta)

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
        return screening_length**2


def test():
    T = 4 * eV_TO_K
    Zi = 2
    rho = 498.16  # kg/m^3
    state = PlasmaState(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_state=Zi,
        atomic_mass=2,
        atomic_number=2,
        binding_energies=None,
    )

    k = np.linspace(1.0e-1 / BOHR_RADIUS, 10 / BOHR_RADIUS, 200)

    sigma_c = 2.15 * BOHR_RADIUS

    kernel = ScreeningCloud(state=state, overlord_state=state)
    f_fws = kernel.get_screening_cloud(
        k=k, ion_core_radius=sigma_c, screening_model="FINITE_WAVELENGTH", ei_potential="COULOMB"
    )
    f_dh = kernel.get_screening_cloud(
        k=k, ion_core_radius=sigma_c, screening_model="DEBYE_HUCKEL", ei_potential="COULOMB"
    )

    plt.figure()
    plt.plot(k * BOHR_RADIUS, f_fws, label="FWS")
    plt.plot(k * BOHR_RADIUS, f_dh, label="DH")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test()

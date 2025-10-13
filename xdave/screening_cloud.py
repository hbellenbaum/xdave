from models import ModelOptions
from plasma_state import PlasmaState
from constants import ELEMENTARY_CHARGE, BOHR_RADIUS
from potentials import *
from freefree_dsf import FreeFreeDSF

from unit_conversions import *

import numpy as np
import matplotlib.pyplot as plt


class ScreeningCloud:

    def __init__(self, state: PlasmaState):
        self.state = state
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
            Uei = coulomb_k(Qa=-1, Qb=Zi, k=k)
        elif ei_potential == "YUKAWA":
            Uei = yukawa_k(Qa=-1, Qb=Zi, k=k, alpha=alpha)
        elif ei_potential == "HARD_CORE":
            Uei = hard_core_k(Qa=-1, Qb=Zi, k=k, sigma_c=ion_core_radius)
        else:
            raise NotImplementedError(f"Cannot recognize ei-potential {ei_potential}.")

        if screening_model == "DEBYE_HUCKEL":
            screening_length = self._debye_huckel_screening(k)
        elif screening_model == "FINITE_WAVELENGTH":
            screening_length = self._finite_wavelength_screening(k, Uee=Uee)
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
        kappa_e = self.state.debye_screening_length(
            1, self.state.electron_number_density, self.state.electron_temperature
        )
        screening_length = kappa_e**2
        return screening_length

    def _finite_wavelength_screening(self, k, Uee):
        # TODO(Hannah): use a simplified form for the dandrea fit here, for now, this works though
        RPA_pol = FreeFreeDSF(state=self.state).dandrea_fit(k, omega=0)
        screening_length = -(k**2) * RPA_pol.real * Uee
        return screening_length


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

    kernel = ScreeningCloud(state=state)
    f_fws = kernel.get_screening_cloud(k=k, ion_core_radius=sigma_c, screening_model="FWS", ei_potential="COULOMB")
    f_dh = kernel.get_screening_cloud(k=k, ion_core_radius=sigma_c, screening_model="DH", ei_potential="COULOMB")

    plt.figure()
    plt.plot(k * BOHR_RADIUS, f_fws, label="FWS")
    plt.plot(k * BOHR_RADIUS, f_dh, label="DH")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test()

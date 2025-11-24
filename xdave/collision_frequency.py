from .potentials import coulomb_k
from .static_sf import OCPStaticStructureFactor

from .freefree_dsf import FreeFreeDSF

# import freefree_dsf
from .plasma_state import PlasmaState

from .constants import PI, DIRAC_CONSTANT, BOHR_RADIUS, ELECTRON_MASS

from scipy.interpolate import interp1d
from scipy.integrate import quad, quad_vec
import numpy as np


class CollisionFrequency:
    """
    Class containing the collision frequency in the Born approximation.

    Attributes:
        state (PlasmaState): object containing all plasma state variables
    """

    def __init__(self, state: PlasmaState):
        self.state = state

    def get(self, k, w, lfc, model="BORN"):
        """
        Main function to call the collision frequency.

        Parameters:
            k (float): wave number in units of 1/m
            w (array): energy grid in units of 1/J
            lfc (float): local field correction, dimensionless
            model (str): controls the model used for calculating the collision frequency, default is BORN

        Returns:
            array: calculated electron-ion collisoin frequency in units of J [????]
        """
        # TODO(HB): check the units here
        if model == "BORN":
            return self.born_ei_collision_frequency(k, w, lfc)
        elif model == "ZIMAN":
            return self.ziman_ei_collision_frequency()
        else:
            raise NotImplementedError(f"Model {model} not recognized.")

    def ziman_ei_collision_frequency(self):
        """
        Calculate the Ziman collision frequency.
        For details see Eqn. (12) in Fortmann et al., Phys. Rev. E 81 (2010).

        Returns:
            array: collision frequency in units of J
        """
        rs = self.state.rs
        EF = self.state.fermi_energy(self.state.free_electron_number_density, ELECTRON_MASS)
        a = 0.11523
        b = 6.02921
        collision_frequency = EF / DIRAC_CONSTANT * (a * rs**2 * (np.log(1 + b / rs) - 1 / (1 + rs / b)))
        return collision_frequency

    def born_ei_collision_frequency(self, k, w, lfc):
        """
        Calculate the Born collision frequency based on Appendix B in Sch\"orner et al., Phys Rev. E 107 (2023).

        Parameters:
            k (float): wave number in units of 1/m
            w (array): energy grid in units of 1/J
            lfc (float): local field correction, dimensionless

        Returns:
            array: collision frequency in units of J
        """

        ne = self.state.free_electron_number_density
        mi = self.state.atomic_mass
        omega = w / DIRAC_CONSTANT
        omega_p = self.state.plasma_frequency(charge=self.state.charge_state, number_density=ne, mass=ELECTRON_MASS)
        kF = self.state.fermi_wave_number(ne)

        # this is going to be really slow...
        k_temp = np.linspace(1.0e-3, 1.0e2, 1000) / BOHR_RADIUS
        Siik = OCPStaticStructureFactor(state=self.state).get_ii_static_structure_factor(k=k_temp, sf_model="HNC")

        interp_Sk = interp1d(k_temp, Siik, fill_value="extrapolate")

        omega_temp = np.linspace(1.0e-6 * omega_p, 1.05 * np.max(omega), 25) * DIRAC_CONSTANT

        def integral_func(k, omega):
            Veek = coulomb_k(-1, -1, k)
            epsilon_kw = 1 - Veek * FreeFreeDSF(state=self.state).susceptibility_function(
                k=k, w=omega, model="DANDREA_FIT"
            )
            epsilon_k0 = 1 - Veek * FreeFreeDSF(state=self.state).susceptibility_function(
                k=k, w=0, model="DANDREA_FIT"
            )
            return -1.0j * (epsilon_kw - epsilon_k0) / epsilon_k0**2

        def _im_integral_func(u):
            x = np.tan(PI * u / 2)
            norm = PI / 2 * (1 + x**2)
            q = kF * x
            Siik_temp = interp_Sk(q)
            f = norm * x**2 * Siik_temp * np.imag(integral_func(q, omega_temp))
            return f

        def _real_integral_func(u):
            x = np.tan(PI * u / 2)
            norm = PI / 2 * (1 + x**2)
            q = kF * x
            Siik_temp = interp_Sk(q)
            f = x**2 * Siik_temp * np.real(integral_func(q, omega_temp)) * norm
            return f

        real_part = quad_vec(_real_integral_func, 0, 1)[0]
        imag_part = quad_vec(_im_integral_func, 0, 1)[0]
        collision_frequency = (
            omega_p
            * (mi / ELECTRON_MASS)
            * (real_part + imag_part * 1.0j)
            / (6 * PI * PI * ne * omega_temp * BOHR_RADIUS**3)
        )

        neg_omega = -omega_temp[::-1]
        neg_collision_frequency = (
            omega_p
            * (mi / ELECTRON_MASS)
            * (real_part[::-1] - imag_part[::-1] * 1.0j)
            / (6 * PI * PI * ne * omega_temp[::-1] * BOHR_RADIUS**3)
        )
        # interp_muei = interp1d(omega_temp, collision_frequency)
        full_w = np.concatenate([neg_omega, omega_temp])
        full_muei = np.concatenate([neg_collision_frequency, collision_frequency])
        muei_interp = interp1d(full_w, full_muei, fill_value="extrapolate")
        return full_w, full_muei, muei_interp(w)

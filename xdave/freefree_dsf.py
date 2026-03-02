from .constants import *
from .unit_conversions import *
from .maths import log1pexp
from .plasma_state import PlasmaState
from .fermi_integrals import fdi
from .static_sf import OCPStaticStructureFactor

from scipy import integrate

# from collision_frequency import CollisionFrequency


import numpy as np

import warnings


class FreeFreeDSF:
    """
    Class containing the free-free dynamic structure factor calculations.

    Attributes:
        state (PlasmaState): object containing all plasma state variables

    """

    def __init__(self, state: PlasmaState) -> None:
        self.state = state

    def get_dsf(self, k, w, lfc, model="NUMERICAL_RPA", input_collision_frequency=None):
        """
        Main function to call the dynamic structure factor for a given model.
        This internally calculates the electron susceptibility and applies a local field correction.

        Parameters:
            k (float): wave number in units of 1/m
            w (array): energy grid in units of 1/J
            lfc (float): local field correction, dimensionless
            model (str): controls the model used for calculating the dsf, default is NUMERICAL_RPA

        Returns:
            array: calculated free-free dsf in units of 1/J
        """

        # This is a last-resort, in case checks in the main run function don't work as expected
        if self.state.free_electron_number_density == 0.0:
            return 0.0

        w0_idx = np.where(w == 0.0)
        if len(w0_idx) is not None:
            w = np.delete(w, w0_idx)

        # Call the susceptibility function for a given model
        chi0 = self.susceptibility_function(k=k, w=w, model=model, input_collision_frequency=input_collision_frequency)

        # coulomb potential for the electron-electron interactions
        Vee = 4 * np.pi * COULOMB_CONSTANT * ELEMENTARY_CHARGE**2 / k**2

        # calculate corrected suscepibility
        chilfc = chi0 / (1 - Vee * (1 - lfc) * chi0)
        im_suspectibility = np.imag(chilfc)

        # DSF calculation
        S_EG_LFC = (
            -(1)
            / (PI * self.state.free_electron_number_density)
            * 1
            / (1 - np.exp(-w / (BOLTZMANN_CONSTANT * self.state.electron_temperature)))
            * im_suspectibility
        )

        # This feels really lazy, but seems to get the job done.
        if len(w0_idx) is not None:
            S_w0 = 0.5 * (S_EG_LFC[w0_idx[0]] + S_EG_LFC[w0_idx[0] - 1])
            S_EG_LFC = np.insert(S_EG_LFC, w0_idx[0], S_w0)

        return S_EG_LFC

    def get_collision_frequency(self, k, w, lfc, model):
        if model == "BORN":
            # return self._born_ei_collision_frequency(k, w, lfc)
            return self._full_born_ei_collision_frequency(k, w, lfc)
        elif model == "ZIMAN":
            return self._ziman_ei_collision_frequency()
        else:
            raise NotImplementedError(f"Model {model} not recognized.")

    def dielectric_function(self, k, w, model, collision_frequency_model="BORN", input_collision_frequency=None):
        """
        Calculate the free electron dielectric function for a given model.

        Parameters:
            k (float): wave number in units of 1/m
            w (array): energy grid in units of 1/J
            lfc (float): local field correction, dimensionless
            model (str): controls the model used for calculating the dsf

        Returns:
            array: calculated free electron dielectric, non-dimensional
        """
        potential_func = 4 * np.pi * COULOMB_CONSTANT * ELEMENTARY_CHARGE**2 / k**2
        if model == "LINDHARD":
            pol_func = self.lindhard_pol_func_dc(k=k, w=w)
            dielectric_func = 1 - potential_func * pol_func
        elif model == "DANDREA_FIT":
            pol_func = self.dandrea_fit(k=k, omega=w)
            dielectric_func = 1 - potential_func * pol_func
        elif model == "NUMERICAL":
            dielectric_func = self.rpa_numerical_dielectric_func(k=k, w=w)
        elif model == "MERMIN":
            warnings.warn(
                f"Model {model} for the free-free component not yet working properly and should not be used."
            )
            dielectric_func = self.mermin_dielectric_function(
                k=k,
                w=w,
                collision_frequency_model=collision_frequency_model,
                input_collision_frequency=input_collision_frequency,
            )
        else:
            dielectric_func = self.rpa_numerical_dielectric_func(k=k, w=w)
            warnings.warn(f"Model {model} for the free-free component not recognized. Overwriting using NUMERICAL.")

        return dielectric_func

    def susceptibility_function(self, k, w, model, collision_frequency_model="BORN", input_collision_frequency=None):
        """
        Calculate the free electron susceptibility (polarisation) function for a given model.

        Parameters:
            k (float): wave number in units of 1/m
            w (array): energy grid in units of 1/J
            lfc (float): local field correction, dimensionless
            model (str): controls the model used for calculating the dsf

        Returns:
            array: calculated free electron susceptibility, non-dimensional
        """
        if model == "LINDHARD":
            warnings.warn(f"Lindhard model currently not working. Try something else.")
            susceptibility_func = self.lindhard_pol_func_dc(k=k, w=w)
        elif model == "DANDREA_FIT":
            susceptibility_func = self.dandrea_fit(k=k, omega=w)
        elif model == "NUMERICAL":
            potential_func = 4 * np.pi * COULOMB_CONSTANT * ELEMENTARY_CHARGE**2 / k**2
            dielectric_func = self.rpa_numerical_dielectric_func(k=k, w=w)  # * potential_func
            susceptibility_func = (1 - dielectric_func) / potential_func
        elif model == "MERMIN":
            potential_func = 4 * np.pi * COULOMB_CONSTANT * ELEMENTARY_CHARGE**2 / k**2
            dielectric_func = self.mermin_dielectric_function(
                k=k,
                w=w,
                collision_frequency_model=collision_frequency_model,
                input_collision_frequency=input_collision_frequency,
            )
            susceptibility_func = (1 - dielectric_func) / potential_func
        else:
            potential_func = 4 * np.pi * COULOMB_CONSTANT * ELEMENTARY_CHARGE**2 / k**2
            dielectric_func = self.rpa_numerical_dielectric_func(k=k, w=w)
            susceptibility_func = (1 - dielectric_func) / potential_func
            warnings.warn(f"Model {model} for the free-free component not recognized. Overwriting using NUMERICAL.")

        return susceptibility_func

    def lindhard_pol_func(self, k, w):
        """
        Lindhard model for the susceptibility function, analytic limit for the fully degenerate plasma.
        Note to self: the definition of the plasma frequency has changed, I will need to check this.

        Parameters:
            k (float): wave number in units of 1/m
            w (array): energy grid in units of 1/J

        Returns:
            array: susceptibility function, non-dimensional

        """
        EF = self.state.fermi_energy(self.state.free_electron_number_density, ELECTRON_MASS)
        kF = self.state.fermi_wave_number(self.state.free_electron_number_density)
        omega_p = self.state.plasma_frequency(
            self.state.charge_state, self.state.free_electron_number_density, ELECTRON_MASS
        )
        gamma = EF / (DIRAC_CONSTANT * omega_p)

        vF = DIRAC_CONSTANT * kF / ELECTRON_MASS  # m/s
        U = w / (k * vF * DIRAC_CONSTANT)  # dimensionless
        Z = k / (2 * kF)  # dimensionless
        prefactor = -(3 * k**2 / (512 * PI * gamma**2 * Z**3 * ELEMENTARY_CHARGE_SQR * COULOMB_CONSTANT))
        input1 = (U - Z - 1) / (U - Z + 1)
        input2 = (U + Z - 1) / (U + Z + 1)
        if input1 <= 0:
            log1 = np.log(np.abs(input1)) + PI * 1.0j
        else:
            log1 = np.log(input1)
            # log1 =
        if input2 <= 0:
            log2 = np.log(np.abs(input2)) + PI * 1.0j
        else:
            log2 = np.log(input2)
        func_left = 4 * Z + (1 - (U - Z) ** 2) * log1
        func_right = 1 - (U + Z) ** 2 * log2
        pol_func = prefactor * (func_left - func_right)
        return pol_func

    def lindhard_pol_func_dc(self, k, w):
        EF = self.state.fermi_energy(self.state.free_electron_number_density, ELECTRON_MASS)
        kF = self.state.fermi_wave_number(self.state.free_electron_number_density)
        vF = DIRAC_CONSTANT * kF / ELECTRON_MASS  # m/s

        q0 = k / (2 * kF)  # []
        w0 = w / (k * vF * DIRAC_CONSTANT)  # []

        def lindhard_func(x):
            real_part = -x - 1 / 2 * (1 - x**2) * np.log(np.abs((x + 1) / (x - 1)))
            im_part = HALF_PI * (1 - x**2) * np.heaviside(1.0 - x**2, 1.0)
            return real_part + 1.0j * im_part

        G_plus = lindhard_func(w0 + q0)
        G_minus = lindhard_func(w0 - q0)
        pol_func = 3 * self.state.free_electron_number_density / (4 * EF * q0) * (G_plus - G_minus)  # / (4 * q0)

        return pol_func

    def rpa_numerical_dielectric_func(self, k, w):
        """
        Numerically solving the dielectric function in RPA by separating real and imaginary components.
        Based on Eqn. (5.5) in K. W\"unsch PhD Thesis (2011)

        Parameters:
            k (float): wave number in units of 1/m
            w (array): energy grid in units of 1/J

        Returns:
            array: dielectric function, non-dimensional
        """
        im_part = self._im_dielectric_rpa(k, w)  # [#]
        real_part = self._real_dielectric_rpa(k, w)
        dielectric_function = real_part + 1.0j * im_part
        return dielectric_function

    def _im_dielectric_rpa(self, k, w):
        """
        Imaginary part of the RPA dielectric function.
        Based on Arista and Brandt, Phys. Rev. A 29 (1984)

        Parameters:
            k (float): wave number in units of 1/m
            w (array): energy grid in units of 1/J

        Returns:
            array: imaginary part of the dielectric function, non-dimensional
        """

        kF = self.state.fermi_wave_number(self.state.free_electron_number_density)  # 1/m
        EF = self.state.fermi_energy(self.state.free_electron_number_density, ELECTRON_MASS)  # J
        vF = DIRAC_CONSTANT * kF / ELECTRON_MASS  # m/s
        u = w / (k * vF * DIRAC_CONSTANT)  # dimensionless
        z = k / (2 * kF)  # dimensionless

        beta = 1 / (BOLTZMANN_CONSTANT * self.state.electron_temperature)
        mu = self.state.chemical_potential_ichimaru(
            self.state.electron_temperature, self.state.free_electron_number_density, ELECTRON_MASS
        )  # Joule

        D = EF * beta  # dimensionless
        eta = mu * beta  # dimensionless
        chi02 = 1 / (PI * kF * BOHR_RADIUS)  # dimensionless
        theta = 1 / D

        if z < 0.1:
            prefactor = (
                2
                * ELECTRON_MASS**2
                * ELEMENTARY_CHARGE**2
                / VACUUM_PERMITTIVITY
                * DIRAC_CONSTANT
                * w
                / (DIRAC_CONSTANT * k) ** 3
            )
            exp_term = EF / (BOLTZMANN_CONSTANT * self.state.electron_temperature) * u**2 - eta**2
            im_part = prefactor / (1 + np.exp(exp_term))

        else:
            xpos = (u + z) ** 2  # Dimensionless
            xneg = (u - z) ** 2  # Dimensionless

            exp_neg = np.exp(eta - D * xneg)
            exp_pos = np.exp(eta - D * xpos)

            log_term = (1 + exp_neg) / (1 + exp_pos)  # [#]
            im_part = 1 * PI * chi02 / (8 * z**3) * theta * np.log(log_term)  # [#]
        return im_part

    def _real_dielectric_rpa(self, k, w):
        """
        Real part of the RPA dielectric function.
        Based on Arista and Brandt, Phys. Rev. A 29 (1984).
        The numerical integration is done using some mathematical tricks described in Ancarani and Jouin, Eur. Phys. J. Plus 131 (2016)

        Parameters:
            k (float): wave number in units of 1/m
            w (array): energy grid in units of 1/J

        Returns:
            array: real part of the dielectric function, non-dimensional
        """

        w_freq = w / DIRAC_CONSTANT

        kF = self.state.fermi_wave_number(self.state.free_electron_number_density)  # 1/m
        EF = self.state.fermi_energy(self.state.free_electron_number_density, ELECTRON_MASS)  # J
        TF = self.state.fermi_temperature(ELECTRON_MASS, self.state.free_electron_number_density)  # K
        vF = DIRAC_CONSTANT * kF / ELECTRON_MASS  # m/s

        u = w_freq / (vF * k)  # [#]
        kappa = k / (2 * kF)  # [#]

        t = self.state.electron_temperature * BOLTZMANN_CONSTANT / EF  # [#]
        t = self.state.electron_temperature / TF  # [#]

        theta = t  # []
        mu = self.state.chemical_potential_ichimaru(
            self.state.electron_temperature, self.state.free_electron_number_density, ELECTRON_MASS
        )  # [J]
        alpha = mu / EF

        def g_ancarni(lambda_val):
            return np.where(lambda_val < 0.0, -g_t(-lambda_val), g_t(lambda_val))

        def g_t(lambda_val, eps=1.0e-9):

            A = lambda_val**2 / t
            B = alpha / t

            def f_prime(X):
                y = A * X**2 - B
                s = np.empty_like(y)

                pos_mask = y >= 0.0
                neg_mask = ~pos_mask

                # y >= 0 branch
                exp_neg = np.exp(-y[pos_mask])
                s[pos_mask] = 1.0 / (1.0 + exp_neg)

                # y < 0 branch
                exp_pos = np.exp(y[neg_mask])
                s[neg_mask] = exp_pos / (1.0 + exp_pos)

                return -2.0 * A * X * (s * (1.0 - s))

            def integrand_u(u):
                X = np.tan(np.pi * u / 2)
                log_term = np.log(abs((X + 1) / (X - 1)))
                bracket_term = -X + 0.5 * (1 - X**2) * log_term
                dX_du = (np.pi / 2) * (1 / np.cos(np.pi * u / 2) ** 2)
                return f_prime(X) * bracket_term * dX_du

            res1, _ = integrate.quad_vec(integrand_u, 0, 0.5 - eps, limit=300)
            res2, _ = integrate.quad_vec(integrand_u, 0.5 + eps, 1, limit=300)

            return (lambda_val**2) * (res1 + res2)

        chi02 = 1 / (PI * kF * BOHR_RADIUS)  # [#]
        lambda_neg = u - kappa  # [#]
        lambda_pos = u + kappa  # [#]
        g_t_pos = g_ancarni(lambda_pos)
        g_t_neg = g_ancarni(lambda_neg)
        real_part = 1.0e0 + chi02 / (4 * kappa**3) * (g_t_pos - g_t_neg)

        return real_part

    def dandrea_fit(self, k, omega):
        """
        Fit to the electron-electron polarisation function based on: Dandrea et al., Phys. Rev. B 34 (1986)

        Parameters:
            k (float): wave number in units of 1/m
            w (array): energy grid in units of 1/J

        Returns:
            array: polarisation function, non-dimensional
        """
        # See Table 1, Dandrea et al., Phys. Rev. B 34 (1986), rows for each set of constants are given by comments
        _cJ = [3248.8, -691.47, -3202700, -4535.6, -462400]  # (A19)
        _cK = [-4.8780e00, +4.7325e02, -2.3375e03, +3.4831e02, +1.5173e03]  # (A20)
        _c2 = [-2.2800e-01, +4.2220e-01, -6.4660e-01, +7.0572e-01, +5.8820e00]  # (A21)
        _c4 = [-3.0375e00, +6.4646e01, +1.9608e01, -9.6978e01, +4.2366e02, -3.3101e02, +2.0833e01]  # (A22)
        _c6 = [-1.9000e-01, +3.6538e-01, -2.2575e00, +2.2942e01, -4.3492e01, +1.0640e02]  # (A23)
        _c8 = [
            -7.1316e00,
            +2.2725e01,
            +5.8092e01,
            -4.3602e02,
            -8.2651e02,
            +4.9129e03,
            +9.1000e-01,
            -6.4453e00,
            +1.22324e01,
        ]  # (A24)

        EF = self.state.fermi_energy(self.state.free_electron_number_density, ELECTRON_MASS)
        kF = self.state.fermi_wave_number(self.state.free_electron_number_density)
        Theta_e = self.state.alt_degeneracy_parameter(
            self.state.free_electron_number_density, self.state.electron_temperature, ELECTRON_MASS
        )
        sqrt_Theta_e = np.sqrt(Theta_e)
        Theta_2 = Theta_e**2
        Theta_3 = Theta_e**3

        # non-dimensional variables for integral
        q0 = 0.5 * k / kF
        w0 = 0.25 * omega / (EF * q0)  #  * DIRAC_CONSTANT
        w = w0 / sqrt_Theta_e
        beta = 1 / (BOLTZMANN_CONSTANT * self.state.electron_temperature)
        mue = self.state.chemical_potential_ichimaru(
            self.state.electron_temperature, self.state.free_electron_number_density, ELECTRON_MASS
        )
        eta = mue * beta
        q = q0 / np.sqrt(Theta_e)

        # normalized Fermi-Dirac integrals
        F_m0p5 = fdi(j=-0.5, eta=eta, normalize=True)
        F_p0p5 = fdi(j=+0.5, eta=eta, normalize=True)

        def phi_function(x):
            """
            See Eq. (4.6) - (4.8) in Dandrea et al., Phys. Rev. B 34 (1986), constants and functions are given in
            the Appendix.
            The original form of the phi function is given by Eq. (4.7), Eq. (4.8b) describes the fit implemented here.
            """
            # Eq. (A21)
            a2 = (_c2[0] + Theta_e) / (_c2[1] + _c2[2] * Theta_e ** _c2[3] + _c2[4] * Theta_e**2)
            # Eq. (A22)
            a4 = (1.0 + Theta_e * (_c4[0] + _c4[1] * Theta_e)) / (
                _c4[2] + Theta_e * (_c4[3] + Theta_e * (_c4[4] + Theta_e * (_c4[5] + Theta_e * _c4[6] * _c4[1])))
            )
            # Eq. (A23)
            a6 = (_c6[0] + Theta_e) / (
                _c6[1] + Theta_e * (_c6[2] + Theta_e * (_c6[3] + Theta_e * (_c6[4] + Theta_e * _c6[5])))
            )
            # Eq. (A24)
            a8 = (_c8[6] + Theta_e * (_c8[7] + Theta_e * _c8[8])) / (
                1.0
                + Theta_e
                * (
                    _c8[0]
                    + Theta_e
                    * (_c8[1] + Theta_e * (_c8[2] + Theta_e * (_c8[3] + Theta_e * (_c8[4] + Theta_e * _c8[5]))))
                )
            )

            # Eq. (A17) and (A19)
            J1 = 1 + _cJ[0] * Theta_2 + _cJ[1] * Theta_2**2 + _cJ[2] * Theta_e**7
            J2 = (
                1
                + (_cJ[0] - PI_SQR / 6) * Theta_2
                + _cJ[3] * Theta_2**2
                + _cJ[4] * Theta_e**6
                + (3 * SQRT_TWO * _cJ[2] / (4 * SQRT_PI)) * Theta_e**7.5
                + (3 * _cJ[2] / 4) * Theta_e**9
            )
            J = J1 / J2

            # Eq. (A18) and (A20)
            K1 = 1 + _cK[0] * Theta_2 + _cK[1] * Theta_2**2 + _cK[2] * Theta_e**7
            K2 = (
                1
                + (_cK[0] - 3 * PI_SQR / 4) * Theta_2
                + _cK[3] * Theta_2**2
                + _cK[4] * Theta_e**7
                - (7 * SQRT_TWO * _cK[2] / (8 * SQRT_PI)) * Theta_e**8.5
                - (3 * _cK[2] / 8) * Theta_e**10
            )
            K = K1 / K2
            # Un-normalized Fermi-Dirac integrals
            Im0p5 = SQRT_PI * F_m0p5
            Ip1p5 = fdi(j=+1.5, eta=eta, normalize=False)
            Ip2p5 = fdi(j=+2.5, eta=eta, normalize=False)

            # Eq. (A12)
            b10 = 1.5 * sqrt_Theta_e * Im0p5 * a8
            # Eq. (A13)
            b8 = 1.5 * sqrt_Theta_e * Im0p5 * a6 - 0.5 * Theta_e**2.5 * Ip1p5 * b10
            # Eq. (A14)
            b6 = (
                sqrt_Theta_e * 1.5 * Im0p5 * a4 - 0.5 * Theta_e**2.5 * Ip1p5 * b8 - 3 / 10 * Theta_e**3.5 * Ip2p5 * b10
            )
            # Eq. (A15)
            b2 = a2 + 2 * J / (3 * sqrt_Theta_e * Im0p5)
            # Eq. (A16)
            b4 = b2**2 - a2 * b2 + a4 + 2 * K / (15 * sqrt_Theta_e * Im0p5)

            x2 = x * x
            # Eq. (4.8b)
            u = 1 + a2 * x2 + a4 * x**4 + a6 * x**6 + a8 * x**8
            v = 1 + b2 * x2 + b4 * x**4 + b6 * x**6 + b8 * x**8 + b10 * x**10
            # Eq. (4.8a)
            return x * u / v

        exp_arg_pos = eta - (w + q) ** 2
        exp_arg_neg = eta - (w - q) ** 2

        imag_part = SQRT_PI * (log1pexp(exp_arg_pos) - log1pexp(exp_arg_neg))
        # Eq. (4.6)
        delta_F = phi_function(w0 + q0) - phi_function(w0 - q0)
        real_part = -2.0 * F_m0p5 * delta_F / sqrt_Theta_e

        prefactor = self.state.free_electron_number_density / (BOLTZMANN_CONSTANT * self.state.electron_temperature)
        pol_func = prefactor * (real_part + 1.0j * imag_part) / (4.0 * F_p0p5 * q)

        return pol_func

    def _ziman_ei_collision_frequency(self):
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

    def _born_ei_collision_frequency_old(self, k, w, lfc):
        k_temp = np.linspace(1.0e-3, 1.0e2, 1000) / BOHR_RADIUS
        Siik = OCPStaticStructureFactor(state=self.state).get_ii_static_structure_factor(k=k_temp, sf_model="HNC")

        Zi = self.state.charge_state
        mi = self.state.atomic_mass
        ni = self.state.ion_number_density
        ne = self.state.free_electron_number_density
        w_plasma = self.state.plasma_frequency(charge=Zi, number_density=ni, mass=mi)
        w_freq = w / DIRAC_CONSTANT

        def integral(q):
            epsilon0 = self.dandrea_fit(k=q, omega=0)
            epsilon = self.dandrea_fit(k=q, omega=w)
            Sii = np.interp(q, xp=k_temp, fp=Siik)
            return 1 / TWO_PI**3 * Sii / epsilon0 * (epsilon - epsilon0)

        prefactor = FOUR_THIRDS_PI * w_plasma / w_freq * mi / ELECTRON_MASS * 1 / ne * 1.0j
        int_term, _ = integrate.quad_vec(integral, 0, 1.0e14)
        return prefactor * int_term
        # interp_Sk = np.interp(k_temp, Siik, fill_value="extrapolate")

    def _born_ei_collision_frequency(self, k, w, lfc):

        ni = self.state.ion_number_density
        mi = self.state.atomic_mass
        me = ELECTRON_MASS
        ne = self.state.free_electron_number_density
        Zi = self.state.charge_state
        wpi = self.state.plasma_frequency(charge=Zi, number_density=ni, mass=mi)

        w_freq = w / DIRAC_CONSTANT

        prefactor = VACUUM_PERMITTIVITY * ni / (6 * PI**2 * ELEMENTARY_CHARGE_SQR * me * ne)

        k_temp = np.linspace(1.0e-3, 1.0e2, 1000) / BOHR_RADIUS
        Siik = OCPStaticStructureFactor(
            state=self.state, verbose=False, mix_fraction=0.99, max_iterations=10000
        ).get_ii_static_structure_factor(k=k_temp, sf_model="HNC")

        def integrand(q):
            Si = np.interp(q, k_temp, Siik)

            Vee = ELEMENTARY_CHARGE**2 / (VACUUM_PERMITTIVITY * q**2)  # []
            # q /= 1.0e-10
            epsilon = 1 - Vee * self.dandrea_fit(q, w)
            epsilon0 = 1 - Vee * self.dandrea_fit(q, 0)
            VeiS = Vee / epsilon0
            I = -1.0j * q**6 * VeiS**2 * Si / w_freq * (epsilon - epsilon0)
            return I

        integral, _ = integrate.quad_vec(integrand, 0, 1.0e10)
        return integral * prefactor

    def _full_born_ei_collision_frequency(self, k, w, lfc):
        w_freq = w / DIRAC_CONSTANT
        omega_p = self.state.plasma_frequency(
            self.state.charge_state, self.state.ion_number_density, self.state.atomic_mass
        )
        col0 = 1.0 * omega_p**2 * self.state.atomic_mass / (2 * w_freq * ELECTRON_MASS)

        ff_kernel = FreeFreeDSF(state=self.state)

        # def int_function(k, w):
        #     Uee = 4 * np.pi * COULOMB_CONSTANT * ELEMENTARY_CHARGE**2 / k**2
        #     epsilon_k_omega = 1 - Uee * ff_kernel.dandrea_fit(k=k, w=0)
        #     epsilon_k_0 = 1 - Uee * ff_kernel.dandrea_fit(k=k, w=0)
        #     return -1.0j * (epsilon_k_omega - epsilon_k_0) / (epsilon_k_0**2)

        temp_k = np.linspace(0.01, 10, 100) / BOHR_RADIUS
        Siis = OCPStaticStructureFactor(self.state).get_ii_static_structure_factor(k=temp_k)

        # def real_integrand(u):
        #     x = np.tan(HALF_PI * u)
        #     j = np.arctan(u) / HALF_PI

        #     kF = self.state.fermi_wave_number(self.state.free_electron_number_density)
        #     k = kF * x

        #     Uee = ELEMENTARY_CHARGE**2 / (VACUUM_PERMITTIVITY * k**2)  # []
        #     epsilon_k_omega = 1 - Uee * ff_kernel.dandrea_fit(k=k, omega=w)
        #     epsilon_k_0 = 1 - Uee * ff_kernel.dandrea_fit(k=k, omega=0)
        #     F = -1.0j * (epsilon_k_omega - epsilon_k_0) / (epsilon_k_0**2)

        #     Sii = np.interp(x=k, xp=temp_k, fp=Siis)
        #     return j * x**2 * Sii * F.real

        def integrand(u):
            x = np.tan(HALF_PI * u)
            j = np.arctan(u) / HALF_PI

            kF = self.state.fermi_wave_number(self.state.free_electron_number_density)
            k = kF * x

            Uee = ELEMENTARY_CHARGE**2 / (VACUUM_PERMITTIVITY * k**2)  # []
            epsilon_k_omega = 1 - Uee * ff_kernel.dandrea_fit(k=k, omega=w)
            epsilon_k_0 = 1 - Uee * ff_kernel.dandrea_fit(k=k, omega=0)
            F = -1.0j * (epsilon_k_omega - epsilon_k_0) / (epsilon_k_0**2)

            Sii = np.interp(x=k, xp=temp_k, fp=Siis)
            return j * x**2 * Sii * F

        # def imag_integrand(u):
        #     x = np.tan(HALF_PI * u)
        #     j = np.arctan(u) / HALF_PI

        #     kF = self.state.fermi_wave_number(self.state.free_electron_number_density)
        #     k = kF * x

        #     Uee = 4 * np.pi * COULOMB_CONSTANT * ELEMENTARY_CHARGE**2 / k**2
        #     epsilon_k_omega = 1 - Uee * ff_kernel.dandrea_fit(k=k, omega=w)
        #     epsilon_k_0 = 1 - Uee * ff_kernel.dandrea_fit(k=k, omega=0)
        #     F = -1.0j * (epsilon_k_omega - epsilon_k_0) / (epsilon_k_0**2)

        #     Sii = np.interp(x=k, xp=temp_k, fp=Siis)
        #     return j * x**2 * Sii * F.imag

        # coll_freq_real = -col0 * integrate.quad_vec(integrand, 0, 1)[0]
        # coll_freq_imag = col0 * integrate.quad_vec(integrand, 0, 1)[0]

        coll_freq = col0 * integrate.quad_vec(integrand, 0, 1)[0]

        return coll_freq  # coll_freq_real + 1.0j * coll_freq_imag

    def _real_dielectric_mermin(self, k, w, mu1, mu2):
        wtilde = w - mu2
        kappa = wtilde * ELECTRON_MASS / (DIRAC_CONSTANT * k)
        delta = ELECTRON_MASS * mu1 / (DIRAC_CONSTANT * k)
        mue = self.state.chemical_potential_ichimaru(
            self.state.electron_temperature, self.state.free_electron_number_density, ELECTRON_MASS
        )
        # kF = self.state.fermi_wave_number(self.state.free_electron_number_density)
        beta = 1 / (BOLTZMANN_CONSTANT * self.state.electron_temperature)
        k_half = k / 2

        def real_integral(q):
            KEq = DIRAC_CONSTANT_SQR * q**2 / (2 * ELECTRON_MASS)
            fq = 1 / (np.exp((KEq - mue) * beta) + 1)
            numerator = (delta**2 + (kappa - k_half - q) ** 2) * (delta**2 + (kappa + k_half + q))
            denominator = (delta**2 + (kappa - k_half + q) ** 2) * (delta**2 + (kappa + k_half - q) ** 2)
            log_term = np.log(numerator / denominator)
            I = 1 / TWO_PI**3 * q * fq * log_term
            return I

        # units: kg * C^2 / (C^2 * kg ^(-1) * m ^ (-3) * s^2 * (J *s) ^ 2 * m^(-3))
        prefactor = TWO_PI * ELECTRON_MASS * ELEMENTARY_CHARGE_SQR / (VACUUM_PERMITTIVITY * DIRAC_CONSTANT_SQR * k**3)
        int_term, _ = integrate.quad_vec(real_integral, 0, 1.12)
        return 1 + prefactor * int_term

    def _im_dielectric_mermin(self, k, w, mu1, mu2):
        # TODO(HB): get this working

        wtilde = w - mu2
        kappa = wtilde * ELECTRON_MASS / (DIRAC_CONSTANT * k)
        delta = ELECTRON_MASS * mu1 / (DIRAC_CONSTANT * k)
        mue = self.state.chemical_potential_ichimaru(
            self.state.electron_temperature, self.state.free_electron_number_density, ELECTRON_MASS
        )
        beta = 1 / (BOLTZMANN_CONSTANT * self.state.electron_temperature)
        w0 = 1
        k_half = k / 2

        def imag_integral(q):
            KEq = DIRAC_CONSTANT_SQR * q**2 / (2 * ELECTRON_MASS)
            fq = 1 / (np.exp((KEq - mue) * beta) + 1)
            I = (
                1
                / TWO_PI**3
                * fq
                * q
                * (
                    np.arctan((kappa - k_half - q) / delta)
                    + np.arctan((kappa + k_half + q) / delta)
                    - np.arctan((kappa - k_half + q) / delta)
                    - np.arctan((kappa + k_half - q) / delta)
                )
            )
            return I  # * dx_dq

        prefactor = (
            -FOUR_PI * ELECTRON_MASS * ELEMENTARY_CHARGE_SQR / (VACUUM_PERMITTIVITY * DIRAC_CONSTANT_SQR * k**3)
        )
        int_term, _ = integrate.quad_vec(imag_integral, 0, 1.0e12)
        imag_part = prefactor * int_term * w0

        return imag_part

    def mermin_dielectric_function(
        self, k, w, lfc=0, collision_frequency_model="BORN", input_collision_frequency=None
    ):
        """
        Mermin dielectric function using the relaxed time approximation.

        Parameters:
            k (float): wave number in units of 1/m
            w (array): energy grid in units of 1/J
            lfc (float): local field correction, non-dimensional
            collision_frequency_model (str): model option for the collision frequency
            input_collision_frequency (float): user-defined static collision frequency in units of plasma frequency.

        Returns:
            array: dielectric function, non-dimensional
        """
        wp = self.state.plasma_frequency(-1, self.state.free_electron_number_density, ELECTRON_MASS)
        w_freq = w / DIRAC_CONSTANT
        if input_collision_frequency is not None:
            mu_ei = input_collision_frequency * wp * (1.0 + 0.0j)
        else:
            mu_ei = self.get_collision_frequency(k=k, w=w, lfc=lfc, model=collision_frequency_model)  # * wp

        Vee = 4 * np.pi * COULOMB_CONSTANT * ELEMENTARY_CHARGE**2 / k**2
        mu_ei_energy = mu_ei * DIRAC_CONSTANT
        ratio = 1.0j * mu_ei / w_freq
        factor = 1 + ratio
        rpa_dielectric = 1 - self.dandrea_fit(k=k, omega=w + 1.0j * mu_ei_energy) * Vee
        rpa_dielectric0 = 1 - self.dandrea_fit(k=k, omega=0) * Vee
        mermin_dielectric = 1 + (factor * (rpa_dielectric - 1)) / (
            1 + ratio * (rpa_dielectric - 1) / (rpa_dielectric0 - 1)
        )
        return mermin_dielectric

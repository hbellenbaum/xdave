from .constants import *
from .unit_conversions import *
from .maths import log1pexp
from .plasma_state import PlasmaState
from .fermi_integrals import fdi

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

    def get_dsf(self, k, w, lfc, model="NUMERICAL_RPA"):
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

        # Call the susceptibility function for a given model
        chi0 = self.susceptibility_function(k=k, w=w, model=model)

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

        return S_EG_LFC

    def dielectric_function(self, k, w, model):
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
            dielectric_func = self.mermin_dielectric_function_old(k=k, w=w)
        else:
            dielectric_func = self.rpa_numerical_dielectric_func(k=k, w=w)
            warnings.warn(f"Model {model} for the free-free component not recognized. Overwriting using NUMERICAL.")

        return dielectric_func

    def susceptibility_function(self, k, w, model):
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
            dielectric_func = self.mermin_dielectric_function(k=k, w=w)  # * potential_func
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

    def _real_dielectric_mermin(self, k, w, mu1, mu2):
        wtilde = w - mu2
        kappa = wtilde * ELECTRON_MASS / (DIRAC_CONSTANT * k)
        delta = ELECTRON_MASS * mu1 / (DIRAC_CONSTANT * k)
        mue = self.state.chemical_potential_ichimaru(
            self.state.electron_temperature, self.state.free_electron_number_density, ELECTRON_MASS
        )
        # kF = self.state.fermi_wave_number(self.state.free_electron_number_density)
        beta = 1 / (BOLTZMANN_CONSTANT * self.state.electron_temperature)
        KE = DIRAC_CONSTANT**2 * k**2 / (2 * ELECTRON_MASS)
        w0 = 1

        def real_integral(q):
            q /= w0
            x = np.tan(HALF_PI * q)
            dx_dq = HALF_PI * (1 / np.cos(HALF_PI * q) ** 2)
            fq = 1 / (np.exp((KE - mue) * beta) + 1)
            numerator = (delta**2 + (kappa - k / 2 - x) ** 2) * (delta**2 + (kappa + k / 2 + x))
            denominator = (delta**2 + (kappa - k / 2 + x) ** 2) * (delta**2 + (kappa + k / 2 - x) ** 2)
            log_term = np.log(numerator / denominator)
            I = 1 / TWO_PI**3 * x * fq * log_term * dx_dq
            return I

        # units: kg * C^2 / (C^2 * kg ^(-1) * m ^ (-3) * s^2 * (J *s) ^ 2 * m^(-3))
        prefactor = TWO_PI * ELECTRON_MASS * ELEMENTARY_CHARGE_SQR / (VACUUM_PERMITTIVITY * DIRAC_CONSTANT_SQR * k**3)
        int_term, _ = integrate.quad_vec(real_integral, 0, 1.0)
        return 1 + prefactor * int_term * w0

    def _im_dielectric_mermin(self, k, w, mu1, mu2):
        # TODO(HB): get this working

        wtilde = w - mu2
        kappa = wtilde * ELECTRON_MASS / (DIRAC_CONSTANT * k)
        delta = ELECTRON_MASS * mu1 / (DIRAC_CONSTANT * k)
        mue = self.state.chemical_potential_ichimaru(
            self.state.electron_temperature, self.state.free_electron_number_density, ELECTRON_MASS
        )
        beta = 1 / (BOLTZMANN_CONSTANT * self.state.electron_temperature)
        KE = DIRAC_CONSTANT**2 * k**2 / (2 * ELECTRON_MASS)
        w0 = 1

        def imag_integral(q):
            q /= w0
            x = np.tan(HALF_PI * q)
            dx_dq = HALF_PI * (1 / np.cos(HALF_PI * q) ** 2)
            fq = 1 / (np.exp((KE - mue) * beta) + 1)
            I = (
                1
                / TWO_PI**3
                * fq
                * x
                * (
                    np.arctan((kappa - k / 2 - x) / delta)
                    + np.arctan((kappa + k / 2 + x) / delta)
                    - np.arctan((kappa - k / 2 + x) / delta)
                    - np.arctan((kappa + k / 2 - x) / delta)
                )
            )
            return I * dx_dq

        prefactor = (
            -FOUR_PI * ELECTRON_MASS * ELEMENTARY_CHARGE_SQR / (VACUUM_PERMITTIVITY * DIRAC_CONSTANT_SQR * k**3)
        )
        # prefactor = (
        #     2
        #     * ELECTRON_MASS**2
        #     * ELEMENTARY_CHARGE**2
        #     / VACUUM_PERMITTIVITY
        #     * DIRAC_CONSTANT
        #     * w
        #     / (DIRAC_CONSTANT * k) ** 3
        # )
        int_term, _ = integrate.quad_vec(imag_integral, 0, 1.0)
        imag_part = prefactor * int_term * w0

        return imag_part

    def mermin_dielectric_function_old(
        self, k, w, lfc=0, collision_frequency_model="BORN", input_collision_frequency=None
    ):
        """
        Mermin dielectric function using the relaxed time approximation.

        Parameters:
            k (float): wave number in units of 1/m
            w (array): energy grid in units of 1/J
            lfc (float): local field correction, non-dimensional
            collision_frequency_model (str): model option for the collision frequency
            input_collision_frequency (float): user-defined static collision frequency in 1/s

        Returns:
            array: polarisation, non-dimensional
        """
        wp = self.state.plasma_frequency(
            -1, self.state.free_electron_number_density, ELECTRON_MASS
        )  # * DIRAC_CONSTANT
        w_freq = w / DIRAC_CONSTANT
        mu_ei = 1.0e-2 * wp * (1.0 + 0.0j)
        mu1 = mu_ei.real
        mu2 = mu_ei.imag
        real_part = self._real_dielectric_mermin(k=k, w=w_freq, mu1=mu1, mu2=mu2)
        imag_part = self._im_dielectric_mermin(k=k, w=w_freq, mu1=mu1, mu2=mu2)
        dielectric_function = real_part + 1.0j * imag_part

        dielectric_rpa0 = self.dielectric_function(k=k, w=0, model="DANDREA_FIT")
        mermin_dielectric = 1 + (1 + 1.0j * mu_ei / w_freq) * (dielectric_function - 1) / (
            1 + 1.0j * mu_ei / w_freq * (dielectric_function - 1) / (dielectric_rpa0 - 1)
        )
        return mermin_dielectric

    def mermin_dielectric_function(
        self, k, w, lfc=0, collision_frequency_model="BORN", input_collision_frequency=None
    ):
        nu = 1.0e-2  # Ha, will need to replace this using the Born model e.g.
        kb = 3.166811563e-6  # Boltzmann constant in units of Ha / K
        me = 1.0  # electron mass in atomic units
        Te = self.state.electron_temperature  # * eV_TO_Ha / kb
        ne = self.state.free_electron_number_density * 1.0e-2 / BOHR_RADIUS**3
        q = k * BOHR_RADIUS
        mue = (
            self.state.chemical_potential_ichimaru(
                self.state.electron_temperature, self.state.free_electron_number_density, ELECTRON_MASS
            )
            * J_TO_Ha
        )

        w_vals = np.array(w) * J_TO_Ha

        eps = 1.0e-6  # integration error

        renu = nu.real
        imnu = nu.imag

        Uee = 8.0 * PI / q**2
        w0 = 0.0
        renu_saved = renu
        imnu_saved = imnu
        # temporarily set renu -> small, imnu -> 0
        renu_tmp = 1e-8
        imnu_tmp = 0.0

        polem = np.abs(0.5 * q - 0.5 * me / q * (w0 - imnu_tmp))
        polep = 0.5 * q + 0.5 * me / q * (w0 - imnu_tmp)
        klim = me * (kb * Te * np.log(1.0 / 1e-4 - 1.0) + mue)

        def repi_int(ks, Te, mu, q, w, imnu, renu):
            """Finite-k real-part integrand"""
            k = ks  # [1/a0]
            beta = 1.0 / (kb * Te)  # [1/Ha]
            fk = 1.0 / (np.exp(beta * (k * k / me - mu)) + 1.0)  # [ ]
            sa = 0.5 * me / q  # [ 1/a0]

            ln1 = np.log((k - 0.5 * q - sa * (w - imnu)) ** 2 + (sa * sa) * (renu * renu))
            ln2 = np.log((k - 0.5 * q + sa * (w - imnu)) ** 2 + (sa * sa) * (renu * renu))
            repi_int1 = 0.5 * k * fk * (ln1 + ln2)

            kneg = -k
            ln1 = np.log((kneg - 0.5 * q - sa * (w - imnu)) ** 2 + (sa * sa) * (renu * renu))
            ln2 = np.log((kneg - 0.5 * q + sa * (w - imnu)) ** 2 + (sa * sa) * (renu * renu))
            repi_int2 = 0.5 * kneg * fk * (ln1 + ln2)

            if abs(repi_int2) > 0.0:
                ratio_test = abs(abs(repi_int1 / repi_int2) - 1.0)
            else:
                ratio_test = np.inf

            if ratio_test < 1e-10:
                return 0.0
            else:
                return repi_int1 + repi_int2

        def repi_int_inf(ks, Te, mu, q, w, imnu, renu):
            """Infinite-domain mapped real-part integrand via k = tan(ks)"""
            k = np.tan(ks)
            beta = 1.0 / (kb * Te)
            fk = 1.0 / (np.exp(beta * (k * k / me - mu)) + 1.0)
            sa = 0.5 * me / q

            ln1 = np.log((k - 0.5 * q - sa * (w - imnu)) ** 2 + (sa * sa) * (renu * renu))
            ln2 = np.log((k - 0.5 * q + sa * (w - imnu)) ** 2 + (sa * sa) * (renu * renu))
            val = 0.5 * k * fk * (ln1 + ln2)

            kneg = -k
            ln1 = np.log((kneg - 0.5 * q - sa * (w - imnu)) ** 2 + (sa * sa) * (renu * renu))
            ln2 = np.log((kneg - 0.5 * q + sa * (w - imnu)) ** 2 + (sa * sa) * (renu * renu))
            val += 0.5 * kneg * fk * (ln1 + ln2)

            return val / (np.cos(ks) ** 2)

        def impi_int(ks, Te, mu, q, w, imnu, renu):
            """Finite-k imaginary-part integrand"""
            k = ks
            beta = 1.0 / (kb * Te)
            fk = 1.0 / (np.exp(beta * (k * k / me - mu)) + 1.0)
            sa = 0.5 * me / q

            den1 = k - 0.5 * q - sa * (w - imnu)
            if den1 < 0.0:
                at1 = -np.arctan(sa * renu / den1)
            else:
                at1 = -np.arctan(sa * renu / den1) + PI

            den2 = k - 0.5 * q + sa * (w - imnu)
            if den2 < 0.0:
                at2 = np.arctan(sa * renu / den2) + PI
            else:
                at2 = np.arctan(sa * renu / den2)

            impi_int1 = k * fk * (at1 + at2)

            kneg = -k
            den1 = kneg - 0.5 * q - sa * (w - imnu)
            if den1 < 0.0:
                at1 = -np.arctan(sa * renu / den1)
            else:
                at1 = -np.arctan(sa * renu / den1) + PI

            den2 = kneg - 0.5 * q + sa * (w - imnu)
            if den2 < 0.0:
                at2 = np.arctan(sa * renu / den2) + PI
            else:
                at2 = np.arctan(sa * renu / den2)

            impi_int2 = kneg * fk * (at1 + at2)

            if abs(impi_int2) > 0.0:
                ratio_test = abs(abs(impi_int1 / impi_int2) - 1.0)
            else:
                ratio_test = np.inf

            if ratio_test < 1e-10:
                return 0.0
            else:
                return impi_int1 + impi_int2

        def impi_int_inf(ks, Te, mu, q, w, imnu, renu):
            """Infinite-domain mapped imaginary-part integrand via k = tan(ks)"""
            k = np.tan(ks)
            beta = 1.0 / (kb * Te)
            fk = 1.0 / (np.exp(beta * (k * k / me - mu)) + 1.0)
            sa = 0.5 * me / q

            den1 = k - 0.5 * q - sa * (w - imnu)
            if den1 < 0.0:
                at1 = -np.arctan(sa * renu / den1)
            else:
                at1 = -np.arctan(sa * renu / den1) + PI

            den2 = k - 0.5 * q + sa * (w - imnu)
            if den2 < 0.0:
                at2 = np.arctan(sa * renu / den2) + PI
            else:
                at2 = np.arctan(sa * renu / den2)

            val = k * fk * (at1 + at2)

            kneg = -k
            den1 = kneg - 0.5 * q - sa * (w - imnu)
            if den1 < 0.0:
                at1 = -np.arctan(sa * renu / den1)
            else:
                at1 = -np.arctan(sa * renu / den1) + PI

            den2 = kneg - 0.5 * q + sa * (w - imnu)
            if den2 < 0.0:
                at2 = np.arctan(sa * renu / den2) + PI
            else:
                at2 = np.arctan(sa * renu / den2)

            val += kneg * fk * (at1 + at2)
            return val / (np.cos(ks) ** 2)

        # ---------------------------
        # Integrators: rcauch and rgauss using scipy.integrate.quad
        # ---------------------------

        def rcauch(func, a, b, c, eps, args=()):
            """
            Cauchy-type integrator: splits the integral at the pole location c and integrates
            from [a, c-eps] and [c+eps, b]. 'func' should be callable with signature func(k, *args).
            """
            if not (a < c < b):
                # no pole inside interval, integrate directly
                val, err = integrate.quad(lambda x: func(x, *args), a, b, limit=200)
                return val
            left, el = integrate.quad(lambda x: func(x, *args), a, c - eps, limit=200)
            right, er = integrate.quad(lambda x: func(x, *args), c + eps, b, limit=200)
            return left + right

        def rgauss(func, a, b, eps, args=()):
            """
            Gaussian-mapped integrator (here: simple integrate.quad over [a,b]) where func accepts (ks, *args)
            """
            val, err = integrate.quad(lambda x: func(x, *args), a, b, limit=200)
            return val

        if (polep < klim) and (polep != polem):
            kum = 0.5 * (polep + klim)
            k0 = 0.5 * (polem + polep)
            repik0 = rcauch(
                lambda k, Te_, mu_, q_, w_, imnu_, renu_: repi_int(k, Te_, mu_, q_, w_, imnu_, renu_),
                0.0,
                k0,
                polem,
                eps,
                args=(Te, mue, q, w0, imnu_tmp, renu_tmp),
            )
            repik0 += rcauch(
                lambda k, Te_, mu_, q_, w_, imnu_, renu_: repi_int(k, Te_, mu_, q_, w_, imnu_, renu_),
                k0,
                kum,
                polep,
                eps,
                args=(Te, mue, q, w0, imnu_tmp, renu_tmp),
            )
            kum_ang = np.arctan(kum)
            klim_ang = np.arctan(klim)
            repik0 += rgauss(
                lambda ks, Te_, mu_, q_, w_, imnu_, renu_: repi_int_inf(ks, Te_, mu_, q_, w_, imnu_, renu_),
                kum_ang,
                klim_ang,
                eps,
                args=(Te, mue, q, w0, imnu_tmp, renu_tmp),
            )
        else:
            if polem < klim:
                kum = 0.5 * (polem + klim)
                repik0 = rcauch(
                    lambda k, Te_, mu_, q_, w_, imnu_, renu_: repi_int(k, Te_, mu_, q_, w_, imnu_, renu_),
                    0.0,
                    kum,
                    polem,
                    eps,
                    args=(Te, mue, q, w0, imnu_tmp, renu_tmp),
                )
                kum_ang = np.arctan(kum)
                klim_ang = np.arctan(klim)
                repik0 += rgauss(
                    lambda ks, Te_, mu_, q_, w_, imnu_, renu_: repi_int_inf(ks, Te_, mu_, q_, w_, imnu_, renu_),
                    kum_ang,
                    klim_ang,
                    eps,
                    args=(Te, mue, q, w0, imnu_tmp, renu_tmp),
                )
            else:
                klim_ang = np.arctan(klim)
                repik0 = rgauss(
                    lambda ks, Te_, mu_, q_, w_, imnu_, renu_: repi_int_inf(ks, Te_, mu_, q_, w_, imnu_, renu_),
                    0.0,
                    klim_ang,
                    eps,
                    args=(Te, mue, q, w0, imnu_tmp, renu_tmp),
                )

        repik0 = repik0 * 0.25 * me / (PI * PI * q)

        # restore renu/imnu
        renu = renu_saved
        imnu = imnu_saved
        real_part = np.zeros_like(w_vals)
        imag_part = np.zeros_like(w_vals)

        # Main loop over frequencies
        # for w in w_vals:
        for i in range(0, len(w_vals)):
            w = w_vals[i]
            polem = abs(0.5 * q - 0.5 * me / q * (w - imnu))
            polep = 0.5 * q + 0.5 * me / q * (w - imnu)
            klim = np.sqrt(me * (kb * Te * np.log(1.0 / 1e-4 - 1.0) + mue))

            # compute repi and impi with same branching
            if (polep < klim) and (polem != polep):
                kum = 0.5 * (polep + klim)
                k0 = 0.5 * (polem + polep)

                impi = rcauch(
                    lambda k, Te_, mu_, q_, w_, imnu_, renu_: impi_int(k, Te_, mu_, q_, w_, imnu_, renu_),
                    0.0,
                    k0,
                    polem,
                    eps,
                    args=(Te, mue, q, w, imnu, renu),
                )
                impi += rcauch(
                    lambda k, Te_, mu_, q_, w_, imnu_, renu_: impi_int(k, Te_, mu_, q_, w_, imnu_, renu_),
                    k0,
                    kum,
                    polep,
                    eps,
                    args=(Te, mue, q, w, imnu, renu),
                )

                repi = rcauch(
                    lambda k, Te_, mu_, q_, w_, imnu_, renu_: repi_int(k, Te_, mu_, q_, w_, imnu_, renu_),
                    0.0,
                    k0,
                    polem,
                    eps,
                    args=(Te, mue, q, w, imnu, renu),
                )
                repi += rcauch(
                    lambda k, Te_, mu_, q_, w_, imnu_, renu_: repi_int(k, Te_, mu_, q_, w_, imnu_, renu_),
                    k0,
                    kum,
                    polep,
                    eps,
                    args=(Te, mue, q, w, imnu, renu),
                )

                kum_ang = np.arctan(kum)
                klim_ang = np.arctan(klim)
                impi += rgauss(
                    lambda ks, Te_, mu_, q_, w_, imnu_, renu_: impi_int_inf(ks, Te_, mu_, q_, w_, imnu_, renu_),
                    kum_ang,
                    klim_ang,
                    eps,
                    args=(Te, mue, q, w, imnu, renu),
                )
                repi += rgauss(
                    lambda ks, Te_, mu_, q_, w_, imnu_, renu_: repi_int_inf(ks, Te_, mu_, q_, w_, imnu_, renu_),
                    kum_ang,
                    klim_ang,
                    eps,
                    args=(Te, mue, q, w, imnu, renu),
                )

            else:
                if polem < klim:
                    kum = 0.5 * (polem + klim)

                    impi = rcauch(
                        lambda k, Te_, mu_, q_, w_, imnu_, renu_: impi_int(k, Te_, mu_, q_, w_, imnu_, renu_),
                        0.0,
                        kum,
                        polem,
                        eps,
                        args=(Te, mue, q, w, imnu, renu),
                    )
                    repi = rcauch(
                        lambda k, Te_, mu_, q_, w_, imnu_, renu_: repi_int(k, Te_, mu_, q_, w_, imnu_, renu_),
                        0.0,
                        kum,
                        polem,
                        eps,
                        args=(Te, mue, q, w, imnu, renu),
                    )

                    kum_ang = np.arctan(kum)
                    klim_ang = np.arctan(klim)

                    impi += rgauss(
                        lambda ks, Te_, mu_, q_, w_, imnu_, renu_: impi_int_inf(ks, Te_, mu_, q_, w_, imnu_, renu_),
                        kum_ang,
                        klim_ang,
                        eps,
                        args=(Te, mue, q, w, imnu, renu),
                    )
                    repi += rgauss(
                        lambda ks, Te_, mu_, q_, w_, imnu_, renu_: repi_int_inf(ks, Te_, mu_, q_, w_, imnu_, renu_),
                        kum_ang,
                        klim_ang,
                        eps,
                        args=(Te, mue, q, w, imnu, renu),
                    )
                else:
                    klim_ang = np.arctan(klim)
                    impi = rgauss(
                        lambda ks, Te_, mu_, q_, w_, imnu_, renu_: impi_int_inf(ks, Te_, mu_, q_, w_, imnu_, renu_),
                        0.0,
                        klim_ang,
                        eps,
                        args=(Te, mue, q, w, imnu, renu),
                    )
                    repi = rgauss(
                        lambda ks, Te_, mu_, q_, w_, imnu_, renu_: repi_int_inf(ks, Te_, mu_, q_, w_, imnu_, renu_),
                        0.0,
                        klim_ang,
                        eps,
                        args=(Te, mue, q, w, imnu, renu),
                    )

            repi = repi * 0.25 * me / (PI * PI * q)
            impi = impi * 0.25 * me / (PI * PI * q)

            # Mermin formulas
            a = w * repik0 - renu * impi - imnu * repi
            b = renu * repi - imnu * impi

            repim = a * ((w - imnu) * repi - renu * impi) + b * ((w - imnu) * impi + renu * repi)
            repim = repim * repik0 * (1 - lfc) / (a * a + b * b)

            impim = a * ((w - imnu) * impi + renu * repi) - b * ((w - imnu) * repi - renu * impi)
            impim = impim * repik0 * (1 - lfc) / (a * a + b * b)

            reeps = 1.0 - Uee * repim
            imeps = -Uee * impim

            a2 = (
                w * repik0
                - w * Uee * (1 - lfc) * repik0 * repi
                - (1.0 - Uee * (1 - lfc) * repik0) * (renu * impi + imnu * repi)
            )
            b2 = -w * Uee * (1 - lfc) * repik0 * impi + (1.0 - Uee * (1 - lfc) * repik0) * (renu * repi - imnu * impi)

            iml = a2 * ((w - imnu) * impi + renu * repi) - b2 * ((w - imnu) * repi - renu * impi)
            iml = iml * repik0 / (a2 * a2 + b2 * b2)
            iminveps = Uee * iml

            real_part[i] = reeps
            imag_part[i] = imeps

        dielectric_function = real_part + 1.0j * imag_part
        return dielectric_function

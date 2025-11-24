from .constants import *
from .unit_conversions import *
from .maths import log1pexp
from .plasma_state import PlasmaState


from scipy import integrate

# from collision_frequency import CollisionFrequency

# from plasmapy.formulary.mathematics import Fermi_integral as fdi
from fermi_integrals import fdi

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

        # NOTE(HB): not sure I need this anymore since I'm controlling this from the main kernel
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
            dielectric_func = self.mermin_dielectric_function(k=k, w=w)
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
        _cJ = [+3.2488e03, -6.9147e02, -3.2027e06, -4.5356e03, -4.6240e05]
        _cK = [-4.8780e00, +4.7325e02, -2.3375e03, +3.4831e02, +1.5173e03]
        _c2 = [-2.2800e-01, +4.2220e-01, -6.4660e-01, +7.0572e-01, +5.8820e00]
        _c4 = [-3.0375e00, +6.4646e01, +1.9608e01, -9.6978e01, +4.2366e02, -3.3101e02, +2.0833e01]
        _c6 = [-1.9000e-01, +3.6538e-01, -2.2575e00, +2.2942e01, -4.3492e01, +1.0640e02]
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
        ]

        EF = self.state.fermi_energy(self.state.free_electron_number_density, ELECTRON_MASS)
        kF = self.state.fermi_wave_number(self.state.free_electron_number_density)
        Theta_e = self.state.alt_degeneracy_parameter(
            self.state.free_electron_number_density, self.state.electron_temperature, ELECTRON_MASS
        )

        sqrt_Theta_e = np.sqrt(Theta_e)

        q0 = 0.5 * k / kF
        w0 = 0.25 * omega / (EF * q0)  #  * DIRAC_CONSTANT
        w = w0 / sqrt_Theta_e
        eta = self.state.chemical_potential_ichimaru(
            self.state.electron_temperature, self.state.free_electron_number_density, ELECTRON_MASS
        )
        q = q0 / np.sqrt(Theta_e)

        F_m0p5 = fdi(j=-0.5, eta=eta, normalize=True)
        F_p0p5 = fdi(j=+0.5, eta=eta, normalize=True)

        def phi_function(x):
            a2 = (_c2[0] + Theta_e) / (_c2[1] + _c2[2] * Theta_e ** _c2[3] + _c2[4] * Theta_e**2)
            a4 = (1.0 + Theta_e * (_c4[0] + _c4[1] * Theta_e)) / (
                _c4[2] + Theta_e * (_c4[3] + Theta_e * (_c4[4] + Theta_e * (_c4[5] + Theta_e * _c4[6] * _c4[1])))
            )
            a6 = (_c6[0] + Theta_e) / (
                _c6[1] + Theta_e * (_c6[2] + Theta_e * (_c6[3] + Theta_e * (_c6[4] + Theta_e * _c6[5])))
            )
            a8 = (_c8[6] + Theta_e * (_c8[7] + Theta_e * _c8[8])) / (
                1.0
                + Theta_e
                * (
                    _c8[0]
                    + Theta_e
                    * (_c8[1] + Theta_e * (_c8[2] + Theta_e * (_c8[3] + Theta_e * (_c8[4] + Theta_e * _c8[5]))))
                )
            )
            Theta_2 = Theta_e**2
            J = (1.0 + Theta_2 * (_cJ[0] + Theta_2 * (_cJ[1] + Theta_2 * Theta_e * _cJ[2]))) / (
                1.0
                + Theta_2
                * (
                    (_cJ[0] - PI_SQR / 6.0)
                    + Theta_2
                    * (
                        _cJ[3]
                        + Theta_2 * (_cJ[4] + Theta_e**1.5 * 0.75 * _cJ[2] * (1.0 / SQRT_HALF_PI + Theta_e**1.5))
                    )
                )
            )
            K = (1.0 + Theta_2 * (_cK[0] + Theta_2 * (_cK[1] + Theta_2 * Theta_e * _cK[2]))) / (
                1.0
                + Theta_2
                * (
                    (_cK[0] - 0.75 * PI_SQR)
                    + Theta_2
                    * (
                        _cK[3]
                        + Theta_e
                        * Theta_2
                        * (_cK[4] - Theta_e**1.5 * 0.125 * _cK[2] * (7.0 / SQRT_HALF_PI + 3.0 * Theta_e**1.5))
                    )
                )
            )
            Theta_3 = Theta_e**3
            Im0p5 = SQRT_PI * F_m0p5
            Ip1p5 = fdi(j=+1.5, eta=eta, normalize=False)
            Ip2p5 = fdi(j=+2.5, eta=eta, normalize=False)
            b10 = 1.5 * sqrt_Theta_e * Im0p5 * a8
            b8 = sqrt_Theta_e * (1.5 * Im0p5 * a6 - 0.5 * Theta_2 * Ip1p5 * b10)
            b6 = sqrt_Theta_e * (1.5 * Im0p5 * a4 - 0.5 * Theta_2 * Ip1p5 * b8 - 0.3 * Theta_3 * Ip2p5 * b10)
            b2 = a2 + J / (1.5 * sqrt_Theta_e * Im0p5)
            b4 = b2**2 - a2 * b2 + a4 + 2.0 * K / (15.0 * sqrt_Theta_e * Im0p5)
            x2 = x * x
            u = 1.0 + x2 * (a2 + x2 * (a4 + x2 * (a6 + x2 * a8)))
            v = 1.0 + x2 * (b2 + x2 * (b4 + x2 * (b6 + x2 * (b8 + x2 * b10))))
            return x * u / v

        exp_arg_pos = eta - (w + q) ** 2
        exp_arg_neg = eta - (w - q) ** 2
        imag_part = SQRT_PI * (log1pexp(exp_arg_pos) - log1pexp(exp_arg_neg))
        delta_F = phi_function(w0 + q0) - phi_function(w0 - q0)
        real_part = -2.0 * F_m0p5 * delta_F / sqrt_Theta_e
        prefactor = self.state.free_electron_number_density / (BOLTZMANN_CONSTANT * self.state.electron_temperature)
        pol_func = prefactor * (real_part + 1.0j * imag_part) / (4.0 * F_p0p5 * q)

        return pol_func

    def _real_dielectric_mermin(self, k, w, mu1, mu2):
        # TODO(Hannah): get this working
        Te = self.state.electron_temperature
        ne = self.state.free_electron_number_density
        EF = self.state.fermi_energy(ne, ELECTRON_MASS)
        be = 1 / (ELEMENTARY_CHARGE * Te)
        pe = np.sqrt(ELECTRON_MASS * ELEMENTARY_CHARGE * Te)
        De = ne * (TWO_PI * DIRAC_CONSTANT**2 / (ELECTRON_MASS * ELEMENTARY_CHARGE * Te)) ** 1.5
        xe = ELECTRON_MASS * (w / k) / (np.sqrt(2) * pe)
        Ke = DIRAC_CONSTANT * (k / 2) / (np.sqrt(2) * pe)
        Reze = ELECTRON_MASS * mu1 / (np.sqrt(2) * k * pe)
        Imze = ELECTRON_MASS * mu2 / (np.sqrt(2) * k * pe)
        alpha = self.state.chemical_potential_ichimaru(Te, ne, ELECTRON_MASS)  # / EF

        def integrand(u):
            x = np.tan(HALF_PI * u)
            xbar = xe - Imze
            log_term = (1 + np.exp(alpha - (x + Ke) ** 2)) / (1 + np.exp(alpha - (x - Ke) ** 2))
            f = (x - xbar) / ((x - xbar) ** 2 + Reze**2) + (x + xbar) / ((x + xbar) ** 2 + Reze**2) * np.log(log_term)
            return f * HALF_PI * (1 + x**2)

        real_part = integrate.quad_vec(integrand, 0, 1)[0]

        return real_part * 2 * ne * be / (2 * SQRT_PI * Ke * De)

        # w_freq = w / DIRAC_CONSTANT

        # kF = self.state.fermi_wave_number(self.state.free_electron_number_density)  # 1/m
        # EF = self.state.fermi_energy(self.state.free_electron_number_density, ELECTRON_MASS)  # J
        # TF = self.state.fermi_temperature(ELECTRON_MASS, self.state.free_electron_number_density)  # K
        # vF = DIRAC_CONSTANT * kF / ELECTRON_MASS  # m/s
        # omega_p = self.state.plasma_frequency(-1, self.state.free_electron_number_density, ELECTRON_MASS)
        # beta = 1 / (BOLTZMANN_CONSTANT * self.state.electron_temperature)
        # mu = self.state.chemical_potential_ichimaru(
        #     self.state.electron_temperature, self.state.free_electron_number_density, ELECTRON_MASS
        # )  # Joule

        # u = w_freq / (vF * k)  # [#]
        # kappa = k / (2 * kF)  # [#]

        # D = EF * beta  # dimensionless
        # alpha = mu / EF
        # t = self.state.electron_temperature / TF  # [#]

        # def g_ancarni(lambda_val):
        #     return np.where(lambda_val < 0.0, -g_t(-lambda_val), g_t(lambda_val))

        # def g_t(lambda_val, eps=1.0e-9):

        #     A = lambda_val**2 / t
        #     B = alpha / t

        #     def f_prime(X):
        #         y = A * X**2 - B
        #         s = np.empty_like(y)

        #         pos_mask = y >= 0.0
        #         neg_mask = ~pos_mask

        #         # y >= 0 branch
        #         exp_neg = np.exp(-y[pos_mask])
        #         s[pos_mask] = 1.0 / (1.0 + exp_neg)

        #         # y < 0 branch
        #         exp_pos = np.exp(y[neg_mask])
        #         s[neg_mask] = exp_pos / (1.0 + exp_pos)

        #         return -2.0 * A * X * (s * (1.0 - s))

        #     def integrand_u(u):
        #         X = np.tan(np.pi * u / 2)
        #         log_term = np.log(abs((X + 1) / (X - 1)))
        #         bracket_term = -X + 0.5 * (1 - X**2) * log_term
        #         dX_du = (np.pi / 2) * (1 / np.cos(np.pi * u / 2) ** 2)
        #         return f_prime(X) * bracket_term * dX_du

        #     res1, _ = integrate.quad_vec(integrand_u, 0, 0.5 - eps, limit=300, points=[0.1, 0.2, 0.3, 0.4])
        #     res2, _ = integrate.quad_vec(integrand_u, 0.5 + eps, 1, limit=300, points=[0.6, 0.7, 0.8, 0.9])

        #     return (lambda_val**2) * (res1 + res2)

        # chi02 = 1 / (PI * kF * BOHR_RADIUS)  # [#]
        # lambda_neg = u - kappa  # [#]
        # lambda_pos = u + kappa  # [#]
        # g_t_pos = g_ancarni(lambda_pos)
        # g_t_neg = g_ancarni(lambda_neg)
        # real_part = 1.0e0 + chi02 / (4 * kappa**3) * (g_t_pos - g_t_neg)

        # return real_part

    def _im_dielectric_mermin(self, k, w, mu1, mu2):
        # TODO(HB): get this working
        Te = self.state.electron_temperature
        ne = self.state.free_electron_number_density
        EF = self.state.fermi_energy(ne, ELECTRON_MASS)
        be = 1 / (ELEMENTARY_CHARGE * Te)
        pe = np.sqrt(ELECTRON_MASS * ELEMENTARY_CHARGE * Te)
        De = ne * (TWO_PI * DIRAC_CONSTANT**2 / (ELECTRON_MASS * ELEMENTARY_CHARGE * Te)) ** 1.5
        xe = ELECTRON_MASS * (w / k) / (np.sqrt(2) * pe)
        Ke = DIRAC_CONSTANT * (k / 2) / (np.sqrt(2) * pe)
        Reze = ELECTRON_MASS * mu1 / (np.sqrt(2) * k * pe)
        Imze = ELECTRON_MASS * mu2 / (np.sqrt(2) * k * pe)
        alpha = self.state.chemical_potential_ichimaru(Te, ne, ELECTRON_MASS)  # / EF

        def integrand(u):
            x = np.tan(HALF_PI * u)
            xbar = xe - Imze
            log_term = (1 + np.exp(alpha - (x + Ke) ** 2)) / (1 + np.exp(alpha - (x - Ke) ** 2))
            f = 1 / ((x - xbar) ** 2 + Reze**2) - 1 / ((x + xbar) ** 2 + Reze**2) * Reze * np.log(log_term)
            return f * HALF_PI * (1 + x**2)

        imag_part = integrate.quad_vec(integrand, 0, 1)[0]
        return imag_part * 2 * ne * be / (2 * SQRT_PI * Ke * De)

        # kF = self.state.fermi_wave_number(self.state.free_electron_number_density)  # 1/m
        # EF = self.state.fermi_energy(self.state.free_electron_number_density, ELECTRON_MASS)  # J
        # vF = DIRAC_CONSTANT * kF / ELECTRON_MASS  # m/s
        # u = w / (k * vF * DIRAC_CONSTANT)  # dimensionless
        # z = k / (2 * kF)  # dimensionless

        # beta = 1 / (BOLTZMANN_CONSTANT * self.state.electron_temperature)
        # mu = self.state.chemical_potential_ichimaru(
        #     self.state.electron_temperature, self.state.free_electron_number_density, ELECTRON_MASS
        # )  # Joule
        # alpha = mu / EF

        # D = EF * beta  # dimensionless
        # eta = mu * beta  # dimensionless
        # chi02 = 1 / (PI * kF * BOHR_RADIUS)  # dimensionless
        # theta = 1 / D

        # # def integrand(u):
        # #     y = np.tan(HALF_PI * u)
        # #     dydy = HALF_PI * (1 + y**2)
        # #     fy = 1 / (np.exp(D * y**2 - alpha) + 1)
        # #     tan_funcs =
        # #     return tan_funcs * y / fy * dydy

        # if z < 1.0e-2:
        #     prefactor = (
        #         2
        #         * ELECTRON_MASS**2
        #         * ELEMENTARY_CHARGE**2
        #         / VACUUM_PERMITTIVITY
        #         * DIRAC_CONSTANT
        #         * w
        #         / (DIRAC_CONSTANT * k) ** 3
        #     )
        #     exp_term = EF / (BOLTZMANN_CONSTANT * self.state.electron_temperature) * u**2 - eta**2
        #     im_part = prefactor / (1 + np.exp(exp_term))

        # else:
        #     xpos = (u + z) ** 2  # Dimensionless
        #     xneg = (u - z) ** 2  # Dimensionless

        #     exp_neg = np.exp(eta - D * xneg)
        #     exp_pos = np.exp(eta - D * xpos)

        #     log_term = (1 + exp_neg) / (1 + exp_pos)  # [#]
        #     idx = np.where(np.abs(log_term.imag) < 1.0e-2)
        #     im_part = 1 * PI * chi02 / (8 * z**3) * theta * np.log(log_term)  # [#]
        # return im_part

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
            input_collision_frequency (float): user-defined static collision frequency in 1/s

        Returns:
            array: polarisation, non-dimensional
        """
        omega_p = (
            self.state.plasma_frequency(-1, self.state.free_electron_number_density, ELECTRON_MASS) * DIRAC_CONSTANT
        )
        mu_ei = np.array([0.5 * omega_p * (1.0 + 1.0j)])
        Vee = 4 * np.pi * COULOMB_CONSTANT * ELEMENTARY_CHARGE**2 / k**2
        # print(f"Plasma frequency = {omega_p * J_TO_eV} eV")
        # print(f"Running with collision frequency: {mu_ei * J_TO_eV} eV")
        # real_part = self._real_dielectric_mermin(k=k, w=w + mu_ei, mu1=np.real(mu_ei), mu2=np.imag(mu_ei))
        # imag_part = self._im_dielectric_mermin(k=k, w=w + mu_ei, mu1=np.real(mu_ei), mu2=np.imag(mu_ei))
        # dielectric_function = 1 - Vee * (real_part + 1.0j * imag_part)

        # TODO(Hannah): replace this with the actual numerical integration
        dielectric_function = self.dielectric_function(k=k, w=w + 1.0j * mu_ei, model="DANDREA_FIT")
        dielectric_rpa0 = self.dielectric_function(k=k, w=0, model="DANDREA_FIT")
        mermin_dielectric = 1 + (1 + 1.0j * mu_ei / w) * (dielectric_function - 1) / (
            1 + 1.0j * mu_ei / w * (dielectric_function - 1) / (dielectric_rpa0 - 1)
        )
        return mermin_dielectric

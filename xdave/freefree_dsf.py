from constants import *
from unit_conversions import *
from maths import log1pexp
from scipy import integrate
from plasma_state import PlasmaState

# from collision_frequency import CollisionFrequency

from plasmapy.formulary.mathematics import Fermi_integral as fdi
from scipy.special import gamma

import numpy as np

import warnings


class FreeFreeDSF:

    def __init__(self, state: PlasmaState) -> None:
        self.state = state

    def get_dsf(self, k, w, lfc, model="NUMERICAL_RPA"):

        if self.state.free_electron_number_density == 0.0:
            return 0.0

        chi0 = self.susceptibility_function(k=k, w=w, model=model)
        Vee = 4 * np.pi * COULOMB_CONSTANT * ELEMENTARY_CHARGE**2 / k**2
        chilfc = chi0 / (1 - Vee * (1 - lfc) * chi0)
        im_suspectibility = np.imag(chilfc)
        S_EG_LFC = (
            -(1)
            / (PI * self.state.free_electron_number_density)
            * 1
            / (1 - np.exp(-w / (BOLTZMANN_CONSTANT * self.state.electron_temperature)))
            * im_suspectibility
        )

        return S_EG_LFC

    def dielectric_function(self, k, w, model):
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
            dielectric_func = self.mermin_dielectric_function(k=k, w=w)
        else:
            dielectric_func = self.rpa_numerical_dielectric_func(k=k, w=w)
            warnings.warn(f"Model {model} not recognized. Overwriting using NUMERICAL.")

        return dielectric_func

    def susceptibility_function(self, k, w, model):
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
            warnings.warn(f"Model {model} not recognized. Overwriting using NUMERICAL.")

        return susceptibility_func

    def lindhard_pol_func(self, k, w):
        """
        Note to self: the definition of the plasma frequency has changed, I will need to check this.
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
        ## taken from Eqn. (5.5) in K. W\"unsch PhD Thesis (2011)
        im_part = self._im_dielectric_rpa(k, w)  # [#]
        real_part = self._real_dielectric_rpa(k, w)
        dielectric_function = real_part + 1.0j * im_part
        return dielectric_function

    def _im_dielectric_rpa(self, k, w):

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
        # Note to self: all of this currently assumes k is a single value, not an array

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

        F_m0p5 = fdi(j=-0.5, x=eta)
        F_p0p5 = fdi(j=+0.5, x=eta)

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
            Ip1p5 = 0.750 * SQRT_PI * fdi(j=+1.5, x=eta)
            Ip2p5 = 1.875 * SQRT_PI * fdi(j=+2.5, x=eta)
            b10 = 1.5 * sqrt_Theta_e * Im0p5 * a8
            b8 = sqrt_Theta_e * (1.5 * Im0p5 * a6 - 0.5 * Theta_2 * Ip1p5 * b10)
            b6 = sqrt_Theta_e * (1.5 * Im0p5 * a4 - 0.5 * Theta_2 * Ip1p5 * b8 - 0.3 * Theta_3 * Ip2p5 * b10)
            b2 = a2 + J / (1.5 * sqrt_Theta_e * Im0p5)
            b4 = b2**2 - a2 * b2 + a4 + 2.0 * K / (15.0 * sqrt_Theta_e * Im0p5)
            x2 = x * x
            u = 1.0 + x2 * (a2 + x2 * (a4 + x2 * (a6 + x2 * a8)))
            v = 1.0 + x2 * (b2 + x2 * (b4 + x2 * (b6 + x2 * (b8 + x2 * b10))))
            return x * u / v

        exp_arg_p = eta - (w + q) ** 2
        exp_arg_m = eta - (w - q) ** 2
        Im_X_kw = SQRT_PI * (log1pexp(exp_arg_p) - log1pexp(exp_arg_m))
        delta_F = phi_function(w0 + q0) - phi_function(w0 - q0)
        Re_X_kw = -2.0 * F_m0p5 * delta_F / sqrt_Theta_e
        Pi_aa_0_MB = self.state.free_electron_number_density / (BOLTZMANN_CONSTANT * self.state.electron_temperature)
        pol_func = Pi_aa_0_MB * (Re_X_kw + 1.0j * Im_X_kw) / (4.0 * F_p0p5 * q)

        return pol_func

    def _real_dielectric_mermin(self, k, w, mu1, mu2):
        # Note to self: all of this currently assumes k is a single value, not an array

        w_freq = w / DIRAC_CONSTANT

        kF = self.state.fermi_wave_number(self.state.free_electron_number_density)  # 1/m
        EF = self.state.fermi_energy(self.state.free_electron_number_density, ELECTRON_MASS)  # J
        TF = self.state.fermi_temperature(ELECTRON_MASS, self.state.free_electron_number_density)  # K
        vF = DIRAC_CONSTANT * kF / ELECTRON_MASS  # m/s
        omega_p = self.state.plasma_frequency(-1, self.state.free_electron_number_density, ELECTRON_MASS)

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

        def g_t(lambda_val, eps=1.0e-6):

            A = lambda_val**2 / t
            B = alpha / t

            def f_prime(X):
                """
                Eqn. (10) in Ancarani et al., Eur. Phys. J. Plus 131 (2016)
                """
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
                """
                Eqn. (9) in Ancarani et al., Eur. Phys. J. Plus 131 (2016)
                """
                X = np.tan(np.pi * u / 2)
                log_term = np.log(abs((X + 1) / (X - 1)))
                bracket_term = -X + 0.5 * (1 - X**2) * log_term
                dX_du = (np.pi / 2) * (1 / np.cos(np.pi * u / 2) ** 2)
                return f_prime(X) * bracket_term * dX_du

            # integrate over
            # res1, _ = integrate.quad_vec(integrand_u, 0, 0.5 - eps, limit=500, epsrel=1.0e-1)
            # res2, _ = integrate.quad_vec(integrand_u, 0.5 + eps, 1, limit=500, epsrel=1.0e-1)
            res, _ = integrate.quad_vec(integrand_u, 0, 1, limit=200, epsrel=1.0e-2)

            return (lambda_val**2) * res  # (res1 + res2)

        chi02 = 1 / (PI * kF * BOHR_RADIUS)  # [#]

        # Get asymptotic limits from Arista et al., Phys. Rev. A 29 (1984)

        # w_small = w[(w > -1 * eV_TO_J) & (w < 1 * eV_TO_J)]
        # u_small = u[(w > -1 * eV_TO_J) & (w < 1 * eV_TO_J)]
        def H1(d):
            def integrand(u):
                y = np.tan(HALF_PI * u)
                return HALF_PI * (1 + y**2) * f_hat(y, d, mu)

            return integrate.quad(integrand, 0, 1)[0]

        def f_hat(y, D, eta):
            return 1.0 / (1.0 + np.exp(D * y**2 - eta))

        def H2(d):
            f0 = f_hat(0, d, alpha)

            def integrand(u):
                y = np.tan(HALF_PI * u)
                return HALF_PI * (1 + y**2) * (f0 - f_hat(y, d, mu)) / y**2

            return integrate.quad(integrand, 0, 1)[0]

        # real_part = np.zeros_like(w)
        # is_z_small = kappa < 5.0e-1
        # is_z_large = kappa > 1.0e1
        # cond1 = (w_freq < 1.0e-1 * vF * k) & (w_freq >= 0.0)
        # print(f"Condition 1: {w[cond1] * J_TO_eV}")
        # print(f"Condition 2: {k < 1.e-1 * 2 * kF}")

        # cond3 = w_freq > 1.0e1 * vF * k
        # print(f"Condition 3: {w[cond3] * J_TO_eV}")
        # cond4 = w_freq > 1.0e1 * vF * k**2 / (2 * kF)
        # print(f"Condition 4: {w[cond4] * J_TO_eV}")

        # cond5 = (w_freq < 1.0e-1 * vF * k**2 / (2 * kF)) & (w_freq >= 0)
        # print(f"Condition 5: {w[cond5] * J_TO_eV}")
        # print(f"Condition 6: {k > 1.e1 * 2 * kF}")

        # D = 1 / t
        # if is_z_small:
        #     cond = (u < 1.0e-1) & (u >= 0.0)
        #     idx1 = np.where((u < 1.0e-1) & (u >= 0.0))
        #     # idx1_neg = np.where((u > -1.0e-1) & u <= 0.0)
        #     u_small = u[idx1]
        #     real_limit1 = 1 - chi02 / 3 * H2(D) + chi02 / kappa**2 * (H1(D) - u_small**2 * H2(D))
        #     real_part[idx1] = 1 - chi02 / 3 * H2(D) + chi02 / kappa**2 * (H1(D) - u[idx1] ** 2 * H2(D))
        #     # real_part[idx1_neg] = real_part[idx1][::-1]

        # if is_z_small:
        #     idx2 = np.where(u > 2.0e0)
        #     u_large = u[idx2]
        #     w_freq_large = w_freq[idx2]
        #     real_limit2 = (
        #         1
        #         - omega_p**2
        #         / w_freq_large**2
        #         * (
        #             1
        #             + kappa**2 / u_large**2
        #             + 3 * fdi(1.5, mu) / gamma(2.5) / (2 * u_large**2 * D**2.5)
        #             + 3 * fdi(2.5, mu) / gamma(3.5) / (2 * u_large**4 * D**3.5)
        #         ).real
        #     )
        #     real_part[idx2] = real_limit2

        # if is_z_large:
        #     Ek = DIRAC_CONSTANT**2 * k**2 / (2 * ELECTRON_MASS)
        #     u_small = u[(u < 1.0e-2) & (u > -1.0e-2)]
        #     real_limit3 = 1 + (DIRAC_CONSTANT * omega_p / Ek) ** 2 * (
        #         1
        #         + u**2 / kappa**2
        #         + 0.5 * fdi(1.5, alpha) / gamma(2.5) / (kappa**2 * D**2.5)
        #         + 3 / 10 * fdi(2.5, alpha) / gamma(3.5) / (kappa**4 * D**3.5)
        #     )

        # u_new = np.delete(u, idx1)
        # u_new = np.delete(u_new, idx2)
        lambda_neg = u - kappa  # [#]
        lambda_pos = u + kappa  # [#]
        # lambda_neg = u_new - kappa  # [#]
        # lambda_pos = u_new + kappa  # [#]
        g_t_pos = g_ancarni(lambda_pos)
        g_t_neg = g_ancarni(lambda_neg)
        real_part = 1.0e0 + chi02 / (4 * kappa**3) * (g_t_pos - g_t_neg)

        # import matplotlib.pyplot as plt

        # plt.figure()
        # plt.plot(w[idx1], real_limit1, label="small")
        # plt.plot(w[idx2], real_limit2, label="large")
        # plt.plot(w, real_part, label="full")
        # plt.legend()
        # plt.xlabel(r"§\omega§ [J]")
        # plt.xlabel(r"re[§\epsilon§]")
        # plt.show()

        return real_part

    def _im_dielectric_mermin(self, k, w, mu1, mu2):
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

        if z < 1.0e-2:
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

    def mermin_dielectric_function(
        self, k, w, lfc=0, collision_frequency_model="BORN", input_collision_frequency=None
    ):
        omega_p = (
            self.state.plasma_frequency(-1, self.state.free_electron_number_density, ELECTRON_MASS) * DIRAC_CONSTANT
        )
        mu_ei = 0.5 * omega_p * (1.0 + 1.0j)
        print(f"Plasma frequency = {omega_p * J_TO_eV} eV")
        print(f"Running with collision frequency: {mu_ei * J_TO_eV} eV")
        real_part = self._real_dielectric_mermin(k=k, w=w + 1.0j * mu_ei, mu1=np.real(mu_ei), mu2=np.imag(mu_ei))
        imag_part = self._im_dielectric_mermin(k=k, w=w + 1.0j * mu_ei, mu1=np.real(mu_ei), mu2=np.imag(mu_ei))
        dielectric_function = real_part + 1.0j * imag_part
        # dielectric_function = self.dielectric_function(k=k, w=w + 1.0j * mu_ei, model="NUMERICAL")
        dielectric_rpa0 = self.dielectric_function(k=k, w=0, model="DANDREA_FIT")
        mermin_dielectric = 1 + (1 + 1.0j * mu_ei / w) * (dielectric_function - 1) / (
            1 + 1.0j * mu_ei / w * (dielectric_function - 1) / (dielectric_rpa0 - 1)
        )
        return mermin_dielectric

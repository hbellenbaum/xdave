from constants import *
from unit_conversions import *
from maths import log1pexp
from scipy import integrate


# from fermi_integrals import fermi_integral
from plasma_state import PlasmaState, get_rho_T_from_rs_theta

from scipy.optimize import root_scalar
from scipy.integrate import quad

from models import ModelOptions
import numpy as np

import warnings

from plasmapy.formulary.mathematics import Fermi_integral as fdi


def effective_coulomb_potential(ionisation, wave_number):
    r"""
    Effective Coulomb potential
    """
    V_aa = -4 * PI * ionisation * ELEMENTARY_CHARGE_SQR * UNIT_COULOMB_POTENTIAL / (wave_number * wave_number)
    return V_aa


class FreeFreeDSF:

    def __init__(self, state: PlasmaState, models: ModelOptions) -> None:
        self.state = state
        self.polarisation_model = models.polarisation_model

    def get_dsf(self, k, w, lfc):
        # dielectric_func = self.dielectric_function(k=k, w=w)
        # im_dielectric = -np.imag(dielectric_func) / ((np.real(dielectric_func)) ** 2 + (np.imag(dielectric_func)) ** 2)

        # S_EG = (
        #     -1
        #     / (1 - np.exp(-w / (BOLTZMANN_CONSTANT * self.state.electron_temperature)))
        #     * VACUUM_PERMITTIVITY
        #     * k**2
        #     / (PI * ELEMENTARY_CHARGE**2 * self.state.free_electron_number_density)
        #     * im_dielectric
        # )
        if self.state.free_electron_number_density == 0.0:
            return 0.0

        chi0 = self.susceptibility_function(k=k, w=w)
        Vee = 4 * np.pi * COULOMB_CONSTANT * ELEMENTARY_CHARGE**2 / k**2
        chilfc = chi0 / (1 - Vee * (1 - lfc) * chi0)
        im_suspectibility = np.imag(chilfc)
        S_EG_LFC = (
            -(1)  # DIRAC_CONSTANT
            / (PI * self.state.free_electron_number_density)  # * Vee)
            * 1
            / (1 - np.exp(-w / (BOLTZMANN_CONSTANT * self.state.electron_temperature)))
            * im_suspectibility
        )

        return S_EG_LFC

    def dielectric_function(self, k, w):
        potential_func = 4 * np.pi * COULOMB_CONSTANT * ELEMENTARY_CHARGE**2 / k**2
        if self.polarisation_model == "LINDHARD":
            pol_func = self.lindhard_pol_func_dc(k=k, w=w)
            dielectric_func = 1 - potential_func * pol_func
        elif self.polarisation_model == "DANDREA_FIT":
            pol_func = self.dandrea_fit(k=k, omega=w)
            dielectric_func = 1 - potential_func * pol_func
        elif self.polarisation_model == "NUMERICAL":
            dielectric_func = self.rpa_numerical_dielectric_func(k=k, w=w)  # * potential_func
            # dielectric_func = self.rpa_numerical_dielectric_func_pontus(k=k, w=w)
        else:
            dielectric_func = self.rpa_numerical_dielectric_func(k=k, w=w)
            warnings.warn(f"Model {self.polarisation_model} not recognized. Overwriting using NUMERICAL.")

        return dielectric_func

    def susceptibility_function(self, k, w):
        if self.polarisation_model == "LINDHARD":
            susceptibility_func = self.lindhard_pol_func_dc(k=k, w=w)
        elif self.polarisation_model == "DANDREA_FIT":
            susceptibility_func = self.dandrea_fit(k=k, omega=w)
        elif self.polarisation_model == "NUMERICAL":
            potential_func = 4 * np.pi * COULOMB_CONSTANT * ELEMENTARY_CHARGE**2 / k**2
            dielectric_func = self.rpa_numerical_dielectric_func(k=k, w=w)  # * potential_func
            susceptibility_func = (1 - dielectric_func) / potential_func
        else:
            potential_func = 4 * np.pi * COULOMB_CONSTANT * ELEMENTARY_CHARGE**2 / k**2
            dielectric_func = self.rpa_numerical_dielectric_func(k=k, w=w)
            susceptibility_func = (1 - dielectric_func) / potential_func
            warnings.warn(f"Model {self.polarisation_model} not recognized. Overwriting using NUMERICAL.")

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
        # Z = np.abs(k / (2 * kF))
        # U = w / (4 * Z * EF)
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
        q0 = 0.5 * k / kF
        w0 = 0.25 * w / (EF * q0)

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
        # kF = self.state.fermi_wave_number(self.state.free_electron_number_density)
        # wF = self.state.fermi_frequency(self.state.free_electron_number_density, ELECTRON_MASS)

        # if use_long_wavelength_limit:
        #     return

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

        w_freq = w / DIRAC_CONSTANT

        kF = self.state.fermi_wave_number(self.state.free_electron_number_density)  # 1/m
        EF = self.state.fermi_energy(self.state.free_electron_number_density, ELECTRON_MASS)  # J
        TF = self.state.fermi_temperature(ELECTRON_MASS, self.state.free_electron_number_density)  # K
        vF = DIRAC_CONSTANT * kF / ELECTRON_MASS  # m/s

        u = w_freq / (vF * k)  # [#]
        kappa = k / (2 * kF)  # [#]

        t = self.state.electron_temperature * BOLTZMANN_CONSTANT / EF  # [#]
        t = self.state.electron_temperature / TF  # [#]

        theta = t  # BOLTZMANN_CONSTANT * self.state.electron_temperature / EF
        mu = self.state.chemical_potential_ichimaru(
            self.state.electron_temperature, self.state.free_electron_number_density, ELECTRON_MASS
        )  # [J]
        alpha = mu / EF
        eta = self.state.reduced_chemical_potential_tobias(theta=theta)
        # alpha = eta * t

        def g_ancarni(lambda_val):
            # if lambda_val < 0.0:
            #     return -g_t(-lambda_val)
            # else:
            #     return g_t(lambda_val)
            return np.where(lambda_val < 0.0, -g_t(-lambda_val), g_t(lambda_val))

        def g_t(lambda_val, eps=1.0e-9):

            A = lambda_val**2 / t
            B = alpha / t

            def f_prime(X):
                # y = A * X**2 - B
                # # scalar implementation; if X can be array, vectorize with np.where
                # if y >= 0.0:
                #     # exp(-y) is safe (<= 1)
                #     exp_neg = np.exp(-y)
                #     s = 1.0 / (1.0 + exp_neg)
                # else:
                #     # exp(y) is safe (<= 1)
                #     exp_pos = np.exp(y)
                #     s = exp_pos / (1.0 + exp_pos)
                # return -2.0 * A * X * (s * (1.0 - s))
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

    def rpa_correction_collisional(self, k, w):
        collision_freq = 0.0  # complex number!!!!!!!

        real_part = None

        omega_tilde = w - np.imag(collision_freq)
        kappa = omega_tilde * ELECTRON_MASS / (DIRAC_CONSTANT * k)
        delta = ELECTRON_MASS * np.real(collision_freq) / (DIRAC_CONSTANT * k)

        def imag_int(q):
            f_q = 1 / (np.exp() + 1)  # distribution function!!!!!!!!!!!
            y1 = (kappa - k / 2 - q) / delta
            y2 = (kappa + k / 2 + q) / delta
            y3 = (kappa - k / 2 + q) / delta
            y4 = (kappa + k / 2 - q) / delta
            I = 1 / (2 * PI) ** 3 * q * f_q * (np.arctan(y1) + np.arctan(y2) - np.arctan(y3) - np.arctan(y4))
            return I

        int_term = None
        im_part = (
            -4
            * PI
            * ELECTRON_MASS
            * ELEMENTARY_CHARGE**2
            / (VACUUM_PERMITTIVITY * DIRAC_CONSTANT**2 * k**3)
            * int_term
        )

        return complex(real_part, im_part)

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


def test():
    import matplotlib.pyplot as plt

    rs = 2
    theta = 1
    rho, Te = get_rho_T_from_rs_theta(rs=rs, theta=theta)
    ks = np.array((0.5, 1.0, 2.0, 4.0)) / BOHR_RADIUS  #  0.5, 1.0, 2.0, 4.0
    rho *= g_per_cm3_TO_kg_per_m3
    Te *= eV_TO_K
    # Te = 200  #
    charge_state = 1.0
    atomic_mass = 1.0
    atomic_number = 1.0
    lfc = 0.0

    models = ModelOptions(polarisation_model="NUMERICAL")
    models2 = ModelOptions(polarisation_model="DANDREA_FIT")

    omega_array = np.linspace(-100, 100, 100) * eV_TO_J  # + 8.5e3 * eV_TO_J
    state = PlasmaState(
        electron_temperature=Te,
        ion_temperature=Te,
        mass_density=rho,
        charge_state=charge_state,
        # frequency=omega,
        atomic_mass=atomic_mass,
        atomic_number=atomic_number,
    )

    # models = ModelOptions(polarisation_model=model)
    fig, axes = plt.subplots(1, 1, figsize=(14, 8))
    colors = ["magenta", "crimson", "orange", "dodgerblue", "lightgreen", "lightgray", "yellow", "cyan"]

    # fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))

    for k, cs in zip(ks, colors):
        # int_terms = []
        # real_dielectrics = np.zeros_like(omega_array)
        # im_dielectric = np.zeros_like(omega_array)
        dsfs = np.zeros_like(omega_array)
        dsfs2 = np.zeros_like(omega_array)
        q = k * BOHR_RADIUS

        # ims_dandrea = np.zeros_like(omega_array)
        # ims_rpa = np.zeros_like(omega_array)
        # reals_dandrea = np.zeros_like(omega_array)
        # reals_rpa = np.zeros_like(omega_array)
        # for i in range(0, len(omega_array)):
        w = omega_array  # [i]
        kernel = FreeFreeDSF(state=state, models=models)
        kernel2 = FreeFreeDSF(state=state, models=models2)
        # int_term = kernel._real_dielectric_rpa(k=k, w=w)
        dsfs = kernel.get_dsf(k=k, w=w, lfc=lfc)
        dsfs2 = kernel2.get_dsf(k=k, w=w, lfc=lfc)
        # dielectric_func, im_part_rpa, real_part_rpa = kernel.rpa_numerical_dielectric_func(k, w)
        # real_dielectrics[i] = np.real(dielectric_func)
        # im_dielectric[i] = np.imag(dielectric_func)
        # dsfs[i] = dsf
        # dsfs2[i] = dsf2

        idx = np.argwhere(np.isnan(dsfs))
        dsfs_new = np.delete(dsfs, idx)
        dsfs2_new = np.delete(dsfs2, idx)
        omega_new = np.delete(omega_array, idx)
        # dsfs_new *= 1 / J_TO_eV  # DIRAC_CONSTANT
        # twinx = axes.twinx()
        # axes.plot(omega_new * J_TO_eV, dsfs_new / J_TO_eV, label=f"q={q} 1/aB", c=cs)  #  /  np.max(dsfs_new)

        # axes2[0].plot(omega_array * J_TO_eV, reals_dandrea, label=f"Fit")
        # axes2[0].plot(omega_array * J_TO_eV, reals_rpa - 1, label=f"RPA")
        # axes2[1].plot(omega_array * J_TO_eV, ims_dandrea, label=f"Fit")
        # axes2[1].plot(omega_array * J_TO_eV, ims_rpa, label=f"RPA")

        # axes[1].plot(omega_array * J_TO_eV, real_dielectrics, label=f"k={k}", c=cs)
        # axes[2].plot(omega_array * J_TO_eV, im_dielectric, label=f"k={k}", c=cs)

        fname = f"validation/ff_dsf/4hannah_rs_{int(rs)}_theta_{int(theta)}_{q}.txt"
        dat_j = np.genfromtxt(fname=fname, skip_header=22)
        # print(dat_j)
        axes.plot(
            dat_j[:, 0] * RYDBERG_TO_eV,
            dat_j[:, 4] / RYDBERG_TO_eV,
            ls=":",
            label=f"Jan: q={q}",
            marker="*",
            markevery=50,
            c=cs,
        )  # / np.max(dat_j[:, 4])
        # ax0.plot(dat_j[:, 0] * RYBBERG_TO_eV, dat_j[:, 5] / np.max(dat_j[:, 5]), c=c, ls="dotted", label=f"LFC: q={q}")
        axes.plot(omega_new * J_TO_eV, dsfs2_new / J_TO_eV, label=f"Fit: q={q}", c=cs, ls="-.")  #  /  np.max(dsfs_new)

    axes.set_xlabel(r"$\omega$ [eV]")
    axes.set_ylabel(r"DSF [1/eV]")
    # axes[1].set_xlabel(r"$\omega$ [eV]")
    # axes[1].set_ylabel(r"$Re\{\epsilon^{RPA}\}$")
    # axes[2].set_xlabel(r"$\omega$ [eV]")
    # axes[2].set_ylabel(r"$Im\{\epsilon^{RPA}\}$")
    axes.legend()

    # axes2[0].legend()
    plt.tight_layout()
    plt.show()
    # fig.savefig("ff_dsf_test3.pdf", dpi=200)


if __name__ == "__main__":
    test()

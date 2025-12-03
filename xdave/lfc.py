r"""
References:
-------
Implemented models:
T. Dornheim et al., Phys. Rev. Lett. 125 (2020), DOI: 10.1103/physrevlett.125.235001
T. Dornheim et al., Phys. Rev. B 103 (2021), DOI: 10.1103/physrevb.103.165102
K. Utsumi and S. Ichimaru, Phys. Rev. A, 26 (1982), DOI: 10.1103/physreva.26.603
D.J.W. Geldart, S.H. Vosko, Can. J. Phys. 44 (1966), DOI: 10.1139/p64-183
G. Gregori et al., High Energy Denity Phys. 3 (2007), DOI: 10.1016/j.hedp.2007.02.006
B. Farid et al., Phys. Rev. B 48 (1993), DOI: 10.1103/physrevb.48.11602
-------
More details:
C. Fortmann et al., Phys. Rev. E 81 026405 (2010), DOI: 10.1103/physreve.81.026405
"""

from .plasma_state import PlasmaState, get_rho_T_from_rs_theta
from .unit_conversions import *
from .constants import *
from .fxc import Groth_A
from .models import ModelOptions

import numpy as np

from scipy import integrate
from scipy.special import i1 as mod_bessel_first


class LFC:
    """
    Class containing all things local field correction

    Attributes:
        state (PlasmaState): container for all plasma variables
        rs (float): non-dimensional Wigner-Seitz radius
        theta (float): non-dimensional temperature parameter
    """

    def __init__(self, state: PlasmaState):
        self.state = state
        self.theta = state.theta
        self.rs = state.rs

    def initialize(self, state: PlasmaState):
        self.z = 4 * (4 / (9 * PI)) ** (1 / 6) * np.sqrt(self.rs / PI)

        TF = self.state.fermi_temperature(ELECTRON_MASS, self.state.free_electron_number_density)
        Tq = TF * K_TO_eV / (1.32510 - 1.779 * np.sqrt(self.rs))
        Tee = np.sqrt((self.state.electron_temperature * K_TO_eV) ** 2 + Tq**2)
        Tee *= eV_TO_K

        self.Gamma_ee = state.electron_electron_coupling_parameter(
            number_density=state.free_electron_number_density, temperature=Tee
        )
        self.gee0 = self._ee_pair_distribution_function(z=self.z)
        self.geeT = self._ee_pair_distribution_function_finite_T(Te=Tee)

    def calculate_lfc(self, k, w, model="DORNHEIM_ESA"):
        """
        Calculate LFC for a given model.

        Parameters:
            k (float): wave number in units of 1/m
            w (array): array of energies in eV
            model (str): controls the model used for the LFC calculation
        """
        if self.state.charge_state == 0:
            return 0.0
        elif model == "NONE":
            return 0.0
        self.initialize(state=self.state)
        if model == "DORNHEIM_ESA":
            return self._dornheim_esa(k, w)
        elif model == "PADE_INTERP":
            return self._pade_interp_static(k, w)
        elif model == "UI":
            return self._utsumi_ichimaru_static(k, w)
        elif model == "GV":
            return self._geldart_vosko(k, w)
        elif model == "FARID":
            return self._farid_static(k, w)
        elif model == "NONE":
            return 0.0
        else:
            raise NotImplementedError(f"Model {model} not a recognized option for the local field correction.")

    def _ee_pair_distribution_function(self, z):
        """
        Eqn. (8) Gregori et al., High Energy Density Phys. 3 (2007)
        """
        I_z = mod_bessel_first(z)
        gee = 1 / 8 * (z / I_z) ** 2  # / 8
        return gee

    def _ee_pair_distribution_function_finite_T(self, Te):
        """
        Eqn. (4) - (7) Gregori et al., High Energy Density Phys. 3 (2007)
        """
        TF = self.state.fermi_temperature(mass=ELECTRON_MASS, number_density=self.state.free_electron_number_density)
        Tq = TF * K_TO_eV / (1.32510 - 1.779 * np.sqrt(self.rs))
        Tee = np.sqrt((self.state.electron_temperature * K_TO_eV) ** 2 + Tq**2)
        Gamma_ee = self.Gamma_ee

        z = ELECTRON_MASS * ELEMENTARY_CHARGE**3 * COULOMB_CONSTANT**2 / (Tee * DIRAC_CONSTANT**2)

        def integrand(u):
            x = np.tan(HALF_PI * u)

            # get asymptotic limits to avoid overflow
            f1 = x * np.exp(-z * x**2 - PI / x)
            f2 = np.exp(np.log(x) - z * x**2 - PI / x - np.log1p(-np.exp(-PI / x)))

            f = np.where(x < 1e-3, f1, f2)
            return f * HALF_PI * (1 + x**2)

        Csc = 1.0754
        geeT_part = np.exp(Csc * Gamma_ee**1.5 / ((Csc / np.sqrt(3.0)) ** 4 + Gamma_ee**4) ** (1 / 4))
        gbin = np.sqrt(2 * PI) * z**1.5 * integrate.quad(integrand, 0, 1, points=[0])[0]

        geeT = gbin * geeT_part
        return geeT

    def _gamma_T(self):
        """
        Eqn. (10) Gregori et al., High Energy Density Phys. 3 (2007)
        """
        Gamma_ee = self.Gamma_ee
        a = 0.0999305
        b = 0.0187058
        c = 0.0013240
        d = 0.0479236
        gammaT = (12 * PI**2) ** (1 / 3) * (a + b / Gamma_ee + c / Gamma_ee ** (4 / 3) - d / Gamma_ee ** (2 / 3))
        return gammaT

    def _gamma_0(self):
        """
        Eqn. (14) Gregori et al., High Energy Density Phys. 3 (2007)
        """
        rs = self.rs
        sqrt_rs = np.sqrt(rs)
        a = 0.0621814
        b = 0.61024
        c = 9.81379
        d = 2.82224
        e = 0.736411
        diff1 = (a + b * sqrt_rs) / (1 + c * sqrt_rs + d * rs + e * sqrt_rs**3)
        numerator = a * (5 * c * sqrt_rs + 6 * d * rs + 7 * e * sqrt_rs**3) + b * sqrt_rs * (
            4 * c * sqrt_rs + 5 * d * rs + 6 * e * sqrt_rs**3 + 3
        )
        denominator = 2 * rs**3 * (c * sqrt_rs + d * rs + e * sqrt_rs**3 + 1) ** 2
        diff2 = -numerator / denominator
        return 0.25 - (PI * (4 / (9 * PI)) ** (1 / 3) / 24) * (rs**3 * diff2 - 2 * rs * diff1)

    def _f_extended(self, t, a, b, c):
        return a + b * t + c * t**1.5

    def _alpha_extended(self, rs, a, b, c):
        return (a + b * rs) / (1.0 + c * rs)

    def _beta_extended(self, rs, a, b, c):
        return (a + b * rs) / (1.0 + c * rs)

    def _gamma_extended(self, rs, a, b, c):
        return (a + b * rs) / (1.0 + c * rs)

    def _delta_extended(self, rs, a, b, c):
        return (a + b * rs) / (1.0 + c * rs)

    def _x_m(self, theta):
        A_x = 2.64
        B_x = 0.31
        C_x = 0.08
        return A_x + B_x * theta + C_x * theta**2

    def _on_top(self, rs, theta):
        """
        Spin-up-down component of the pair distribution function at zero distance, g(0)
        """
        t = theta

        # parameters from the ground-state parametrization by Spink et al. [Phys. Rev. B 88, 085121 (2013)]
        a_Spink = 0.18315
        b_Spink = -0.0784043
        c_Spink = 1.02232
        d_Spink = 0.0837741

        # parameters for the temperature-dependent parametrization:
        Qalpha_1_a = 18.4377
        Qbeta_1_a = 24.1339
        Qbeta_2_a = 1.86499
        Qalpha_1_b = -0.24368
        Qbeta_1_b = 0.252577
        Qbeta_2_b = 0.127043
        Qalpha_1_c = 2.23663
        Qbeta_1_c = 0.445526
        Qbeta_2_c = 0.408504
        Qalpha_2_c = 0.448937
        Qalpha_1_d = 0.0589015
        Qbeta_1_d = -0.598508
        Qbeta_2_d = 0.513162

        return (
            1.0
            + (a_Spink + Qalpha_1_a * t) / (1.0 + t * Qbeta_1_a + t * t * t * Qbeta_2_a) * np.sqrt(rs)
            + (b_Spink + Qalpha_1_b * np.sqrt(t)) / (1.0 + t * Qbeta_1_b + t * t * Qbeta_2_b) * rs
        ) / (
            1.0
            + (c_Spink + Qalpha_1_c * np.sqrt(t) + Qalpha_2_c * t * np.sqrt(t))
            / (1.0 + t * Qbeta_1_c + t * t * Qbeta_2_c)
            * rs
            + (d_Spink + Qalpha_1_d * np.sqrt(t)) / (1.0 + t * Qbeta_1_d + t * t * Qbeta_2_d) * rs * rs * rs
        )

    def _activation_analytical(self, x, a, b):
        return 0.5 * (np.tanh(b * (x - a)) + 1.0)

    def _G_fit_wrap_extended(self, x, alpha, beta, gamma, delta):
        rs = self.rs
        theta = self.theta

        # compute the first part of ESA, i.e., fit to the neural-net representation [Dornheim et al, J. Chem. Phys. 151, 194104 (2019)] of the static LFC
        # gdb.Groth_A gives pre-factor to the exact compressibility sum-rule (CSR) computed from the prametrization of fxc by Groth et al [PRL 119 (13), 135001 (2017)]
        G_ML_fit = (
            Groth_A(rs, theta)
            * x
            * x
            * (1.0 + alpha * x + beta * x**0.5)
            / (1.0 + gamma * x + delta * x**1.25 + Groth_A(rs, theta) * x**2)
        )

        # Obtain the value of the full ontop PDF g(0). Factor 0.5, because, OnTop returns only same-spin component
        onTop = 0.5 * self._on_top(rs, theta)

        # consistent large-q limit of an effectively static theory for the LFC
        Ginfty = 1.0 - onTop

        # width of transition between limits in the activation function is constant in this analytical representation
        ETA = 3.0

        # the position (wave-number) of the transition depends on theta only and has been parametrized above
        XM = self._x_m(theta)

        # the final result for the LFC within ESA is given by the combination of the static LFC (G_ML) and the large-g limit (Ginfty), connected by the Activation function
        A = self._activation_analytical(x, XM, ETA)
        return G_ML_fit * (1.0 - A) + A * Ginfty

    def _dornheim_esa(self, k, w):
        """
        Effective static approximation based on PIMC simulations of the uniform electron gas.
        T. Dornheim et al., Phys. Rev. Lett. 125 (2020), DOI: 10.1103/physrevlett.125.235001
        T. Dornheim et al., Phys. Rev. B 103 (2021), DOI: 10.1103/physrevb.103.165102

        Code copied over from: https://github.com/ToDor90/LFC (and modified slightly).

        Parameters:
            k (float/array): scattering wavenumber in units of a_B^{-1}
            w (array): array of energies in eV

        Returns:
            float/array: output type depending on the input type of k, local field correction, non-dimensional
        """
        rs = self.rs
        theta = self.theta
        kF = 1 / rs * (3 / 4 * np.pi) ** 3
        r = k * BOHR_RADIUS / kF  # self.state.fermi_wave_number(self.state.free_electron_number_density)

        coeff = [
            0.66477593,
            -4.59280227,
            1.24649624,
            -1.27089927,
            1.26706839,
            -0.4327608,
            2.09717766,
            1.15424724,
            -0.65356955,
            -1.0206202,
            5.16041218,
            -0.23880981,
            1.07356921,
            -1.67311761,
            0.58928105,
            0.8469662,
            1.54029035,
            -0.71145445,
            -2.31252076,
            5.83181391,
            2.29489749,
            1.76614589,
            -0.09710839,
            -0.33180686,
            0.56560236,
            1.10948188,
            -0.43213648,
            1.3742155,
            -4.01393906,
            -1.65187145,
            -1.75381153,
            -1.17022854,
            0.76772906,
            0.63867766,
            1.07863273,
            -0.35630091,
        ]

        # ### Determination of first rs-parameter alpha

        a = coeff[0]
        b = coeff[1]
        c = coeff[2]
        my_alpha_a = self._f_extended(theta, a, b, c)
        a = coeff[3]
        b = coeff[4]
        c = coeff[5]
        my_alpha_b = self._f_extended(theta, a, b, c)
        a = coeff[6]
        b = coeff[7]
        c = coeff[8]
        my_alpha_c = self._f_extended(theta, a, b, c)
        my_alpha = self._alpha_extended(rs, my_alpha_a, my_alpha_b, my_alpha_c)

        # ### Determination of second rs-parameter beta
        a = coeff[9]
        b = coeff[10]
        c = coeff[11]
        my_beta_a = self._f_extended(theta, a, b, c)
        a = coeff[12]
        b = coeff[13]
        c = coeff[14]
        my_beta_b = self._f_extended(theta, a, b, c)
        a = coeff[15]
        b = coeff[16]
        c = coeff[17]
        my_beta_c = self._f_extended(theta, a, b, c)
        my_beta = self._beta_extended(rs, my_beta_a, my_beta_b, my_beta_c)

        # ### Determination of third rs-parameter gamma
        a = coeff[18]
        b = coeff[19]
        c = coeff[20]
        my_gamma_a = self._f_extended(theta, a, b, c)
        a = coeff[21]
        b = coeff[22]
        c = coeff[23]
        my_gamma_b = self._f_extended(theta, a, b, c)
        a = coeff[24]
        b = coeff[25]
        c = coeff[26]
        my_gamma_c = self._f_extended(theta, a, b, c)
        my_gamma = self._gamma_extended(rs, my_gamma_a, my_gamma_b, my_gamma_c)

        # ### Determination of fourth rs-parameter delta
        a = coeff[27]
        b = coeff[28]
        c = coeff[29]
        my_delta_a = self._f_extended(theta, a, b, c)
        a = coeff[30]
        b = coeff[31]
        c = coeff[32]
        my_delta_b = self._f_extended(theta, a, b, c)
        a = coeff[33]
        b = coeff[34]
        c = coeff[35]
        my_delta_c = self._f_extended(theta, a, b, c)
        my_delta = self._delta_extended(rs, my_delta_a, my_delta_b, my_delta_c)

        return_value = self._G_fit_wrap_extended(r, my_alpha, my_beta, my_gamma, my_delta)
        return return_value

    def _utsumi_ichimaru_static(self, k, w):
        """
        Static local field correction based on K. Utsumi and S. Ichimaru, Phys. Rev. A, 26 (1982), DOI: 10.1103/physreva.26.603
        Details also in Gregori et al., High Energy Density Phys. 3 (2007)
        and Farid et al., Phys. Rev. B 48 (1993)

        Parameters:
            k (float/array): scattering wavenumber in units of a_B^{-1}
            w (array): array of energies in eV, not used here

        Returns:
            float/array: output type depending on the input type of k, local field correction, non-dimensional
        """
        kF = self.state.fermi_wave_number(self.state.free_electron_number_density)
        kF = (3 * np.pi * np.pi * self.state.free_electron_number_density) ** (1 / 3)
        Q = k / kF
        gee0 = self.gee0
        gamma0 = self._gamma_0()

        A = 0.029
        B = 9 / 16 * gamma0 - 3 / 64 * (1 - gee0) - 16 / 15 * A
        C = -3 / 4 * gamma0 + 9 / 16 * (1 - gee0) - 16 / 5 * A

        tol = 1e-4
        G_k = np.where(
            np.abs(Q - 2.0) < tol,
            A * 2**4 + B * 2**2 + C,  # stable limit
            A * Q**4
            + B * Q**2
            + C
            + (A * Q**4 + (B + 8 * A / 3) * Q**2 - C) * ((4 - Q**2) / (4 * Q)) * np.log(np.abs((2 + Q) / (2 - Q))),
        )

        return G_k

    def _geldart_vosko(self, k, w):
        """
        Static local field correction based on D.J.W. Geldart, S.H. Vosko, Can. J. Phys. 44 (1966), DOI: 10.1139/p64-183
        Details in Gregori et al., High Energy Density Phys. 3 (2007)

        Parameters:
            k (float/array): scattering wavenumber in units of a_B^{-1}
            w (array): array of energies in eV

        Returns:
            float/array: output type depending on the input type of k, local field correction, non-dimensional
        """
        kF = self.state.fermi_wave_number(self.state.free_electron_number_density)
        Gamma_ee = self.Gamma_ee
        gammaT = self._gamma_T()
        geeT = self.geeT

        G_k = k**2 / (kF**2 / gammaT + k**2 / (1 - geeT))
        return G_k

    def _pade_interp_static(self, k, w):
        """
        Static local field correction based on G. Gregori et al., High Energy Denity Phys. 3 (2007), DOI: 10.1016/j.hedp.2007.02.006

        Parameters:
            k (float/array): scattering wavenumber in units of a_B^{-1}
            w (array): array of energies in eV, not used here

        Returns:
            float/array: output type depending on the input type of k, local field correction, non-dimensional
        """
        G0 = self._geldart_vosko(k, w)
        GT = self._utsumi_ichimaru_static(k, w)
        G_k = (G0 + self.theta * GT) / (1 + self.theta)
        return G_k

    def _deltas(self):
        x = np.sqrt(self.rs)
        xi_2 = [-2.2963827e-3, 5.6991691e-2, -8.533622e-1, -8.7736539e0, 7.881997e-1, -1.2707788e-2]
        rho_2 = [-7.9968454e1, -1.405268938e2, -3.52575566e1, -1.06331769e1]
        xi_4 = [
            2.30118890e1,
            -6.48378723e1,
            6.35105927e1,
            -1.39457829e1,
            -1.26252782e1,
            1.38524989e1,
            -5.2740937e0,
            1.0156885e0,
            -1.1039532e-2,
        ]
        rho_4 = [9.5753544e0, -3.29770151e1, 4.8252887e1, -3.87189788e1, 2.05595956e1, -6.306675e0]
        delta2 = x * (xi_2[0] + x * (xi_2[1] + x * (xi_2[2] + x * (xi_2[3] + x * (xi_2[4] + x * xi_2[5])))))
        delta2 /= x**4 + rho_2[0] + x * (rho_2[1] + x * (rho_2[2] + x * rho_2[3]))
        delta4 = xi_4[0] + x * (
            xi_4[1]
            + x
            * (xi_4[2] + x * (xi_4[3] + x * (xi_4[4] + x * (xi_4[5] + x * (xi_4[6] + x * (xi_4[7] + x * xi_4[8]))))))
        )
        delta4 /= x**6 + rho_4[0] + x * (rho_4[1] + x * (rho_4[2] + x * (rho_4[3] + x * (rho_4[4] + x * rho_4[5]))))
        delta4 *= delta2
        return delta2, delta4

    def _farid_static(self, k, w):
        """
        Static local field correction based on B. Farid et al., Phys. Rev. B 48 (1993), DOI: 10.1103/physrevb.48.11602.

        Parameters:
            k (float/array): scattering wavenumber in units of a_B^{-1}
            w (array): array of energies in eV, not used here

        Returns:
            float/array: output type depending on the input type of k, local field correction, non-dimensional
        """
        kF = self.state.fermi_wave_number(self.state.free_electron_number_density)
        Q = k / kF
        gee = self.gee0
        omega_F = self.state.fermi_frequency(self.state.free_electron_number_density, ELECTRON_MASS)
        omega_p = self.state.plasma_frequency(
            self.state.charge_state, self.state.free_electron_number_density, ELECTRON_MASS
        )

        y = (omega_F / omega_p) ** 2

        delta2, delta4 = self._deltas()

        gamma0 = self._gamma_0()
        b0A = 2 / 3 * (1 - gee)
        b0B = 48 / 35 * y * delta4
        b0C = -16 / 25 * y * (2 * delta2 + delta2**2)
        b0 = b0A + b0B + b0C
        bneg2 = 4 / 5 * y * delta2
        a = 0.029

        A = 63 / 64 * a + 15 / 4096 * (b0A - 2 * (b0B + b0C) - 16 * bneg2)
        B = 9 / 16 * gamma0 + 7 / 16 * bneg2 - 3 / 64 * b0 - 16 / 15 * A
        C = -3 / 4 * gamma0 + 3 / 4 * bneg2 + 9 / 16 * b0 - 16 / 5 * A
        D = 9 / 16 * gamma0 - 9 / 16 * bneg2 - 3 / 64 * b0 + 8 / 5 * A

        G_k = (
            A * Q**4
            + B * Q**2
            + C
            + (A * Q**4 + D * Q**2 - C)
            * (4 * kF**2 - k**2)
            / (4 * kF * k)
            * np.log(np.abs((2 * kF + k) / (2 * kF - k)))
        )
        return G_k

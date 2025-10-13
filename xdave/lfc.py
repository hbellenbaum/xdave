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

from plasma_state import PlasmaState, get_rho_T_from_rs_theta
from unit_conversions import *
from constants import *
import fxc as gdb
from models import ModelOptions

import numpy as np

from scipy import integrate
from scipy.special import i1 as mod_bessel_first


class LFC:

    def __init__(self, state: PlasmaState):
        self.state = state
        self.theta = state.theta
        self.rs = state.rs

        if state.electron_number_density == 0.0:
            print(f"Cannot calculate LFC.")
        else:
            self.initialize(state=state)

    def initialize(self, state: PlasmaState):
        self.z = 4 * (4 / (9 * PI)) ** (1 / 6) * np.sqrt(self.rs / PI)

        self.Gamma_ee = state.electron_electron_coupling_parameter(
            number_density=state.free_electron_number_density, temperature=state.electron_temperature
        )
        self.gee0 = self._ee_pair_distribution_function(z=self.z)
        self.geeT = self._ee_pair_distribution_function_finite_T(Te=state.electron_temperature)

    def calculate_lfc(self, k, w, model="DORNHEIM_ESA"):
        if self.state.charge_state == 0:
            return 0.0
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
            raise NotImplementedError(f"Model {model} not a recognized option.")

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
        xi = (
            ELECTRON_MASS
            * ELEMENTARY_CHARGE**4
            / ((4 * PI * VACUUM_PERMITTIVITY) ** 2 * BOLTZMANN_CONSTANT * Te * DIRAC_CONSTANT**2)
        )
        Gamma_ee = self.Gamma_ee
        C_sc = 1.0754
        H0 = C_sc * np.sqrt(Gamma_ee) / (1.0 + (C_sc / (Gamma_ee * SQRT_THREE)) ** 4) ** 0.25

        def integral(u):
            X = np.tan(u * PI / 2)
            dXdu = PI / 2 * 1 / (np.cos(PI * u / 2)) ** 2
            return dXdu * (X * np.exp(-xi * X**2)) / (np.expm1(PI / X)) if X > 0 else 0.0

        gbin = np.sqrt(2 * PI) * xi ** (3 / 2) * integrate.quad(integral, 0, 1, limit=200)[0]
        geeT = gbin * np.exp(H0)
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
        return 0.25 - PI * (4 / (9 * PI)) ** (1 / 3) / 24 * (rs**3 * diff2 - 2 * rs * diff1)

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
            gdb.Groth_A(rs, theta)
            * x
            * x
            * (1.0 + alpha * x + beta * x**0.5)
            / (1.0 + gamma * x + delta * x**1.25 + gdb.Groth_A(rs, theta) * x**2)
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
        T. Dornheim et al., Phys. Rev. Lett. 125 (2020), DOI: 10.1103/physrevlett.125.235001
        T. Dornheim et al., Phys. Rev. B 103 (2021), DOI: 10.1103/physrevb.103.165102

        Code copied over from: https://github.com/ToDor90/LFC (and modified slightly)
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
        K. Utsumi and S. Ichimaru, Phys. Rev. A, 26 (1982), DOI: 10.1103/physreva.26.603
        Details also in Gregori et al., High Energy Density Phys. 3 (2007)
        and Farid et al., Phys. Rev. B 48 (1993)
        """
        kF = self.state.fermi_wave_number(self.state.free_electron_number_density)
        Q = k / kF
        gee0 = self.gee0

        gamma0 = self._gamma_0()

        A = 0.029
        B = 9 / 16 * gamma0 - 3 / 64 * (1 - gee0) - 16 / 15 * A
        C = -3 / 4 * gamma0 + 9 / 16 * (1 - gee0) - 16 / 5 * A

        if np.isclose(Q, 2.0, rtol=1.0e-4):
            G_k = A * Q**4 + B * Q**2 + C
        else:
            G_k = (
                A * Q**4
                + B * Q**2
                + C
                + (A * Q**4 + (B + 8 * A / 3) * Q**2 - C) * (4 - Q**2) / (4 * Q) * np.log(np.abs((2 + Q) / (2 - Q)))
            )
        return G_k

    def _geldart_vosko(self, k, w):
        """
        D.J.W. Geldart, S.H. Vosko, Can. J. Phys. 44 (1966), DOI: 10.1139/p64-183
        Details in Gregori et al., High Energy Density Phys. 3 (2007)
        """
        kF = self.state.fermi_wave_number(self.state.free_electron_number_density)
        geeT = self.geeT
        gammaT = self._gamma_T()
        G_k = k**2 / (kF**2 / gammaT + k**2 / (1 - geeT))
        return G_k

    def _pade_interp_static(self, k, w):
        """
        G. Gregori et al., High Energy Denity Phys. 3 (2007), DOI: 10.1016/j.hedp.2007.02.006
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
        B. Farid et al., Phys. Rev. B 48 (1993), DOI: 10.1103/physrevb.48.11602
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


def test():
    import matplotlib.pyplot as plt

    ks = np.linspace(0.01, 6, 100) / BOHR_RADIUS
    rs = 2
    theta = 1

    rho, T = get_rho_T_from_rs_theta(rs=rs, theta=theta)
    rho *= g_per_cm3_TO_kg_per_m3
    T *= eV_TO_K

    state = PlasmaState(
        electron_temperature=T, ion_temperature=T, mass_density=rho, charge_state=2.0, atomic_mass=1, atomic_number=2
    )

    lfc_interp = np.zeros_like(ks)
    lfc_ui = np.zeros_like(ks)
    lfc_gv = np.zeros_like(ks)
    lfc_dornheim = np.zeros_like(ks)
    lfc_farid = np.zeros_like(ks)

    for i in range(0, len(ks)):
        k = ks[i]
        models = ModelOptions()
        kernel = LFC(models=models, state=state)
        lfc_interp[i] = kernel.calculate_lfc(k=k, w=0.0, model="PADE_INTERP")
        lfc_ui[i] = kernel.calculate_lfc(k=k, w=0.0, model="UI")
        lfc_gv[i] = kernel.calculate_lfc(k=k, w=0.0, model="GV")
        lfc_dornheim[i] = kernel.calculate_lfc(k=k, w=0.0, model="DORNHEIM_ESA")
        lfc_farid[i] = kernel.calculate_lfc(k=k, w=0.0, model="FARID")

    kF = state.fermi_wave_number(state.free_electron_number_density)
    plt.figure()
    plt.plot(ks / kF, lfc_interp, label="Interp")
    plt.plot(ks / kF, lfc_ui, label="UI")
    plt.plot(ks / kF, lfc_gv, label="GV")
    plt.plot(ks / kF, lfc_farid, label="Farid")
    plt.plot(ks / kF, lfc_dornheim, label="ESA")
    plt.xlabel(r"$k/k_F$")  # [$a_B^{-1}$]")
    plt.ylabel(r"$G_{ee}(k)$")
    plt.legend()
    plt.show()


def test_gregori_2007():
    import matplotlib.pyplot as plt

    ne = 2.5e23  # cm^{-3}
    rs = 1.86
    T1 = 20  # eV
    T2 = 4  # eV

    rho, _ = get_rho_T_from_rs_theta(rs=rs, theta=1)

    state1 = PlasmaState(
        electron_temperature=T1 * eV_TO_K,
        mass_density=rho * g_per_cm3_TO_kg_per_m3,
        ion_temperature=T1 * eV_TO_K,
        charge_state=1.0,
        atomic_mass=1,
        atomic_number=1,
    )

    print(f"ne = {state1.free_electron_number_density * per_m3_TO_per_cm3} 1/cc")
    print(rf"$\theta$ = {state1.theta}")

    state2 = PlasmaState(
        electron_temperature=T2 * eV_TO_K,
        mass_density=rho * g_per_cm3_TO_kg_per_m3,
        ion_temperature=T2 * eV_TO_K,
        charge_state=1.0,
        atomic_mass=1,
        atomic_number=1,
    )

    print(f"ne = {state2.free_electron_number_density * per_m3_TO_per_cm3} 1/cc")
    print(f"$\\theta$ = {state2.theta}")

    kernel1 = LFC(models=ModelOptions(), state=state1)
    kernel2 = LFC(models=ModelOptions(), state=state2)

    ks1 = np.linspace(0, 6, 500) * state1.fermi_wave_number(state1.free_electron_number_density)
    ks2 = np.linspace(0, 6, 500) * state2.fermi_wave_number(state2.free_electron_number_density)
    lfcs1 = np.zeros_like(ks1)
    lfcs2 = np.zeros_like(ks2)

    for i in range(0, len(ks1)):
        lfcs1[i] = kernel1.calculate_lfc(k=ks1[i], w=0, model="PADE_INTERP")
    for i in range(0, len(ks2)):
        lfcs2[i] = kernel2.calculate_lfc(k=ks2[i], w=0, model="PADE_INTERP")

    datT1 = np.genfromtxt(f"validation/lfc/Gregori_2007_Fig1a_rs_1.86_T_20eV.csv", delimiter=",")
    datT2 = np.genfromtxt(f"validation/lfc/Gregori_2007_Fig1a_rs_1.86_T_4eV.csv", delimiter=",")

    plt.figure()
    plt.plot(
        ks1 / state1.fermi_wave_number(state1.free_electron_number_density), lfcs1, label=f"T=20", ls="-.", c="navy"
    )
    plt.plot(datT1[:, 0], datT1[:, 1], label=f"Gregori et al., T=20", ls="solid", c="navy")
    plt.plot(
        ks2 / state2.fermi_wave_number(state2.free_electron_number_density), lfcs2, label=f"T=4", ls="-.", c="crimson"
    )
    plt.plot(datT2[:, 0], datT2[:, 1], label=f"Gregori et al., T=20", ls="solid", c="crimson")
    plt.legend()
    plt.show()
    # plt.savefig(f"gregori_test.pdf", dpi=200)


def test_fortmann_2010():
    import matplotlib.pyplot as plt

    ne = 2.5e23  # cm^{-3}
    rs = 2
    T = 10  #

    rho, _ = get_rho_T_from_rs_theta(rs=rs, theta=1)

    state = PlasmaState(
        electron_temperature=T * eV_TO_K,
        mass_density=rho * g_per_cm3_TO_kg_per_m3,
        ion_temperature=T * eV_TO_K,
        charge_state=1.0,
        atomic_mass=1,
        atomic_number=1,
    )

    kernel = LFC(models=ModelOptions(), state=state)

    ks = np.linspace(0, 4, 100) * state.fermi_wave_number(state.free_electron_number_density)
    lfcs_iu = np.zeros_like(ks)
    lfcs_FARID = np.zeros_like(ks)

    for i in range(0, len(ks)):
        lfcs_iu[i] = kernel.calculate_lfc(k=ks[i], w=0, model="UI")
        lfcs_FARID[i] = kernel.calculate_lfc(k=ks[i], w=0, model="FARID")

    dat_iu = np.genfromtxt(f"validation/lfc/Fortmann_2010_Fig2_utsumi_ichimaru.csv", delimiter=",")
    dat_farid = np.genfromtxt(f"validation/lfc/Fortmann_2010_Fig2_farid.csv", delimiter=",")

    plt.figure()
    plt.plot(ks / state.fermi_wave_number(state.free_electron_number_density), lfcs_iu, label=f"UI", ls="-.", c="navy")
    plt.plot(dat_iu[:, 0], dat_iu[:, 1], label=f"Fortmann et al., UI", ls="solid", c="navy")
    plt.plot(
        ks / state.fermi_wave_number(state.free_electron_number_density),
        lfcs_FARID,
        label=f"FARID",
        ls="-.",
        c="crimson",
    )
    plt.plot(dat_farid[:, 0], dat_farid[:, 1], label=f"Fortmann et al., Farid", ls="solid", c="crimson")
    plt.legend()
    plt.show()
    # plt.savefig(f"fortmann_test.pdf", dpi=200)


def test_farid():
    import matplotlib.pyplot as plt

    rss = np.array([1, 2, 5, 10, 15])

    colors = ["navy", "crimson", "magenta", "dodgerblue", "limegreen"]

    plt.figure()

    for rs, c in zip(rss, colors):
        rho, T = get_rho_T_from_rs_theta(rs=rs, theta=1)
        state = PlasmaState(
            electron_temperature=T * eV_TO_K,
            mass_density=rho * g_per_cm3_TO_kg_per_m3,
            ion_temperature=T * eV_TO_K,
            charge_state=1.0,
            atomic_mass=1,
            atomic_number=1,
        )
        kF = state.fermi_wave_number(state.free_electron_number_density)
        ks = np.linspace(0, 5, 500) * kF
        lfcs = np.zeros_like(ks)
        lfcs_iu = np.zeros_like(ks)

        kernel = LFC(models=ModelOptions(), state=state)

        for i in range(0, len(ks)):
            lfcs[i] = kernel.calculate_lfc(k=ks[i], w=0, model="FARID")
            lfcs_iu[i] = kernel.calculate_lfc(k=ks[i], w=0, model="UI")

        fn = f"validation/lfc/validation_data/Farid_et_al_Geek0_rs={rs:.0f}.csv"
        dat = np.genfromtxt(fn, delimiter=",")
        plt.plot(ks / kF, lfcs, label=f"rs={rs}", ls="-.", c=c)
        plt.plot(ks / kF, lfcs_iu, label=f"UI: rs={rs}", ls=":", c=c)
        plt.plot(dat[:, 0], dat[:, 1], label=f"Farid et al., rs={rs}", ls="solid", c=c)

    plt.legend()
    plt.ylabel(r"$G_{ee}(k)$")
    plt.xlabel(r"$k/k_F$")
    plt.show()


def test_dornheim_2021():
    import matplotlib.pyplot as plt

    rs = 2
    theta1 = 1
    theta2 = 4

    rho1, T1 = get_rho_T_from_rs_theta(rs=rs, theta=theta1)
    rho2, T2 = get_rho_T_from_rs_theta(rs=rs, theta=theta2)

    state1 = PlasmaState(
        electron_temperature=T1 * eV_TO_K,
        mass_density=rho1 * g_per_cm3_TO_kg_per_m3,
        ion_temperature=T1 * eV_TO_K,
        charge_state=1.0,
        atomic_mass=1,
        atomic_number=1,
    )
    kernel1 = LFC(models=ModelOptions(), state=state1)

    state2 = PlasmaState(
        electron_temperature=T2 * eV_TO_K,
        mass_density=rho2 * g_per_cm3_TO_kg_per_m3,
        ion_temperature=T2 * eV_TO_K,
        charge_state=1.0,
        atomic_mass=1,
        atomic_number=1,
    )
    kernel2 = LFC(models=ModelOptions(), state=state2)

    ks = np.linspace(0, 7, 100) * state1.fermi_wave_number(state1.free_electron_number_density)
    lfc_theta1 = np.zeros_like(ks)
    lfc_theta2 = np.zeros_like(ks)

    for i in range(0, len(ks)):
        lfc_theta1[i] = kernel1.calculate_lfc(k=ks[i], w=0, model="DORNHEIM_ESA")
        lfc_theta2[i] = kernel2.calculate_lfc(k=ks[i], w=0, model="DORNHEIM_ESA")

    if rs == 2:
        fn = f"validation/lfc/Dornheim_2021_Fig7b"
        dat_theta1 = np.genfromtxt(fn + f"_theta_{theta1:.0f}.csv", delimiter=",")
        dat_theta2 = np.genfromtxt(fn + f"_theta_{theta2:.0f}.csv", delimiter=",")
    elif rs == 5:
        fn = f"validation/lfc/validation_data/Dornheim_et_al_Geek0_rs=5"
        dat_theta1 = np.genfromtxt(fn + f"_Theta={theta1:.0f}.csv", delimiter=",")
        dat_theta2 = np.genfromtxt(fn + f"_Theta={theta2:.0f}.csv", delimiter=",")

    plt.figure()
    plt.plot(
        ks / state1.fermi_wave_number(state1.free_electron_number_density), lfc_theta1, label=rf"$\theta$={theta1}"
    )
    plt.plot(
        ks / state2.fermi_wave_number(state2.free_electron_number_density), lfc_theta2, label=rf"$\theta$={theta2}"
    )
    plt.plot(dat_theta1[:, 0], dat_theta1[:, 1], label=f"Dornheim et al., theta={theta1}")
    plt.plot(dat_theta2[:, 0], dat_theta2[:, 1], label=f"Dornheim et al., theta={theta2}")
    plt.title(rf"$r_s$={rs}")
    plt.ylabel(r"$G_{ee}(k)$")
    plt.xlabel(r"$k/k_F$")
    plt.legend()
    plt.show()
    # plt.savefig(f"dornheim_test.pdf", dpi=200)


def test_ui():
    import matplotlib.pyplot as plt

    rss = np.array([1, 4, 10])

    colors = ["navy", "crimson", "magenta", "dodgerblue", "limegreen"]

    plt.figure()

    for rs, c in zip(rss, colors):
        rho, T = get_rho_T_from_rs_theta(rs=rs, theta=1)
        state = PlasmaState(
            electron_temperature=T * eV_TO_K,
            mass_density=rho * g_per_cm3_TO_kg_per_m3,
            ion_temperature=T * eV_TO_K,
            charge_state=1.0,
            atomic_mass=1,
            atomic_number=1,
        )
        kF = state.fermi_wave_number(state.free_electron_number_density)
        ks = np.linspace(0, 5, 500) * kF
        # lfcs = np.zeros_like(ks)
        lfcs_iu = np.zeros_like(ks)

        kernel = LFC(models=ModelOptions(), state=state)

        for i in range(0, len(ks)):
            # lfcs[i] = kernel.calculate_lfc(k=ks[i], w=0, model="FARID")
            lfcs_iu[i] = kernel.calculate_lfc(k=ks[i], w=0, model="UI")

        fn = f"validation/lfc/validation_data/Utsumi_Ichimaru_Geek0_rs={rs:.0f}.csv"
        dat = np.genfromtxt(fn, delimiter=",")
        # plt.plot(ks / kF, lfcs, label=f"rs={rs}", ls="-.", c=c)
        plt.plot(ks / kF, lfcs_iu, label=f"UI: rs={rs}", ls=":", c=c)
        plt.plot(dat[:, 0], dat[:, 1], label=f"Ichimaru et al., rs={rs}", ls="solid", c=c)

    plt.legend()
    plt.xlim(0, 5)
    plt.ylabel(r"$G_{ee}(k)$")
    plt.xlabel(r"$k/k_F$")
    plt.show()
    # plt.savefig(f"ui_test.pdf", dpi=200)


def test_gv():
    import matplotlib.pyplot as plt

    # rs1 = 2
    # rs2 = 3
    theta = 1
    rss = np.array([2, 3])

    colors = ["navy", "crimson", "magenta", "dodgerblue", "limegreen"]

    plt.figure()

    for rs, c in zip(rss, colors):
        rho, T = get_rho_T_from_rs_theta(rs=rs, theta=1)
        state = PlasmaState(
            electron_temperature=T * eV_TO_K,
            mass_density=rho * g_per_cm3_TO_kg_per_m3,
            ion_temperature=T * eV_TO_K,
            charge_state=1.0,
            atomic_mass=1,
            atomic_number=1,
        )
        kF = state.fermi_wave_number(state.free_electron_number_density)
        ks = np.linspace(0, 5, 500) * kF
        # lfcs = np.zeros_like(ks)
        lfcs_iu = np.zeros_like(ks)

        kernel = LFC(models=ModelOptions(), state=state)

        for i in range(0, len(ks)):
            # lfcs[i] = kernel.calculate_lfc(k=ks[i], w=0, model="FARID")
            lfcs_iu[i] = kernel.calculate_lfc(k=ks[i], w=0, model="GV")

        fn = f"/home/bellen85/code/dev/xdave/mcss_tests/mcss_outputs_lfc/lfc=gv_rs={rs:.0f}_theta=1.csv"
        dat = np.genfromtxt(fn, delimiter=",", skip_header=1)
        # plt.plot(ks / kF, lfcs, label=f"rs={rs}", ls="-.", c=c)
        plt.plot(ks / kF, lfcs_iu, label=f"GV: rs={rs}", ls=":", c=c)
        plt.plot(dat[:, 0], dat[:, -1], label=f"MCSS, rs={rs}", ls="solid", c=c)

    plt.legend()
    plt.xlim(0, 5)
    plt.ylabel(r"$G_{ee}(k)$")
    plt.xlabel(r"$k/k_F$")
    plt.show()


if __name__ == "__main__":
    # test()
    test_gv()
    test_gregori_2007()
    test_fortmann_2010()
    test_dornheim_2021()
    test_farid()
    test_ui()

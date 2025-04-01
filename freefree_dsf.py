from constants import *
from unit_conversions import *
from maths import log1pexp

# from fermi_integrals import fermi_integral
from plasma_state import PlasmaState
from models import ModelOptions
import numpy as np


from plasmapy.formulary.mathematics import Fermi_integral as fdi


def effective_coulomb_potential(ionisation, wave_number):
    r"""
    Effective Coulomb potential
    """
    V_aa = -4 * PI * ionisation * ELEMENTARY_CHARGE_SQR * UNIT_COULOMB_POTENTIAL / (wave_number * wave_number)
    return V_aa


# def lindhard_rpa(state: PlasmaState, wave_number):
#     """
#     Limiting case for full degeneracy
#     """
#     z = state.frequency / (4 * state.fermi_frequency())
#     q = (wave_number / 2) / state.fermi_wave_number()
#     x_pos = z / q + q
#     x_neg = z / q - q

#     def func(x):
#         return x + 0.5 * (1 - x**2) * np.log((x + 1) / (x - 1))

#     pol_func = 3 * state.electron_number_density / (4 * state.fermi_energy() * q) * (func(x_pos) - func(x_neg))
#     return pol_func


class FreeFreeDSF:

    def __init__(self, state: PlasmaState) -> None:
        self.state = state

    def get_dsf(self, k, w, lfc, model):
        pol_func = self.polarisation_function(k, w, model=model)
        potential_func = 4 * np.pi * COULOMB_CONSTANT * ELEMENTARY_CHARGE**2 / k**2
        dielectric_func = 1 - potential_func * pol_func
        S_EG = (
            1
            / (1 - np.exp(-w / (BOLTZMANN_CONSTANT * self.state.electron_temperature)))
            * (k**2 / (4 * PI_SQR * ELEMENTARY_CHARGE_SQR))
            * np.imag(-1 / dielectric_func)
        )
        return S_EG

    def polarisation_function(self, k, w, model="LINDHARD"):
        if model == "LINDHARD":
            return self.lindhard_pol_func(k=k, w=w)
        elif model == "FIT":
            return self.dandrea_fit(k=k, omega=w)
        else:
            raise NotImplementedError(f"Model {model} not recognized. Try LINDHARD for now.")

    def lindhard_pol_func(self, k, w):
        EF = self.state.fermi_energy(self.state.electron_number_density, ELECTRON_MASS)
        kF = self.state.fermi_wave_number(self.state.electron_number_density)
        omega_p = self.state.plasma_frequency(self.state.mass_density, self.state.atomic_mass)
        gamma = EF / (DIRAC_CONSTANT * omega_p)
        Z = np.abs(k / (2 * kF))
        U = w / (4 * Z * EF)
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

    def lindhard_pol_func_tc(self, k, w):
        EF = self.state.fermi_energy(self.state.electron_number_density)
        kF = self.state.fermi_wave_number(self.state.electron_number_density)
        q0 = k / kF
        nu0 = w / EF
        nu = nu0.astype("complex128")

        qplus_sq = nu + q0**2
        qminus_sq = nu - q0**2
        xplus = (nu + q0**2) / 2.0 / q0
        xminus = (nu - q0**2) / 2.0 / q0
        result = (
            1
            - (4 * q0**2 - qminus_sq**2) / 8 / q0**3 * log(-1 * (1 + xminus) / (1 - xminus))
            + (4 * q0**2 - qplus_sq**2) / 8 / q0**3 * log(-1 * (1 + xplus) / (1 - xplus))
        )

        # log(a+ib) = log(sqrt(a*a+b+b))+ i arg(a+ib)
        # numpy complex log pole cut convention makes this tricky
        # so for a<0 and b<=0, then arg approx -pi,
        # but for a<0 and b>0, then arg approx +pi
        if np.allclose(nu0.imag, 0.0):
            result = np.conjugate(result)

        return result

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

        EF = self.state.fermi_energy(self.state.electron_number_density, ELECTRON_MASS)
        kF = self.state.fermi_wave_number(self.state.electron_number_density)
        Theta_e = self.state.alt_degeneracy_parameter(
            self.state.electron_number_density, self.state.electron_temperature, ELECTRON_MASS
        )

        sqrt_Theta_e = np.sqrt(Theta_e)

        q0 = 0.5 * k / kF
        w0 = 0.25 * omega / (EF * q0)  #  * DIRAC_CONSTANT
        w = w0 / sqrt_Theta_e
        eta = self.state.chemical_potential(
            self.state.electron_temperature, self.state.electron_number_density, ELECTRON_MASS
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
        Pi_aa_0_MB = self.state.electron_number_density / (BOLTZMANN_CONSTANT * self.state.electron_temperature)
        pol_func = Pi_aa_0_MB * (Re_X_kw + 1.0j * Im_X_kw) / (4.0 * F_p0p5 * q)

        return pol_func


def test():
    import matplotlib.pyplot as plt

    lfc = 0
    Te = 30 * eV_TO_K
    rho = 1.0 * g_per_cm3_TO_kg_per_m3
    charge_state = 1.0
    atomic_number = 1
    atomic_mass = 1.0
    # k = 1.02e9  # e11  # 1/m
    scattering_angle = 30
    E0 = 2.96 * eV_TO_J
    k = 2 * E0 / (DIRAC_CONSTANT * SPEED_OF_LIGHT) * np.sin(scattering_angle / 2)
    k2 = 2 * E0 / (DIRAC_CONSTANT * SPEED_OF_LIGHT) * np.sin(60 / 2)
    k3 = 2 * E0 / (DIRAC_CONSTANT * SPEED_OF_LIGHT) * np.sin(120 / 2)

    omega_array = np.linspace(-100, 700, 500) * eV_TO_J  # + 8.5 * eV_TO_J
    ks = [k, k2, k3]  # [1.0e8, 1.0e9, 1.0e10, 1.0e11, 1.0e12]
    state = PlasmaState(
        electron_temperature=Te,
        ion_temperature=Te,
        mass_density=rho,
        charge_state=charge_state,
        # frequency=omega,
        atomic_mass=atomic_mass,
        atomic_number=atomic_number,
    )

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for k in ks:
        dsfs = []
        for omega in omega_array:
            kernel = FreeFreeDSF(state=state)  # , models=ModelOptions)
            dsf = kernel.get_dsf(k=k, w=omega, lfc=lfc, model="FIT")
            # print(dsf)
            dsfs.append(dsf)

        # plt.figure()
        ax.plot(omega_array * J_TO_eV, dsfs, label=f"k={k}")
    ax.legend()
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.set_ylabel(r"$S_{ff}$")
    plt.tight_layout()
    plt.show()
    fig.savefig("initial_ff_results.pdf")


if __name__ == "__main__":
    test()

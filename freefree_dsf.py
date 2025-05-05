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


class FreeFreeDSF:

    def __init__(self, state: PlasmaState, models: ModelOptions) -> None:
        self.state = state
        self.polarisation_model = models.polarisation_model

    def get_dsf(self, k, w, lfc):
        pol_func = self.polarisation_function(k, w)
        potential_func = 4 * np.pi * COULOMB_CONSTANT * ELEMENTARY_CHARGE**2 / k**2
        dielectric_func = 1 - potential_func * pol_func
        S_EG = (
            1
            / (1 - np.exp(-w / (BOLTZMANN_CONSTANT * self.state.electron_temperature)))
            * (k**2 / (4 * PI_SQR * ELEMENTARY_CHARGE_SQR))
            * np.imag(-1 / dielectric_func)
        )
        return S_EG

    def polarisation_function(self, k, w):  # , model="LINDHARD"):
        if self.polarisation_model == "LINDHARD":
            return self.lindhard_pol_func_dc(k=k, w=w)
        elif self.polarisation_model == "DANDREA_FIT":
            return self.dandrea_fit(k=k, omega=w)
        else:
            raise NotImplementedError(f"Model {self.polarisation_model} not recognized. Try LINDHARD for now.")

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

    def lindhard_pol_func_dc(self, k, w):
        EF = self.state.fermi_energy(self.state.electron_number_density, ELECTRON_MASS)
        kF = self.state.fermi_wave_number(self.state.electron_number_density)
        q0 = 0.5 * k / kF
        w0 = 0.25 * w / (EF * q0)

        def lindhard_func(x):
            real_part = -x - 1 / 2 * (1 - x**2) * np.log(np.abs((x + 1) / (x - 1)))
            im_part = HALF_PI * (1 - x**2) * np.heaviside(1.0 - x**2, 1.0)
            return real_part + 1.0j * im_part

        G_plus = lindhard_func(w0 + q0)
        G_minus = lindhard_func(w0 - q0)
        pol_func = 3 * self.state.electron_number_density / (4 * EF * q0) * (G_plus - G_minus)  # / (4 * q0)

        return pol_func

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
    rho = 0.1 * g_per_cm3_TO_kg_per_m3
    charge_state = 1.0
    atomic_number = 1
    atomic_mass = 1.0
    E0 = 4.0e3 * eV_TO_J
    angles_rad = np.array([10, 30, 60, 120]) * np.pi / 180  # , 20, 30, 45, 60, 80, 100, 120, 140
    ks = 2 * E0 / (DIRAC_CONSTANT * SPEED_OF_LIGHT) * np.sin(angles_rad / 2)

    omega_array = np.linspace(-100, 100, 2000) * eV_TO_J  # + 8.5 * eV_TO_J
    state = PlasmaState(
        electron_temperature=Te,
        ion_temperature=Te,
        mass_density=rho,
        charge_state=charge_state,
        # frequency=omega,
        atomic_mass=atomic_mass,
        atomic_number=atomic_number,
    )
    model = "LINDHARD"
    if model == "LINDHARD":
        mcss_model = "LINDHARD_RPA"
    elif model == "DANDREA_FIT":
        mcss_model = "DANDREA_RPA_FIT"
    models = ModelOptions(polarisation_model=model)
    colors = ["magenta", "crimson", "orange", "dodgerblue", "lightgreen", "lightgray", "yellow", "cyan"]
    fig, ax0 = plt.subplots(figsize=(14, 10))
    # xmins = np.array([])
    i = 0
    for k, c in zip(ks, colors):
        angle = angles_rad[i] * 180 / np.pi
        angle = int(np.round(angle, 0))
        dsfs = []
        pols = []
        for omega in omega_array:
            kernel = FreeFreeDSF(state=state, models=models)
            dsf = kernel.get_dsf(k=k, w=omega, lfc=lfc)
            pol_func = kernel.polarisation_function(k=k, w=omega)
            # print(dsf)
            dsfs.append(dsf)
            pols.append(pol_func)

        # plt.figure()

        mcss_fn = f"mcss_tests/mcss_outputs_model={mcss_model}/mcss_ff_test_angle={angle}.csv"
        En, Es, _, wff, wbf, Pff, Pbf, Pel, tot = np.genfromtxt(mcss_fn, unpack=True, delimiter=",", skip_header=1)
        pols = np.array(pols)
        # print(np.max(wff) / eV_TO_J)
        # print(np.max(dsfs) * eV_TO_J)

        ax0.plot(En[::-1], wff[::-1] / np.max(wff), label="MCSS", c=c, ls="dotted")
        ax0.plot(
            omega_array[::-1] * J_TO_eV,
            np.array(dsfs[::-1]) / np.max(dsfs),
            label=f"$\\theta$={angles_rad[i] * 180 / np.pi:.2f}",
            c=c,
        )
        # ax1.plot(
        #     omega_array[::-1] * J_TO_eV,
        #     pols[::-1].real,
        #     label=f"Re: $\\theta$={angles_rad[i] * 180 / np.pi:.2f}",
        #     ls="dashed",
        #     c=c,
        # )
        # ax1.plot(
        #     omega_array[::-1] * J_TO_eV,
        #     pols[::-1].imag,
        #     label=f"Im: $\\theta$={angles_rad[i] * 180 / np.pi:.2f}",
        #     ls="dotted",
        #     c=c,
        # )
        i += 1
    ax0.legend()
    ax0.set_xlim(-100, 100)
    ax0.set_xlabel(r"$\omega$ [eV]")
    ax0.set_ylabel(r"$S_{ff}$")
    # axes[1].legend()
    # axes[1].set_xlabel(r"$\omega$ [eV]")
    # axes[1].set_ylabel(r"§\PI_{ee}$")
    plt.tight_layout()
    plt.show()
    fig.savefig(f"initial_ff_results_model={model}.pdf")


if __name__ == "__main__":
    test()

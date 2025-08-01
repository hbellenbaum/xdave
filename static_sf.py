# Ionic form factors
from models import ModelOptions
from plasma_state import PlasmaState

from constants import BOHR_RADIUS, ELEMENTARY_CHARGE, PI, BOLTZMANN_CONSTANT, VACUUM_PERMITTIVITY
import numpy as np


def eta_func(sigmac, ni):
    return PI / 6 * ni * sigmac**3


def gamma_func(Z, sigmac, Ti):
    gamma = (Z**2 * ELEMENTARY_CHARGE**2) / (4 * PI * VACUUM_PERMITTIVITY * sigmac * BOLTZMANN_CONSTANT * Ti)
    return gamma


def xi_func(eta, gamma):
    return np.sqrt(24 * eta * gamma)


def h0_func(eta, xi, gamma):
    h0 = (1 + 2 * eta) / (1 - eta) * (1 - np.sqrt(1 + (2 * (1 - eta) ** 3 * xi) / (1 + 2 * eta) ** 2))
    return h0


def h1_func(eta, xi, gamma):
    h0 = h0_func(eta, xi, gamma)
    h1 = h0**2 / (24 * eta) - (1 + eta / 2) / (1 - eta) ** 2
    return h1


def h2_func(eta, xi, gamma):
    h0 = h0_func(eta, xi, gamma)
    h2 = -(1 + eta - eta**2 / 5) / (12 * eta) - (1 - eta) * h0 / (12 * eta * xi)
    return h2


def y0_func(eta, gamma, xi):
    h0 = h0_func(eta, gamma, xi)
    y = (
        -((1 + 2 * eta) ** 2) / (1 - eta) ** 4
        + h0**2 / (4 * (1 - eta) ** 2)
        - (1 + eta) * h0 * xi / (12 * eta)
        - (5 + eta**2) / xi**2 / (60 * eta)
    )
    return y


def y1_func(eta, gamma, xi):
    h1 = h1_func(eta, gamma, xi)
    return 6 * eta * h1**2


def y2_func(eta, gamma, xi):
    return xi**2 / 6


def y3_func(eta, gamma, xi):
    y0 = y0_func(eta, gamma, xi)
    h2 = h2_func(eta, gamma, xi)
    return eta / 2 * (y0 + xi**2 * h2)


def y4_func(eta, gamma, xi):
    return eta * xi**2 / 60


def find_sigma_c(ni, Z, Ti):
    from scipy import optimize

    def func(x):
        # b = h0_sqr(x)
        eta = eta_func(x, ni)
        gamma = gamma_func(Z, x, Ti)
        xi = xi_func(eta, gamma)
        # sqrt_term = 1 + 2 * (1 - eta) ** 3 * xi / (1 + 2 * eta) ** 2
        # h0 = (1 + 2 * eta) / (1 - eta) * (1 - np.sqrt(sqrt_term))
        # h1 = h0**2 / (24 * eta) - (1 + eta / 2) / (1 - eta) ** 2
        # return np.abs(h0**2 / (24 * x) - (1 + x / 2) / (1 - x) ** 2)
        h1 = h1_func(eta, gamma, xi)
        return np.abs(h1)

    res = optimize.minimize(func, x0=BOHR_RADIUS, method="Nelder-Mead", tol=1.0e-25)

    print(func(res.x[0]))
    # print(res)

    return res.x[0]


class StaticStructureFactor:

    def __init__(self, state: PlasmaState, models: ModelOptions, sigma=float()):
        self.state = state
        self.ss_model = models.static_structure_factor_approximation
        self.ion_particle_diameter = sigma  # 1 * BOHR_RADIUS  ## this needs to be moved to the plasma state

    def get_ii_static_structure_factor_ocp(self, k):
        if self.ss_model == "MSA":
            return self.mean_spherical_approximation_ocp_ii(k)
        else:
            raise NotImplementedError(
                f"Model {self.ss_model} for the static structure factor not yet implemented. Try MSA :)"
            )

    def get_ii_static_structure_factor(self, k):
        if self.ss_model == "MSA":
            return self.mean_spherical_approximation_ss(k)
        else:
            raise NotImplementedError(
                f"Model {self.ss_model} for the static structure factor not yet implemented. Try MSA :)"
            )

    def mean_spherical_approximation_ocp_ii(self, k):
        # This should really be a user-defined input
        ion_particle_diameter = self.ion_particle_diameter

        q = k * ion_particle_diameter

        # eta = PI / 6 * self.state.ion_number_density * ion_particle_diameter**3
        # gamma = (
        #     self.state.charge_state**2
        #     * ELEMENTARY_CHARGE
        #     / (4 * PI * VACUUM_PERMITTIVITY * ion_particle_diameter * BOLTZMANN_CONSTANT * self.state.ion_temperature)
        # )
        # xi = np.sqrt(24 * eta * gamma)

        sinq = np.sin(q)
        qsinq = q * sinq
        cosq = np.cos(k)
        qcosq = q * cosq
        qsqr = q * q
        qcub = qsqr * q
        qquad = qsqr * qsqr
        # Zstar = self.state.atomic_number - self.state.charge_state

        # sqrt_term = 1 + 2 * (1 - eta) ** 3 * xi / (1 + 2 * eta) ** 2
        # h0 = (1 + 2 * eta) / (1 - eta) * (1 - np.sqrt(sqrt_term))
        # h1 = h0**2 / (24 * eta) - (1 + eta / 2) / (1 - eta) ** 2
        # h2 = -(1 + eta - eta**2 / 5) / (12 * eta) - (1 - eta) * h0 / (12 * eta * xi)
        eta = eta_func(sigmac=ion_particle_diameter, ni=self.state.ion_number_density)
        gamma = gamma_func(Z=self.state.charge_state, sigmac=ion_particle_diameter, Ti=self.state.ion_temperature)
        xi = xi_func(eta=eta, gamma=gamma)
        h0 = h0_func(eta, gamma, xi)
        h1 = h1_func(eta, gamma, xi)
        h2 = h2_func(eta, gamma, xi)
        y0 = y0_func(eta, gamma, xi)
        y1 = y1_func(eta, gamma, xi)
        y2 = y2_func(eta, gamma, xi)
        y3 = y3_func(eta, gamma, xi)
        y4 = y4_func(eta, gamma, xi)

        # y0 = (
        #     -((1 + 2 * eta) ** 2) / (1 - eta) ** 4
        #     + h0**2 / (4 * (1 - eta) ** 2)
        #     - (1 + eta) * h0 * xi / (12 * eta)
        #     - (5 + eta**2) * xi**2 / (60 * eta)
        # )
        # y1 = 6 * eta * h1**2
        # y2 = xi**2 / 6
        # y3 = eta / 2 * (y0 + xi**2 * h2)
        # y4 = eta * xi**2 / 60

        c0 = y0 * qcub * (sinq - qcosq)
        c1 = y1 * qsqr * (2 * qsinq - (q**2 - 2) * cosq - 2)
        c2 = y2 * q * ((3 * qsqr - 6) * sinq - (qsqr - 6) * qcosq)
        c3 = y3 * ((4 * qsqr - 24) * qsinq - (qquad - 12 * qsqr + 24) * cosq + 24)
        c4 = y4 * (6 * (qquad - 20 * qsqr + 120) * qsinq - (q**6 - 30 * qquad + 360 * qsqr - 720) * cosq - 720) / qsqr
        c5 = -gamma * qquad * cosq

        c_ii = 24 * eta / k**6 * (c0 + c1 + c2 + c3 + c4 + c5)
        S_ii_OCP = 1 / (1 - c_ii)
        return S_ii_OCP

    def mean_spherical_approximation_ss(self, k):
        ion_particle_diameter = self.ion_particle_diameter
        S_ii_OCP = self.get_ii_static_structure_factor_ocp(k)
        # S_ee =

        kappa_i = np.sqrt(
            self.state.charge_state
            * self.state.electron_number_density
            * ELEMENTARY_CHARGE**2
            / (VACUUM_PERMITTIVITY * BOLTZMANN_CONSTANT * self.state.ion_temperature)
        )
        kappa_e = np.sqrt(
            self.state.electron_number_density
            * ELEMENTARY_CHARGE**2
            / (VACUUM_PERMITTIVITY * BOLTZMANN_CONSTANT * self.state.electron_temperature)
        )
        # screening_charge = self.state.charge_state * kappa_e**2 / k**2 * S_ee
        # screening_correction = (
        #     kappa_i**2
        #     / k**2
        #     * (np.cos(k * ion_particle_diameter / 2)) ** 2
        #     * screening_charge
        #     / self.state.charge_state
        # )
        # static dielectric function in the weakly coupled limit, should be replaced by RPA or something more sophisticated
        lfc = 0.0
        dielectric = 1 + kappa_e**2 / k**2
        S_ee0 = k**2 / (k**2 - kappa_e**2 * (1 - lfc))
        q_sc = self.state.charge_state * kappa_e**2 / k**2 * S_ee0
        # screening_correction = kappa_i**2 / k**2 * (np.cos(k * ion_particle_diameter / 2)) ** 2 * (1 / dielectric - 1)
        screening_correction = (
            -(kappa_i**2) / k**2 * (np.cos(k * ion_particle_diameter / 2)) ** 2 * kappa_e**2 / k**2 * S_ee0
        )
        S_ii = S_ii_OCP / (1 + screening_correction * S_ii_OCP)
        return S_ii, screening_correction, q_sc, S_ee0


def test():
    from unit_conversions import eV_TO_J, g_per_cm3_TO_kg_per_m3, eV_TO_K, per_cm3_TO_per_m3
    from constants import DIRAC_CONSTANT, SPEED_OF_LIGHT
    import matplotlib.pyplot as plt

    Te = 20 * eV_TO_K
    rho = 2 * g_per_cm3_TO_kg_per_m3
    charge_state = 2.0
    atomic_number = 6
    atomic_mass = 6.0
    E0 = 8.0e3 * eV_TO_J

    # angles_rad = np.array([10, 30, 60, 120]) * np.pi / 180  # , 20, 30, 45, 60, 80, 100, 120, 140
    angles_rad = np.linspace(10, 120, 5000) * np.pi / 180

    state = PlasmaState(
        electron_temperature=Te,
        ion_temperature=Te,
        mass_density=rho,
        charge_state=charge_state,
        atomic_mass=atomic_mass,
        atomic_number=atomic_number,
    )

    kF = state.fermi_wave_number(state.electron_number_density)

    optimal_diameter = find_sigma_c(ni=state.ion_number_density, Z=state.charge_state, Ti=state.ion_temperature)
    print(f"Calculated diameter: {optimal_diameter / BOHR_RADIUS} a_B")

    ks = 2 * E0 / (DIRAC_CONSTANT * SPEED_OF_LIGHT) * np.sin(angles_rad / 2) / kF * 1.0e3
    # ks = np.linspace(0.01, 4, 500) / BOHR_RADIUS  # * kF
    # ks = np.linspace(0, 6, 1000) * kF

    models = ModelOptions()

    kernel = StaticStructureFactor(state=state, models=models, sigma=optimal_diameter)

    sfs_screened = []
    sfs_ocp = []
    fs_k = []
    qs_sc = []
    S_ees = []
    for k in ks:
        sf_screened, f_k, q_sc, S_ee = kernel.get_ii_static_structure_factor(k)
        sf_ocp = kernel.get_ii_static_structure_factor_ocp(k)
        sfs_screened.append(sf_screened)
        sfs_ocp.append(sf_ocp)
        fs_k.append(f_k)
        qs_sc.append(q_sc)
        S_ees.append(S_ee)

    # print(sfs_ocp)
    plt.figure()
    plt.axhline(1.0, c="gray", ls="dashed")
    # plt.plot(ks, sfs_screened, ls="dashed", c="navy", label="Screened")
    # plt.plot(ks / kF, fs_k, ls="dashed", c="crimson", label="Screening correction")
    # plt.plot(ks / kF, qs_sc, ls="dashed", c="orange", label="Screening")
    # plt.plot(ks / kF, S_ees, ls="dashed", c="navy", label="S_ee0")
    plt.plot(ks, sfs_ocp, ls="dashed", c="crimson", label="OCP")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test()

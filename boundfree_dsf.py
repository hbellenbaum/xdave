from ii_ff import PaulingShermanIonicFormFactor, ScreeningConstants
from ipd import get_ipd
from plasma_state import PlasmaState
from models import ModelOptions
from unit_conversions import *
from constants import *
from scipy.special import gamma
import numpy as np


class BoundFreeDSF:
    """
    Currently only doing basic Impulse Approximation :)
    """

    def __init__(self, state: PlasmaState, models: ModelOptions) -> None:
        self.state = state
        self.ff_model = PaulingShermanIonicFormFactor()
        self.screening_constants = ScreeningConstants
        self.bf_model = models.bf_model

    def get_dsf(self, ZA, Zb, k, w, Eb, ipd_model="NONE"):
        """
        Inputs:
            - ZA: Net charge state
            - Zb: Number of bound electrons
            - k: wave number
            - w: frequency
            - Eb: binding energy (array)
            - bf_model: str
            - ipd_model: str
        """

        # Calculate IPD
        ipd = 0.0
        if ipd_model != "NONE":
            ipd = get_ipd(state=self.state, model=ipd_model)
        Eb_eff = Eb + ipd  # "+ ipd" if Eb < 0 and IPD > 0

        # Load correct bf model
        if self.bf_model == "SCHUMACHER":
            Sce = self.schuhmacher_ia(ZA, Zb, k, w, Eb_eff)
        elif self.bf_model == "HR_CORRECTION":
            Sce = self.schumacher_ia_correction(ZA, Zb, k, w, Eb_eff)
        elif self.bf_model == "TRUNCATED_IA":
            Sce = self.truncated_IA(ZA, Zb, k, w, Eb)
        else:
            raise NotImplementedError(f"Model {self.bf_model} not recognised. Try SCHUMACHER :)")
        return Sce / DIRAC_CONSTANT

    def _shell_amplitude(self, Znl, n, l):
        Anl = (
            2.0 ** (4.0 * l + 3.0)
            * gamma(n - l)
            * (n * gamma(l + 1.0)) ** 2.0
            / (np.pi * FINE_STRUCTURE_CONSTANT * Znl * gamma(n + l + 1.0))
        )
        return Anl

    def schuhmacher_ia(self, ZA, Zb, k, w, Eb):
        """
        Bound-free DSF from Schumacher, Smend and Borchert, J. Phys. B 8, 1428 (1975)
        Parameters
        ----------
        ZA: Net charge state
        Zb: Number of bound electrons
        Te: electronic temperature in K
        k: Wave vector in m^-1
        w: Frequency in hz
        EB: electronic binding energy in Joule

        Returns
        -------
        Sce: dynamic structure factor
        """

        Sce = 0.0

        if Zb > 0:
            c1s = 0
            c2s = 0
            c2p = 0
            c3s = 0
            c3p = 0
            c4s = 0
            c3d = 0

            if Zb > 0:
                c1s = int(min([2, Zb]))
            if Zb > 2:
                c2s = int(min([2, Zb - 2]))
            if Zb > 4:
                c2p = int(min([6, Zb - 4]))
            if Zb > 10:
                c3s = int(min([2, Zb - 2]))
            if Zb > 12:
                c3p = int(min([6, Zb - 12]))
            if Zb > 18:
                c4s = int(min([2, Zb - 18]))
            if Zb > 20:
                c3d = int(min([10, Zb - 20]))

            E = np.abs(w)  # * J_TO_eV  # PLANCK_CONSTANT *
            w_freq = np.abs(w) / DIRAC_CONSTANT  # convert the energy range to an actual frequency: E = \hbar \omega

            # Compton frequency
            wC = DIRAC_CONSTANT * k**2 / (2 * ELECTRON_MASS)  # units of s^{-1} = Hz

            # dimensionless scattering wave number
            q = (w_freq - wC) / (SPEED_OF_LIGHT * k)

            J = 0.0

            if c1s > 0:
                n = 1
                l = 0

                Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)  # [Znl] = [#]
                xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)  # [xnl] =

                Jnl = self._shell_amplitude(Znl, n, l) * xnl**3 / 3

                for i in range(c1s):
                    if Eb[i] >= 0:
                        continue
                    J = J + Jnl * np.heaviside(E + Eb[i], 1)  # np.heaviside(E, Eb[i])

            if c2s > 0:

                n = 2
                l = 0

                Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)
                xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)

                Jnl = (
                    self._shell_amplitude(Znl, n, l)
                    * 4.0
                    * (xnl**3.0 / 3.0 - xnl**4.0 + 4.0 * xnl**5.0 / 5.0)
                    * 1.0e-6
                )

                for i in range(c2s):
                    if Eb[i + 2] >= 0:
                        continue
                    # J = J + Jnl * np.heaviside(E, Eb[i + 2 - 1])
                    J = J + Jnl * np.heaviside(E + Eb[i + 2], 1)

            if c2p > 0:

                n = 2
                l = 1

                Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)
                xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)

                Jnl = self._shell_amplitude(Znl, n, l) * (xnl**4.0 / 4.0 - xnl**5.0 / 5.0)

                for i in range(c2p):
                    if Eb[i + 4] >= 0:
                        continue
                    # J = J + Jnl * np.heaviside(E, Eb[i + 4 - 1])
                    J = J + Jnl * np.heaviside(E + Eb[i + 4], 1)

            if c3s > 0:

                n = 3
                l = 0

                Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)
                xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)
                Jnl = self._shell_amplitude(Znl, n, l) * (
                    3.0 * xnl**3.0
                    - 2.4e1 * xnl**4.0
                    + 3.52e2 * xnl**5.0 / 5.0
                    - 2.56e2 * xnl**6.0 / 3.0
                    + 2.56e2 * xnl**7.0 / 7.0
                )

                for i in range(c3s):
                    if Eb[i + 10] >= 0:
                        continue
                    # J = J + Jnl * np.heaviside(E, Eb[i + 10 - 1])
                    J = J + Jnl * np.heaviside(E + Eb[i + 10], 1)

            if c3p > 0:

                n = 3
                l = 1

                Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)
                xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)
                Jnl = (
                    self._shell_amplitude(Znl, n, l)
                    * 1.6e1
                    * (xnl**4.0 / 4.0 - xnl**5.0 + 4.0 * xnl**6.0 / 3.0 - 4.0 * xnl**7.0 / 7.0)
                )

                for i in range(c3p):
                    if Eb[i + 12] >= 0:
                        continue
                    # J = J + Jnl * np.heaviside(E, Eb[i + 12 - 1])
                    J = J + Jnl * np.heaviside(E + Eb[i + 12], 1)

            if c4s > 0:
                n = 4
                l = 0
                Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)
                xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)

                Jnl = (
                    self._shell_amplitude(Znl, n, l)
                    * 1.6e1
                    * (
                        xnl**3.0 / 3.0
                        - 5.0 * xnl**4.0
                        + 1.48e2 * xnl**5.0 / 5.0
                        - 2.56e2 * xnl**6.0 / 3.0
                        + 1.28e2 * xnl**7.0
                        - 9.6e1 * xnl**8.0
                        + 2.56e2 * xnl**9.0 / 9.0
                    )
                )

                for i in range(c4s):
                    if Eb[i + 18] >= 0:
                        continue
                    # J = J + Jnl * np.heaviside(E, Eb[i + 18 - 1])
                    J = J + Jnl * np.heaviside(E + Eb[i + 18], 1)

            if c3d > 0:

                n = 3
                l = 2

                Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)
                xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)
                Jnl = self._shell_amplitude(Znl, n, l) * (xnl**5.0 / 5.0 - xnl**6.0 / 3.0 + xnl**7.0 / 7.0)

                for i in range(c3d):
                    if Eb[i + 20] >= 0:
                        continue
                    # J = J + Jnl * np.heaviside(E, Eb[i + 20 - 1])
                    J = J + Jnl * np.heaviside(E + Eb[i + 20], 1)

            Sce = J / (SPEED_OF_LIGHT * k)

            # Detailed balanced
            Sce = np.where(w < 0, np.exp(-E / (BOLTZMANN_CONSTANT * self.state.electron_temperature)) * Sce, Sce)

        return Sce

    def schumacher_ia_correction(self, ZA, Zb, k, w, Eb):
        """
        First order correction to the bound-free Schumacher Impulse Approximation based on
        Holm and Ribberfors, Phys. Rev. A 40 (1989)
        ----------
        ZA: Net charge state
        Zb: Number of bound electrons
        Te: electronic temperature in K
        k: Wave vector in m^-1
        w: Frequency in hz
        EB: electronic binding energy in Joule

        Returns
        -------
        Sce: dynamic structure factor
        """

        Sce = 0.0

        if Zb > 0:
            c1s = 0
            c2s = 0
            c2p = 0
            c3s = 0
            c3p = 0
            c4s = 0
            c3d = 0

            if Zb > 0:
                c1s = min([2, Zb])
            if Zb > 2:
                c2s = min([2, Zb - 2])
            if Zb > 4:
                c2p = min([6, Zb - 4])
            if Zb > 10:
                c3s = min([2, Zb - 2])
            if Zb > 12:
                c3p = min([6, Zb - 12])
            if Zb > 18:
                c4s = min([2, Zb - 18])
            if Zb > 20:
                c3d = min([10, Zb - 20])

            E = np.abs(w)  # * J_TO_eV  # PLANCK_CONSTANT *
            w_freq = np.abs(w) / DIRAC_CONSTANT  # convert the energy range to an actual frequency: E = \hbar \omega

            # Compton frequency
            wC = DIRAC_CONSTANT * k**2 / (2 * ELECTRON_MASS)  # units of s^{-1} = Hz

            # dimensionless scattering wave number
            q = (w_freq - wC) / (SPEED_OF_LIGHT * k)

            J = 0.0

            if c1s > 0:
                n = 1
                l = 0

                Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)  # [Znl] = [#]
                xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)  # [xnl] =

                Jnl0 = self._shell_amplitude(Znl, n, l) * xnl**3 / 3
                Jnl1 = Jnl0 * (1.5 * xnl - 2.0 * np.arctan(xnl)) / (k * BOHR_RADIUS / (Znl * FINE_STRUCTURE_CONSTANT))

                for i in range(c1s):
                    if Eb[i] >= 0:
                        continue
                    J = J + (Jnl0 + Jnl1) * np.heaviside(E + Eb[i], 1)  # np.heaviside(E, Eb[i])

            if c2s > 0:

                n = 2
                l = 0

                Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)
                xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)

                Jnl0 = (
                    6.4e1
                    * (
                        1.0 / (3.0 * (1.0 + xnl**2.0) ** 3.0)
                        - 1.0 / (1.0 + xnl**2) ** 4
                        + 4.0 / (5.0 * (1.0 + xnl**2.0) ** 5.0)
                    )
                    / (PI * Znl * FINE_STRUCTURE_CONSTANT)
                )
                Jnl1 = (
                    Jnl0
                    * (
                        5.0 * xnl * (1.0 + 3.0 * xnl**4.0) / (1.0 - 2.5 * xnl**2.0 + 2.5 * xnl**4.0) / 8.0
                        - 2.0 * np.arctan(xnl)
                    )
                    / (k * BOHR_RADIUS / (Znl * FINE_STRUCTURE_CONSTANT))
                )

                for i in range(c2s):
                    if Eb[i + 2] >= 0:
                        continue
                    # J = J + Jnl * np.heaviside(E, Eb[i + 2 - 1])
                    J = J + (Jnl0 + Jnl1) * np.heaviside(E + Eb[i + 2], 1)

            if c2p > 0:

                n = 2
                l = 1

                Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)
                xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)

                # Jnl = self._shell_amplitude(Znl, n, l) * (xnl**4.0 / 4.0 - xnl**5.0 / 5.0)
                Jnl0 = (
                    6.4e1
                    * (1.0 + 5.0 * xnl**2.0)
                    / (1.0 + xnl**2.0) ** 5.0
                    / (1.5e1 * PI * Znl * FINE_STRUCTURE_CONSTANT)
                )
                Jnl1 = (
                    Jnl0
                    * (xnl * (1.0e1 + 1.5e1 * xnl**2.0) / (1.0 + 5.0 * xnl**2.0) / 3.0 - np.arctan(xnl))
                    / (k * BOHR_RADIUS / (Znl * FINE_STRUCTURE_CONSTANT))
                )

                for i in range(c2p):
                    if Eb[i + 4] >= 0:
                        continue
                    # J = J + Jnl * np.heaviside(E, Eb[i + 4 - 1])
                    J = J + (Jnl0 + Jnl1) * np.heaviside(E + Eb[i + 4], 1)

        Sce = J / (SPEED_OF_LIGHT * k)

        # Detailed balanced
        Sce = np.where(w < 0, np.exp(-E / (BOLTZMANN_CONSTANT * self.state.electron_temperature)) * Sce, Sce)

        return Sce

    def truncated_IA(self, ZA, Zb, k, w, Eb):
        Sce = self.schuhmacher_ia(ZA, Zb, k, w, Eb)
        beta = 1 / (BOLTZMANN_CONSTANT * self.state.electron_temperature)
        exp_term = np.exp(beta * (w - Eb)) + 1
        Sce_trunc = Sce * (1 - 1 / exp_term)
        return Sce_trunc


def test():
    import matplotlib.pyplot as plt
    from utils import calculate_q, load_mcss_result, calculate_angle

    Te = 10 * eV_TO_K
    rho = 0.1 * g_per_cm3_TO_kg_per_m3
    charge_state = 0.0
    atomic_number = 1
    atomic_mass = 1.0
    beam_energy = 8.0e3

    angles = np.array([13, 30, 45, 60, 80, 100, 120, 140, 160])
    ks = calculate_q(angle=angles, energy=beam_energy) / BOHR_RADIUS
    omega_array = np.linspace(-40, 300, 500) * eV_TO_J
    EB = (
        np.array(
            [
                -13.7,  # -13.6,  # -13.6,
            ]
        )
        * eV_TO_J
    )
    state = PlasmaState(
        electron_temperature=Te,
        ion_temperature=Te,
        mass_density=rho,
        charge_state=charge_state,
        atomic_mass=atomic_mass,
        atomic_number=atomic_number,
    )
    models = ModelOptions(bf_model="SCHUMACHER")

    colors = ["red", "green", "blue", "orange", "gray", "black", "yellow", "magenta", "purple"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for i in range(len(ks)):
        k = ks[i]
        k_bohr = k * BOHR_RADIUS
        angle = np.round(calculate_angle(q=k * BOHR_RADIUS, energy=beam_energy))  # angles[i]
        print(f"Running for k={k * BOHR_RADIUS} 1/aB and angle={angle}")
        c = colors[i]
        kernel = BoundFreeDSF(state=state, models=models)
        dsf = kernel.get_dsf(ZA=1, Zb=1, Eb=EB, w=omega_array, k=k)
        kernel = BoundFreeDSF(state=state, models=ModelOptions(bf_model="HR_CORRECTION"))
        dsf_hr = kernel.get_dsf(ZA=1, Zb=1, Eb=EB, w=omega_array, k=k)
        kernel = BoundFreeDSF(state=state, models=ModelOptions(bf_model="TRUNCATED_IA"))
        dsf_tr = kernel.get_dsf(ZA=1, Zb=1, Eb=EB, w=omega_array, k=k)

        En, wff, wbf, ff, bf, el = load_mcss_result(
            filename=f"mcss_tests/mcss_outputs_model=IA/mcss_bf_test_angle={angle:0.0f}.csv"
        )
        ax.plot(
            omega_array * J_TO_eV,
            np.array(dsf) / J_TO_eV,
            label=f"IA: k={k_bohr:.2f}",
            c=c,
            ls="solid",
            marker=".",
            markevery=10,
        )
        ax.plot(
            omega_array * J_TO_eV,
            np.array(dsf_hr) / J_TO_eV,
            label=f"HR: k={k_bohr:.2f}",
            c=c,
            ls="dotted",
            marker="<",
            markevery=14,
        )
        ax.plot(
            omega_array * J_TO_eV,
            np.array(dsf_tr) / J_TO_eV,
            label=f"IA tr: k={k_bohr:.2f}",
            c=c,
            ls="-.",
            marker="*",
            markevery=12,
        )
        ax.plot(En, wbf, label=f"MCSS k={k_bohr:.2f}", c=c, ls="dashed")
    for eb in EB:
        ax.axvline(np.abs(eb) * J_TO_eV, c="gray", ls="dotted")
    ax.legend()
    ax.set_xlim(-40, 300)
    plt.show()
    # fig.savefig("bf_test_hydrogen.pdf")


def test_be():
    # Comparison to Fig. 2 Mattern and Seidel, Phys. Plasmas 20 (2013)
    # Be at q~10.2 1/A, phi = 171, Eb = 9890 eV
    import matplotlib.pyplot as plt

    # values from Fortman et al., PRL 108, 175006 (2012)
    Te = 13 * eV_TO_K
    ne = 1.8e24 * per_cm3_TO_per_m3
    Zf = 2.0

    rho = 0.1 * g_per_cm3_TO_kg_per_m3
    charge_state = 0.0
    atomic_number = 1
    atomic_mass = 1.0
    beam_energy = 9890  # eV
    scattering_angle = 171  # degree
    q_approx = 10.2  # 1/Angstrom

    angles = np.array([13, 30, 45, 60, 80, 100, 120, 140, 160])
    omega_array = np.linspace(-40, 300, 500) * eV_TO_J
    EB = (
        np.array(
            [
                -111.5,
            ]
        )
        * eV_TO_J
    )
    state = PlasmaState(
        electron_temperature=Te,
        ion_temperature=Te,
        mass_density=rho,
        charge_state=charge_state,
        atomic_mass=atomic_mass,
        atomic_number=atomic_number,
    )
    models = ModelOptions(bf_model="SCHUMACHER")

    exp_data = np.genfromtxt(f"validation/bf_dsf/mattern2013/Be_Fig1.csv", delimiter=",")
    hm_data = np.genfromtxt(f"validation/bf_dsf/mattern2013/Be_HM.csv", delimiter=",")
    ia_data = np.genfromtxt(f"validation/bf_dsf/mattern2013/Be_IA.csv", delimiter=",")
    pwffa = np.genfromtxt(f"validation/bf_dsf/mattern2013/Be_PWFFA.csv", delimiter=",")
    rsgf = np.genfromtxt(f"validation/bf_dsf/mattern2013/Be_RSGF.csv", delimiter=",")
    rsgf_core = np.genfromtxt(f"validation/bf_dsf/mattern2013/RSGF_Be_3.6_core.csv", delimiter=",")

    # plt.figure()
    # plt.scatter(exp_data[:, 0], exp_data[:, 1], label="Exp", c="k")
    # plt.plot(hm_data[:, 0], hm_data[:, 1], label="HM", c="magenta")
    # plt.plot(ia_data[:, 0], ia_data[:, 1], label="IA", c="navy")
    # plt.plot(pwffa[:, 0], pwffa[:, 1], label="PWFFA", c="crimson")
    # plt.plot(rsgf[:, 0], rsgf[:, 1], label="RSGF", c="orange", marker=".")
    # plt.plot(rsgf_core[:, 0], rsgf_core[:, 1], label="RSGF - core", c="darkgreen", marker=".")
    # plt.legend()
    # plt.ylim(-0.05, 0.35)
    # plt.show()


if __name__ == "__main__":
    # test_be()
    test()

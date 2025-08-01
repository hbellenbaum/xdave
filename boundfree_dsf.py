from ii_ff import PaulingShermanIonicFormFactor, ScreeningConstants
from ipd import get_ipd
from plasma_state import PlasmaState
from unit_conversions import *
from constants import *
from scipy.special import gamma
import numpy as np


class BoundFreeDSF:
    """
    Currently only doing basic Impulse Approximation :)
    """

    def __init__(
        self,
        state: PlasmaState,
    ) -> None:
        self.state = state
        self.ff_model = PaulingShermanIonicFormFactor()
        self.screening_constants = ScreeningConstants

    def get_dsf(self, ZA, Zb, k, w, Eb, bf_model="SCHUMACHER", ipd_model="NONE"):
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
        # TODO(Hannah): decide what convention to stick with: negative or positive IPD/ binding energy
        ipd = 0.0
        if ipd_model != "NONE":
            ipd = get_ipd(state=self.state, model=ipd_model)
        Eb_eff = Eb + ipd # "+ ipd" if Eb < 0 and IPD > 0

        # Load correct bf model
        if bf_model == "SCHUMACHER":
            Sce = self.schuhmacher_ia(ZA, Zb, k, w, Eb_eff)
        else:
            raise NotImplementedError(f"Model {bf_model} not recognised. Try SCHUMACHER :)")
        return Sce

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
            w_freq = w / DIRAC_CONSTANT  # convert the energy range to an actual frequency: E = \hbar \omega

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
                    if Eb[i] >= 0: continue
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
                    if Eb[i + 2] >= 0: continue
                    # J = J + Jnl * np.heaviside(E, Eb[i + 2 - 1])
                    J = J + Jnl * np.heaviside(E + Eb[i + 2], 1)

            if c2p > 0:

                n = 2
                l = 1

                Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)
                xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)

                Jnl = self._shell_amplitude(Znl, n, l) * (xnl**4.0 / 4.0 - xnl**5.0 / 5.0)

                for i in range(c2p):
                    if Eb[i + 4] >= 0: continue
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
                    if Eb[i + 10] >= 0: continue
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
                    if Eb[i + 12] >= 0: continue
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
                    if Eb[i + 18] >= 0: continue
                    # J = J + Jnl * np.heaviside(E, Eb[i + 18 - 1])
                    J = J + Jnl * np.heaviside(E + Eb[i + 18], 1)

            if c3d > 0:

                n = 3
                l = 2

                Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)
                xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)
                Jnl = self._shell_amplitude(Znl, n, l) * (xnl**5.0 / 5.0 - xnl**6.0 / 3.0 + xnl**7.0 / 7.0)

                for i in range(c3d):
                    if Eb[i + 20] >= 0: continue
                    # J = J + Jnl * np.heaviside(E, Eb[i + 20 - 1])
                    J = J + Jnl * np.heaviside(E + Eb[i + 20], 1)

            Sce = J / (SPEED_OF_LIGHT * k)
            # print(f"{Sce}")

              
        # Detailed balanced
        neg_freq_cond = w<0
        Sce[neg_freq_cond] = np.exp(-E[neg_freq_cond] / (BOLTZMANN_CONSTANT * self.state.electron_temperature)) * Sce[neg_freq_cond]
        
        return Sce


def test():
    import matplotlib.pyplot as plt
    from utils import calculate_q, load_mcss_result, calculate_angle

    Te = 10 * eV_TO_K
    rho = 0.1 * g_per_cm3_TO_kg_per_m3
    charge_state = 0.0
    atomic_number = 1
    atomic_mass = 1.0
    beam_energy = 9.0e3

    angles = np.array(
        [
            # 13,
            30,
            # 45,
        ]
    )  # , 30, 45, 60, 80, 100, 120, 140, 160])
    ks = calculate_q(angle=angles, energy=beam_energy) / BOHR_RADIUS
    omega_array = np.linspace(-40, 300, 500) * eV_TO_J
    EB = (
        np.array(
            [
                -13.6,  # -13.6,
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

    colors = ["red", "green", "blue", "orange", "gray", "black", "yellow", "magenta", "purple"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for i in range(len(angles)):
        dsfs = []
        k = ks[i]
        k = 1.1105316e000 / BOHR_RADIUS
        k_bohr = k * BOHR_RADIUS
        # angle = calculate_angle(q=k * BOHR_RADIUS, energy=beam_energy)
        angle = angles[i]
        c = colors[i]
        for omega in omega_array:
            kernel = BoundFreeDSF(state=state)
            dsf = kernel.get_dsf(ZA=1, Zb=1, Eb=EB, w=omega, k=k)

            # print(dsf)
            dsfs.append(dsf)

        wC = PLANCK_CONSTANT * k**2 / (2 * ELECTRON_MASS)  # units of s^{-1} = Hz
        EC = DIRAC_CONSTANT**2 * k**2 / (2 * ELECTRON_MASS * ELEMENTARY_CHARGE)
        # wC = 0.0
        print(f"Compton frequency: {wC * DIRAC_CONSTANT * J_TO_eV}")
        print(f"Compton energy: {EC}")

        idx = np.argwhere(np.isnan(dsfs))
        dsfs_new = np.delete(dsfs, idx)
        omega_new = np.delete(omega_array, idx)
        En, wff, wbf, ff, bf, el = load_mcss_result(
            filename=f"mcss_tests/mcss_outputs_model=IA/mcss_bf_test_angle={angle:0.0f}.csv"
        )
        # ax.axvline(EC, ls="dashed")
        ax.axvline(wC * DIRAC_CONSTANT * J_TO_eV, ls="dotted")
        ax.plot(
            omega_new * J_TO_eV,
            np.array(dsfs_new) / DIRAC_CONSTANT / J_TO_eV,  # / np.max(dsfs_new),  * J_TO_eV
            label=f"k={k_bohr:.2f}",
            c=c,
            ls="solid",
        )  #  / J_TO_eV
        # twinx = ax.twinx()
        ax.plot(En, wbf, label=f"MCSS k={k_bohr:.2f}", c=c, ls="dashed")  #  / np.max(wbf)

    for eb in EB:
        # print(eb)
        ax.axvline(np.abs(eb) * J_TO_eV, c="gray", ls="dotted")
    ax.legend()
    # twinx.legend()
    ax.set_xlim(-40, 300)
    # ax.set_ylim(1.0e-8, 1.5e0)
    plt.show()


if __name__ == "__main__":
    test()

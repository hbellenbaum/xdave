from .ii_ff import PaulingShermanIonicFormFactor, ScreeningConstants
from .plasma_state import PlasmaState
from .unit_conversions import *
from .constants import *
from scipy.special import gamma
import numpy as np


class BoundFreeDSF:
    """
    Class containing the bound-free dynamic structure factor calculations.
    As there is only one option each for the form factor and the screening constant calculations, these are hard-coded.

    Attributes:
        state (PlasmaState): object containing all plasma state variables
        ff_model (PaulingShermanIonicFormFactor): ionic form factor class
        screening_constants (ScreeningConstants): screening constant class
    """

    def __init__(self, state: PlasmaState) -> None:
        self.state = state
        self.ff_model = PaulingShermanIonicFormFactor()
        self.screening_constants = ScreeningConstants
        # self.bf_model = models.bf_model

    def get_dsf(self, ZA, Zb, k, w, Eb, model="SCHUMACHER"):
        """
        Main function to call the dynamic structure factor for a given model.

        Parameters:
            ZA (float): Net charge state
            Zb (float): Number of bound electrons
            k (float): wave number in units of 1/m
            w (array): energy grid in units of 1/J
            Eb (array): binding energies for the different shells in units of J
            bf_model (str): controls the model used for calculating the dsf, default is SCHUMACHER

        Returns:
            array: bound-free dsf in units of 1/J
        """

        # Load correct bf model
        if model == "SCHUMACHER":
            Sce = self.schuhmacher_ia(ZA, Zb, k, w, Eb)
        elif model == "HR_CORRECTION":
            Sce = self.schumacher_ia_correction(ZA, Zb, k, w, Eb)
        elif model == "TRUNCATED_IA":
            Sce = self.truncated_IA(ZA, Zb, k, w, Eb)
        else:
            raise NotImplementedError(f"Model {model} for the bound-free component not recognised. Try SCHUMACHER :)")
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
        Bound-free DSF from Schumacher, Smend and Borchert, J. Phys. B 8, 1428 (1975).

        Parameters:
            ZA (float): Net charge state
            Zb (float): Number of bound electrons
            k (float): wave number in units of 1/m
            w (array): energy grid in units of 1/J
            Eb (array): binding energies for the different shells in units of J
        Returns:
            array: bound-free dsf in units of 1/J
        """

        Sce = 0.0
        J = np.zeros_like(w)

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

            if c1s > 0:

                n = 1
                l = 0

                Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)  # [Znl] = [#]
                xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)  # [xnl] =

                Jnl = self._shell_amplitude(Znl, n, l) * xnl**3 / 3
                # Jnl10 = Jnl
                for i in range(c1s):
                    if Eb[i] > 0:
                        continue
                    J += Jnl * np.heaviside(E + Eb[i], 1)  # np.heaviside(E, Eb[i])
                    # Jnl10 *= np.heaviside(E + Eb[i], 1)

            if c2s > 0:

                n = 2
                l = 0

                Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)
                xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)

                Jnl = self._shell_amplitude(Znl, n, l) * 4.0 * (xnl**3.0 / 3.0 - xnl**4.0 + 4.0 * xnl**5.0 / 5.0)
                # Jnl20 = Jnl
                for i in range(c2s):
                    if Eb[i + 2] > 0:
                        continue
                    J += Jnl * np.heaviside(E + Eb[i + 2], 1)
                    # Jnl20 *= np.heaviside(E + Eb[i + 2], 1)

            if c2p > 0:

                n = 2
                l = 1

                Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)
                xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)

                Jnl = self._shell_amplitude(Znl, n, l) * (xnl**4.0 / 4.0 - xnl**5.0 / 5.0)
                # Jnl21 = Jnl
                for i in range(c2p):
                    if Eb[i + 4] > 0:
                        continue
                    J += Jnl * np.heaviside(E + Eb[i + 4], 1)
                    # Jnl21 *= np.heaviside(E + Eb[i + 4], 1)

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
                    J += Jnl * np.heaviside(E + Eb[i + 10], 1)

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
                    J += Jnl * np.heaviside(E + Eb[i + 12], 1)

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
                    J += Jnl * np.heaviside(E + Eb[i + 18], 1)

            if c3d > 0:

                n = 3
                l = 2

                Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)
                xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)
                Jnl = self._shell_amplitude(Znl, n, l) * (xnl**5.0 / 5.0 - xnl**6.0 / 3.0 + xnl**7.0 / 7.0)

                for i in range(c3d):
                    if Eb[i + 20] >= 0:
                        continue
                    J += Jnl * np.heaviside(E + Eb[i + 20], 1)

            # Sce = (c1s * Jnl10 + c2s * Jnl20 + c2p * Jnl21) / (SPEED_OF_LIGHT * k)
            Sce = J / (SPEED_OF_LIGHT * k)

            # Detailed balanced
            Sce = np.where(w < 0, np.exp(-E / (BOLTZMANN_CONSTANT * self.state.electron_temperature)) * Sce, Sce)

        return Sce

    def schumacher_ia_correction(self, ZA, Zb, k, w, Eb):
        """
        First order correction to the bound-free Schumacher Impulse Approximation based on
        Holm and Ribberfors, Phys. Rev. A 40 (1989).

        Parameters:
            ZA (float): Net charge state
            Zb (float): Number of bound electrons
            k (float): wave number in units of 1/m
            w (array): energy grid in units of 1/J
            Eb (array): binding energies for the different shells in units of J
        Returns:
            array: bound-free dsf in units of 1/J
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
        """
        Correction applied to the Impulse Approximation based on Mattern and Seidler, Phys. Plasmas 20 (2013)

        Parameters:
            ZA (float): Net charge state
            Zb (float): Number of bound electrons
            k (float): wave number in units of 1/m
            w (array): energy grid in units of 1/J
            Eb (array): binding energies for the different shells in units of J
        Returns:
            array: bound-free dsf in units of 1/J
        """
        Sce = self.schuhmacher_ia(ZA, Zb, k, w, Eb)
        beta = 1 / (BOLTZMANN_CONSTANT * self.state.electron_temperature)
        exp_term = np.exp(beta * (w - Eb[0])) + 1
        Sce_trunc = Sce * (1 - 1 / exp_term)
        return Sce_trunc

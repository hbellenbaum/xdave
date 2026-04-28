from .ii_ff import PaulingShermanIonicFormFactor, ScreeningConstants
from .freefree_dsf import FreeFreeDSF
from .plasma_state import PlasmaState
from .unit_conversions import *
from .constants import *
from scipy.special import gamma
import numpy as np
import warnings

import math


class BoundFreeDSF:
    """
    Class containing the bound-free dynamic structure factor calculations.
    As there is only one option each for the form factor and the screening constant calculations, these are hard-coded.

    Parameters:
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
            warnings.warn(f"Bound-free model {model} not recognized. Overwriting using the default SCHUMACHER.")
            Sce = self.schuhmacher_ia(ZA, Zb, k, w, Eb)
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
        J = np.zeros(w.shape, dtype=np.float64)

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

            if (c1s > 0) and (Eb[0] <= 0):

                n = 1
                l = 0

                Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)  # [Znl] = [#]
                xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)  # [xnl] =

                Jnl = self._shell_amplitude(Znl, n, l) * xnl**3 / 3

                J += c1s * Jnl * np.heaviside(E + Eb[0], 1)

            if (c2s > 0) and (Eb[1] <= 0):

                n = 2
                l = 0

                Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)
                xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)

                Jnl = self._shell_amplitude(Znl, n, l) * 4.0 * (xnl**3.0 / 3.0 - xnl**4.0 + 4.0 * xnl**5.0 / 5.0)

                J += c2s * Jnl * np.heaviside(E + Eb[1], 1)

            if (c2p > 0) and ((Eb[2] <= 0) or (Eb[3] <= 0)):

                n = 2
                l = 1

                Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)
                xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)

                Jnl = self._shell_amplitude(Znl, n, l) * (xnl**4.0 / 4.0 - xnl**5.0 / 5.0)

                # Divide c2p between the states
                c_12 = 0
                c_32 = 0
                while c2p > 0:
                    if (c_12 < 2) and (Eb[2] <= 0):
                        c_12 += 1
                    elif (c_32 < 4) and (Eb[3] <= 0):
                        c_32 += 1
                    c2p -= 1

                J += c_12 * Jnl * np.heaviside(E + Eb[2], 1)
                J += c_32 * Jnl * np.heaviside(E + Eb[3], 1)

            if (c3s > 0) and (Eb[4] <= 0):

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

                J += c3s * Jnl * np.heaviside(E + Eb[4], 1)

            if (c3p > 0) and ((Eb[5] <= 0) or (Eb[6] <= 0)):

                n = 3
                l = 1

                Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)
                xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)
                Jnl = (
                    self._shell_amplitude(Znl, n, l)
                    * 1.6e1
                    * (xnl**4.0 / 4.0 - xnl**5.0 + 4.0 * xnl**6.0 / 3.0 - 4.0 * xnl**7.0 / 7.0)
                )

                # Divide c3p between the states
                c_12 = 0
                c_32 = 0
                while c3p > 0:
                    if (c_12 < 2) and (Eb[5] <= 0):
                        c_12 += 1
                    elif (c_32 < 4) and (Eb[6] <= 0):
                        c_32 += 1
                    c3p -= 1

                J += c_12 * Jnl * np.heaviside(E + Eb[5], 1)
                J += c_32 * Jnl * np.heaviside(E + Eb[6], 1)

            if (c4s > 0) and (Eb[9] <= 0):
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

                J += c4s * Jnl * np.heaviside(E + Eb[9], 1)

            if (c3d > 0) and ((Eb[7] <= 0) or (Eb[8] <= 0)):

                n = 3
                l = 2

                Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)
                xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)
                Jnl = self._shell_amplitude(Znl, n, l) * (xnl**5.0 / 5.0 - xnl**6.0 / 3.0 + xnl**7.0 / 7.0)

                # Divide c3d between the states
                c_32 = 0
                c_52 = 0
                while c3d > 0:
                    if (c_32 < 4) and (Eb[7] <= 0):
                        c_32 += 1
                    elif (c_52 < 6) and (Eb[8] <= 0):
                        c_52 += 1
                    c3d -= 1

                J += c_32 * Jnl * np.heaviside(E + Eb[7], 1)
                J += c_52 * Jnl * np.heaviside(E + Eb[8], 1)

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
        J = np.zeros(w.shape, dtype=np.float64)

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

            if (c1s > 0) and (Eb[0] <= 0):
                n = 1
                l = 0

                Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)  # [Znl] = [#]
                xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)  # [xnl] =

                Jnl0 = self._shell_amplitude(Znl, n, l) * xnl**3 / 3
                Jnl1 = Jnl0 * (1.5 * xnl - 2.0 * np.arctan(xnl)) / (k * BOHR_RADIUS / (Znl * FINE_STRUCTURE_CONSTANT))

                J += c1s * (Jnl0 + Jnl1) * np.heaviside(E + Eb[0], 1)

            if (c2s > 0) and (Eb[1] <= 0):

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

                J += c2s * (Jnl0 + Jnl1) * np.heaviside(E + Eb[1], 1)

            if (c2p > 0) and ((Eb[2] <= 0) or (Eb[3] <= 0)):

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

                # Divide c2p between the states
                c_12 = 0
                c_32 = 0
                while c2p > 0:
                    if (c_12 < 2) and (Eb[2] <= 0):
                        c_12 += 1
                    elif (c_32 < 4) and (Eb[3] <= 0):
                        c_32 += 1
                    c2p -= 1

                J += c_12 * (Jnl0 + Jnl1) * np.heaviside(E + Eb[2], 1)
                J += c_32 * (Jnl0 + Jnl1) * np.heaviside(E + Eb[3], 1)

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

    def _fletcher_effective_charge_state(self, Zf, xm):  # Effective nuclear ionization
        m = np.polyfit(np.arange(len(xm)) - 1, xm, 1)
        S = m[1] + (m[0] * np.arange(0, len(xm), 0.1))
        Znl = S[int(np.round((Zf * 10)) - 1)]

        return Znl

    def _bf_normalization(self, dsf, Z, w, k):
        Z = Z + 1.0e-6
        I_in = np.trapezoid(w, dsf * w)
        # [Cx]
        Cx = (I_in * (2 * ELECTRON_MASS)) / (Z * DIRAC_CONSTANT * (k * k))  # / ELEMENTARY_CHARGE
        return dsf / Cx

    def fletcher_modified_IA(self, ZA, Zb, Zf, k, w, Eb, Zl, Zk):
        """
        Modified impulse approximation specifically designed for carbon.
        The valence electrons are treated using RPA.
        K and L-shell contributions are considered separately.
        """
        # w *= J_TO_eV
        assert ZA == 6, f"You are using a BF model calibrated to carbon."

        Zf = 0.1

        # Empirical shift to match Compton for normalization (Numerical differences from the energy range En?)
        # convert energy to frequency domain
        E = -7 * eV_TO_J  # [J]

        # Compton frequency
        wC = DIRAC_CONSTANT * k**2 / (2 * ELECTRON_MASS)  # units of s^{-1} = Hz

        # TODO(HB): check units in the k-vector

        Z_10 = self._fletcher_effective_charge_state(Zf, [5.7, 5.7, 5.7, 5.7, 5.7, 6])  # [ ]
        n = 1
        l = 0
        Y10 = (
            1
            + (
                (n / (Z_10 * FINE_STRUCTURE_CONSTANT * SPEED_OF_LIGHT * k))
                * (w / DIRAC_CONSTANT - E / DIRAC_CONSTANT - ((DIRAC_CONSTANT * k**2) / (2 * ELECTRON_MASS)))
            )
            ** 2
        )  # [ ]
        A_10 = (
            ((2 ** ((4 * l) + 3)) / np.pi)
            * ((math.factorial(n - l - 1)) / (math.factorial(n + l)))
            * ((n**2 * (math.factorial(l)) ** 2) / (Z_10 * FINE_STRUCTURE_CONSTANT))
        )  # [ ]
        phi_10 = 1 / (3 * (Y10**3))
        # phi_10 = 1 / (3 * (Ynl**2)) # Initial settings per Schumacher, Bloch above leads to a sharper peak and faster decay
        phi_10 = A_10 * phi_10  # [ ]

        Z_20 = self._fletcher_effective_charge_state(Zf, [3.25, 3.6, 3.95, 4.3, 4.3, 5.15])  # [ ]
        n = 2
        l = 0
        Y20 = (
            1
            + (
                (n / (Z_20 * FINE_STRUCTURE_CONSTANT * SPEED_OF_LIGHT * k))
                * (w / DIRAC_CONSTANT - E / DIRAC_CONSTANT - ((DIRAC_CONSTANT * k**2) / (2 * ELECTRON_MASS)))
            )
            ** 2
        )  # [ ]
        A_20 = (
            ((2 ** ((4 * l) + 3)) / np.pi)
            * ((math.factorial(n - l - 1)) / (math.factorial(n + l)))
            * ((n**2 * (math.factorial(l)) ** 2) / (Z_20 * FINE_STRUCTURE_CONSTANT))
        )
        phi_20 = 4 * ((1 / (3 * (Y20**3))) - (1 / (Y20**4)) + (4 / (5 * (Y20**5))))  # [ ]
        phi_20 = A_20 * phi_20  # [ ]

        Z_21 = self._fletcher_effective_charge_state(Zf, [3.25, 3.6, 3.6, 3.95, 4.3, 5.15])  # [ ]
        n = 2
        l = 0
        Y21 = (
            1
            + (
                (n / (Z_21 * FINE_STRUCTURE_CONSTANT * SPEED_OF_LIGHT * k))
                * (w / DIRAC_CONSTANT - E / DIRAC_CONSTANT - ((DIRAC_CONSTANT * k**2) / (2 * ELECTRON_MASS)))
            )
            ** 2
        )  # [ ]
        A_21 = (
            ((2 ** ((4 * l) + 3)) / np.pi)
            * ((math.factorial(n - l - 1)) / (math.factorial(n + l)))
            * ((n**2 * (math.factorial(l)) ** 2) / (Z_21 * FINE_STRUCTURE_CONSTANT))
        )  # [ ]
        phi_21 = 4 * ((1 / (3 * (Y21**3))) - (1 / (Y21**4)) + (4 / (5 * (Y21**5))))  # [ ]
        phi_21 = A_21 * phi_21  # [ ]

        # TODO(HB): check the negative sign here
        # TODO(HB): check the units here, this is not dimensionless like it should be...
        continuum_edge = 1 / (1 + np.exp(-1 * w / DIRAC_CONSTANT))
        k_edgeC = 1 / (1 + np.exp(-1 * (w / DIRAC_CONSTANT - Eb / DIRAC_CONSTANT)))
        # k_edgeC = 1 / (1 + np.exp(-1 * (w / DIRAC_CONSTANT - 284.2 * eV_TO_J / DIRAC_CONSTANT)))
        # continuum_edge = 1 / (1 + np.exp(-1 * wC))
        # k_edgeC = 1 / (1 + np.exp(-1 * (w / DIRAC_CONSTANT - 284.2 * eV_TO_J / DIRAC_CONSTANT)))
        # Carbon K-edge = 284.2

        phi_10 *= k_edgeC
        phi_20 *= continuum_edge
        phi_21 *= continuum_edge

        phi_L = -1 * self._bf_normalization(dsf=phi_21 + phi_20, Z=4 - Zl, w=w, k=k)
        carbon_Kshell = -1 * self._bf_normalization(dsf=phi_10, Z=2 - Zk, w=w, k=k)

        C = 1.827e3  # ????

        temp_state = PlasmaState(
            electron_temperature=4 * eV_TO_K,
            ion_temperature=4 * eV_TO_K,
            mass_density=1 * g_per_cm3_TO_kg_per_m3,
            charge_state=4,
            atomic_number=1,
            atomic_mass=1,
            binding_energies=None,
        )
        valence_carbon_solid = FreeFreeDSF(state=temp_state).get_dsf(k=k, w=w, lfc=0.0, model="DANDREA_FIT")  # [1/J]
        # This changes nothing
        # valence_carbon = valence_carbon_solid / J_TO_eV
        valence_carbon = (np.sum(phi_L) / np.sum(valence_carbon_solid)) * valence_carbon_solid

        # return (
        #     C * 0.9964219451371571 * carbon_Kshell * DIRAC_CONSTANT,
        #     C * 1.0547927578124998 * valence_carbon * DIRAC_CONSTANT,
        # )
        C1 = 2.51624627
        return carbon_Kshell * DIRAC_CONSTANT / C1, valence_carbon * DIRAC_CONSTANT / C1

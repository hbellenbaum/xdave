
from ii_ff import PaulingShermanIonicFormFactor
from plasma_state import PlasmaState
from unit_conversions import J_TO_eV
from constants import *
from scipy.special import gamma
import numpy as np


class BoundFreeDSF:
    """
    Currently only doing basic Impulse Approximation :)
    """

    def __init__(self, state: PlasmaState, binding_energy) -> None:
        self.state = state
        self.binding_energy = binding_energy

    def _shell_amplitude(self, Znl, n, l):
        Anl = (
            2.0 ** (4.0 * l + 3.0)
            * gamma(n - l)
            * (n * gamma(l + 1.0)) ** 2.0
            / (np.pi * FINE_STRUCTURE_CONSTANT * Znl * gamma(n + l + 1.0))
        )
        return Anl


    def schuhmacher_ia(self, ZA, Zb):
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

        w = self.state.frequency
        k = self.state.wave_number
        EB = self.binding_energy

        Sce = 0.
        if Zb <= 0:
            return Sce
        
        c1s, c2s, c2p, c3s, c3p , c4s, c3d = self._calculate_shell_coefficients(Zb)
        print(c1s, c2s, c2p, c3s, c3p , c4s, c3d)
        E = PLANCK_CONSTANT * w * J_TO_eV
        ## TODO(Hannah): figure out where to store things like beam energy etc.
        wC = self._compton_frequency(k, ELECTRON_MASS)
        q = (w - wC) / (SPEED_OF_LIGHT * k)
        J = 0.0

        if c1s > 0.0:
            n = 1
            l = 0
            Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)
            xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT))**2.0)
            Jnl = self._shell_amplitude(Znl, n, l) * xnl ** 3.0 / 3.0
            # for i in range(len(c1s)):
            #     J = J + Jnl * np.heaviside(E, EB[i])
            J = J + Jnl * np.heaviside(E, EB)

        if c2s > 0:
            n = 2
            l = 0
            Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)
            xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT))**2.0)
            Jnl = self.shell__shell_amplitudemp(Znl, n, l) * 4.0 * (xnl ** 3.0 / 3.0 - xnl ** 4.0 + 4.0 * xnl ** 5.0 / 5.0)
            for i in range(c2s):
                J =  J + Jnl * np.heaviside(E, EB[i + 2 - 1])

        if c2p > 0:
            n = 2
            l = 1
            Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)
            xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)
            Jnl = self._shell_amplitude(Znl, n, l) * (xnl ** 4.0 / 4.0 - xnl ** 5.0 / 5.0)
            for i in range(c2p):
                J = J + Jnl * np.heaviside(E, EB[i + 4 - 1] )

        if c3s > 0:
            n = 3
            l = 0
            Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)
            xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)
            Jnl = self._shell_amplitude(Znl, n, l) * (3.0 * xnl ** 3.0 - 2.4e1 * xnl ** 4.0 + 3.52e2 * xnl ** 5.0 / 5.0 - 2.56e2 * xnl ** 6.0 / 3.0 + 2.56e2 * xnl ** 7.0 / 7.0)
            for i in range(c3s):
                J = J + Jnl * np.heaviside(E, EB[i + 10 - 1])

        if c3p > 0:
            n = 3
            l = 1
            Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)
            xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)
            Jnl = self._shell_amplitude(Znl, n, l) * 1.6e1 * (xnl**4.0 / 4.0 - xnl**5.0 + 4.0 * xnl**6.0 / 3.0 - 4.0 * xnl**7.0 / 7.0)
            for i in range(c3p):
                J = J + Jnl * np.heaviside(E, EB[i + 12 - 1])

        if c4s > 0:
            n = 4
            l = 0
            Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)
            xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)
            Jnl = self._shell_amplitude(Znl, n, l) *  1.6e1 * (xnl**3.0 / 3.0 - 5.0 * xnl**4.0 + 1.48e2 * xnl**5.0 / 5.0 - 2.56e2 * xnl**6.0 / 3.0 + 1.28e2 * xnl**7.0 - 9.6e1 * xnl**8.0 + 2.56e2 * xnl**9.0 / 9.0)
            for i in range(c4s):
                J = J + Jnl * np.heaviside(E, EB[i + 18 - 1])

        if c3d > 0:
            n = 3
            l = 2
            Znl = self.ff_model.calculate_effective_charge_state(ZA, Zb, n, l)
            xnl = 1.0 / (1.0 + (n * q / (Znl * FINE_STRUCTURE_CONSTANT)) ** 2.0)
            Jnl = self._shell_amplitude(Znl, n, l) * (xnl**5.0 / 5.0 - xnl**6.0 / 3.0 + xnl**7.0 / 7.0)
            for i in range(c3p):
                J = J + Jnl * np.heaviside(E, EB[i + 20 - 1])

        Sce = J / (SPEED_OF_LIGHT * k)

        return Sce

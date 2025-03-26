from constants import *
from unit_conversions import *
from ii_ff import PaulingShermanIonicFormFactor
from ii_ssf import hnc_ab_structure
from plasma_state import PlasmaState

import numpy as np


class RayleighWeight:

    def __init__(self, state: PlasmaState):
        self.state = state

    def screening_cloud():
        return

    def form_factor(self, Z, Zb, k):
        ZA = int(Z)
        Zb = int(Zb) # self.state.AN - Z
        # ZA = None
        return PaulingShermanIonicFormFactor().calculate_form_factor(self, Z, ZA, Zb, k)

    def calculate_rayleigh_weight(self):
        S_ii = hnc_ab_structure(state=self.state)
        ff = self.form_factor(self.state.charge_state, self.state.scattering_number)
        qi = self.screening_cloud()
        return

from ii_ff import PaulingShermanIonicFormFactor
from screening_cloud import ScreeningCloud
from static_sf import OCPStaticStructureFactor, MCPStaticStructureFactor

import numpy as np


class OCPRayleighWeight:
    def __init__(self, overlord_state, state, ion_core_radius):
        self.overlord_state = overlord_state
        self.state = state
        self.ion_core_radius = ion_core_radius

    def get_rayleigh_weight(
        self,
        k,
        sf_model,
        ii_potential,
        ee_potential,
        ei_potential,
        bridge_function,
        lfc=0.0,
        screening_model="NONE",
        return_full=False,
    ):
        sf = OCPStaticStructureFactor(state=self.state, ion_core_radius=self.ion_core_radius)

        Siik = sf.get_ii_static_structure_factor(
            k=k, sf_model=sf_model, pseudo_potential=ii_potential, bridge_function=bridge_function, return_full=False
        )

        qs = ScreeningCloud(state=self.state, overlord_state=self.overlord_state).get_screening_cloud(
            k=k,
            ion_core_radius=self.ion_core_radius,
            lfc=lfc,
            screening_model=screening_model,
            ee_potential=ee_potential,
            ei_potential=ei_potential,
        )
        fs = PaulingShermanIonicFormFactor().calculate_form_factor(Z=self.state.atomic_number, Z_b=self.state.Zb, k=k)

        rayleigh_weight = (fs + qs) * Siik

        if return_full:
            return k, Siik, rayleigh_weight, qs, fs
        else:
            return rayleigh_weight


class MCPRayleighWeight:

    def __init__(self, overlord_state, states, ion_core_radius) -> None:
        self.overlord_state = overlord_state
        self.states = states
        self.nspecies = len(states)
        self.ion_core_radius = ion_core_radius

    def get_rayleigh_weight(
        self, k, lfc, sf_model, ii_potential, ee_potential, ei_potential, screening_model, return_full=False
    ):

        nspecies = self.nspecies
        states = self.states

        if np.shape(k) == ():
            qs = np.zeros((nspecies, 1))
            fs = np.zeros((nspecies, 1))
        else:
            qs = np.zeros((nspecies, len(k)))
            fs = np.zeros((nspecies, len(k)))

        for i in (0, nspecies - 1):
            qs[i, :] = ScreeningCloud(state=self.states[i], overlord_state=self.overlord_state).get_screening_cloud(
                k=k,
                ion_core_radius=self.ion_core_radius,
                lfc=lfc,
                screening_model=screening_model,
                ee_potential=ee_potential,
                ei_potential=ei_potential,
            )
            fs[i, :] = PaulingShermanIonicFormFactor().calculate_form_factor(
                Z=states[i].atomic_number, Z_b=states[i].Zb, k=k
            )

        sf = MCPStaticStructureFactor(
            overlord_state=self.overlord_state,
            states=states,
            mix_fraction=0.9,
            delta=1.0e-8,
        )
        Sab = sf.get_ab_static_structure_factor(
            k=k,
            sf_model=sf_model,
            pseudo_potential=ii_potential,
            return_full=False,
        )

        xs = sf.xs

        if np.shape(k) == ():
            rayleigh_weight = 0
            # TODO(Hannah): there has to be a better way of doing this...
            for n1 in range(nspecies):
                for n2 in range(nspecies):
                    rayleigh_weight += np.sqrt(xs[n1] * xs[n2]) * (fs[n1] + qs[n1]) * (fs[n2] + qs[n2]) * Sab[n1, n2]

            if return_full:
                return k, Sab, rayleigh_weight, qs, fs
        else:
            rayleigh_weight = np.zeros_like(k)
            for n1 in range(nspecies):
                for n2 in range(nspecies):
                    rayleigh_weight += (
                        np.sqrt(xs[n1] * xs[n2]) * (fs[n1] + qs[n1]) * (fs[n2] + qs[n2]) * Sab[n1, n2, :]
                    )

            if return_full:
                return k, Sab, rayleigh_weight, qs, fs
        return rayleigh_weight

from .ii_ff import PaulingShermanIonicFormFactor
from .screening_cloud import ScreeningCloud
from .static_sf import OCPStaticStructureFactor, MCPStaticStructureFactor

import numpy as np
import warnings


class OCPRayleighWeight:
    """
    Class holding calculations for the Rayleigh weight of a one-component plasma.

    Attributes:
        overlord_state (PlasmaState):
        state (PlasmaState):
    """

    def __init__(self, overlord_state, state, verbose: bool = False):
        self.overlord_state = overlord_state
        self.state = state
        self.verbose = verbose

    def get_rayleigh_weight(
        self,
        k,
        sf_model,
        ii_potential,
        ee_potential,
        ei_potential,
        bridge_function,
        hnc_max_iterations,
        hnc_mix_fraction,
        hnc_delta,
        lfc=0.0,
        screening_model="FINITE_WAVELENGTH",
        return_full=False,
    ):
        """
        Main run function to obtain the rayleigh from the ionic form factors,
        screening cloud and the static structure factors.

        Parameters:
            k (float/array): wave number in units of 1/m
            sf_model (str): controls the model used for the static structure factor model
            ii_potential (str): controls the model used for the ion-ion potential in the HNC solver
            ee_potential (str): controls the model used for the electron-electron potential in the screening cloud
            ei_potential (str): controls the model used for the electron-ion potential in the screening cloud
            bridge_function (str): controls the model used for the bridge function
            hnc_max_iterations (int): maximum number of iterations in the HNC solver
            hnc_mix_fraction (float): mix fraction in the HNC solver
            hnc_delta (float): tolerance in the HNC solver
            lfc (float/array): local field correction, non-dimensional
            screening_model (str): controls the model used for the screening cloud
            return_full (bool): if True, returns all components of the rayleigh weight
        Returns:
            float/array: k-grid in units of 1/m, will only be returned if return_full=True
            float/array: non-dimensional static structure factors, will only be returned if return_full=True
            float/array: non-dimensional rayleigh weight, always returned
            float/array: non-dimensional screening cloud, will only be returned if return_full=True
            float/array: non-dimensional ionic form factor, will only be returned if return_full=True
        """
        sf = OCPStaticStructureFactor(
            state=self.state,
            max_iterations=hnc_max_iterations,
            mix_fraction=hnc_mix_fraction,
            delta=hnc_delta,
            verbose=self.verbose,
        )

        if self.state.free_electron_number_density > 0:
            Siik = sf.get_ii_static_structure_factor(
                k=k,
                sf_model=sf_model,
                pseudo_potential=ii_potential,
                bridge_function=bridge_function,
                return_full=False,
            )
        else:
            Siik = np.full_like(k, 1.0)

        if self.state.free_electron_number_density > 0:
            qs = ScreeningCloud(state=self.state, overlord_state=self.overlord_state).get_screening_cloud(
                k=k,
                lfc=lfc,
                screening_model=screening_model,
                ee_potential=ee_potential,
                ei_potential=ei_potential,
            )
        else:
            qs = np.zeros_like(k)
        fs = PaulingShermanIonicFormFactor().calculate_form_factor(Z=self.state.atomic_number, Z_b=self.state.Zb, k=k)

        rayleigh_weight = (fs + qs) * Siik

        if return_full:
            return k, Siik, rayleigh_weight, qs, fs
        else:
            return rayleigh_weight


class MCPRayleighWeight:
    """
    Class holding calculations for the Rayleigh weight of a multi-component plasma.

    Attributes:
        overlord_state (PlasmaState):
        state (PlasmaState):
        nspecies (int): number of species
    """

    def __init__(self, overlord_state, states, verbose: bool = False) -> None:
        self.overlord_state = overlord_state
        self.states = states
        self.nspecies = len(states)
        self.verbose = verbose

    def get_rayleigh_weight(
        self,
        k,
        lfc,
        sf_model,
        ii_potential,
        ee_potential,
        ei_potential,
        screening_model,
        hnc_max_iterations,
        hnc_mix_fraction,
        hnc_delta,
        return_full=False,
    ):
        """
        Main run function to obtain the rayleigh from the ionic form factors,
        screening cloud and the static structure factors.

        Parameters:
            k (float/array): wave number in units of 1/m
            sf_model (str): controls the model used for the static structure factor model
            ii_potential (str): controls the model used for the ion-ion potential in the HNC solver
            ee_potential (str): controls the model used for the electron-electron potential in the screening cloud
            ei_potential (str): controls the model used for the electron-ion potential in the screening cloud
            hnc_max_iterations (int): maximum number of iterations in the HNC solver
            hnc_mix_fraction (float): mix fraction in the HNC solver
            hnc_delta (float): tolerance in the HNC solver
            lfc (float/array): local field correction, non-dimensional
            screening_model (str): controls the model used for the screening cloud
            return_full (bool): if True, returns all outputs from the HNC calculations

        Returns:
            float/array: k-grid in units of 1/m, will only be returned if return_full=True
            float/array: non-dimensional static structure factors, will only be returned if return_full=True
            float/array: non-dimensional rayleigh weight, always returned
            float/array: non-dimensional screening cloud, will only be returned if return_full=True
            float/array: non-dimensional ionic form factor, will only be returned if return_full=True
        """

        nspecies = self.nspecies
        states = self.states

        if np.shape(k) == ():
            qs = np.zeros((nspecies, 1))
            fs = np.zeros((nspecies, 1))
        else:
            qs = np.zeros((nspecies, len(k)))
            fs = np.zeros((nspecies, len(k)))

        for i in range(0, nspecies):
            if self.states[i].charge_state > 0:
                qs[i, :] = ScreeningCloud(
                    state=self.states[i], overlord_state=self.overlord_state
                ).get_screening_cloud(
                    k=k,
                    lfc=lfc,
                    screening_model=screening_model,
                    ee_potential=ee_potential,
                    ei_potential=ei_potential,
                )
            else:
                # print("Hitting Z=0 in screening cloud.")
                qs[i, :] = np.zeros_like(k)
            fs[i, :] = PaulingShermanIonicFormFactor().calculate_form_factor(
                Z=states[i].atomic_number, Z_b=states[i].Zb, k=k
            )

        sf = MCPStaticStructureFactor(
            overlord_state=self.overlord_state,
            states=states,
            mix_fraction=hnc_mix_fraction,
            delta=hnc_delta,
            max_iterations=hnc_max_iterations,
            verbose=self.verbose,
        )
        if self.overlord_state.charge_state > 0:
            Sab = sf.get_ab_static_structure_factor(
                k=k,
                sf_model=sf_model,
                pseudo_potential=ii_potential,
                return_full=False,
            )
            if not sf.success:
                warnings.warn(
                    f"The static structure factor solver has failed. This will only occur for HNC, so check your inputs and try increasing your mix fraction."
                )
        else:
            # print("Hitting Z=0 in Sab.")
            nspecies = len(self.states)
            try:
                Sab = np.zeros((nspecies, nspecies, len(k)))
            except TypeError:
                # filter out cases where k is a single value
                Sab = np.zeros((nspecies, nspecies, 1))
            for n1 in range(0, nspecies):
                for n2 in range(0, nspecies):
                    if n1 == n2:
                        Sab[n1, n2, :] = 1

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

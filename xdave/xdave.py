from .plasma_state import PlasmaState, get_fractions_from_Z
from .models import ModelOptions
from .freefree_dsf import FreeFreeDSF
from .boundfree_dsf import BoundFreeDSF
from .rayleigh_weight import OCPRayleighWeight, MCPRayleighWeight
from .lfc import LFC
from .ipd import get_ipd

from .utils import (
    calculate_q,
    laplace,
    get_atomic_data_for_all_elements,
    get_binding_energies_from_element,
    calculate_angle,
)
from .unit_conversions import *
from .constants import *

from datetime import datetime
import numpy as np
import warnings

from scipy import interpolate

import matplotlib.pyplot as plt

import json
import os

warnings.filterwarnings("ignore", category=FutureWarning)


# @profile
class xDave:
    """
    A class containing all functionality to create dynamic structure factors from the Chihara decomposition.
    corresponding ITCFs and convolve the output with a source function to create a spectrum.

    Attributes:
        mass_density (float): in g/cc
        electron_temperature (float): in eV
        ion_temperature (float): in eV, currently, these can be set separately, but the models assume equilibrium
        elements (array): array of the element of each species by symbol
        partial_densities (array): array of the partial densities for each species
        charge_states (array): array of charge state for each species
        models (ModelOptions): instance of the class model options to specify the models used for each
        enforce_fsum (bool): flag to specify whether the bound-free output should be forced to obey the f-sum, default is False
        hnc_mix_fraction (float): mix fraction for the HNC solver, needs to be between 0 and 1. For higher values, less of the new solution is taken into account which makes the solver more stable for strong coupling but also slower.
        hnc_delta (float): convergence criterium on the HNC solver.
        hnc_max_iterations (int): maximum number of iterations in the HNC solver.
        user_defined_inputs (dict): list containing values set by the user for LFC, IPD, the ion core radius and parameters to define the pseudo-potentials used in the HNC solver and the screening cloud; this is optional
        verbose (bool): Option to print run statements throughout or run silently.
        save_to_json (bool): option to save outputs to json file.
        output_file_name (str): directory and file to save the json output to, has to be set if save_to_json = True.
    """

    def __init__(
        self,
        mass_density: float,
        electron_temperature: float,
        ion_temperature: float,
        elements: np.array,
        partial_densities: np.array,
        charge_states: np.array,
        models: ModelOptions,
        enforce_fsum: bool = False,
        hnc_mix_fraction: float = 0.9,
        hnc_delta: float = 1.0e-6,
        hnc_max_iterations: int = 1000,
        user_defined_inputs: dict = None,
        verbose: bool = False,
        save_to_json: bool = False,
        output_file_name: str = None,
    ):
        assert np.sum(partial_densities) == 1.0, f"Fractional densities do not add up 1."
        assert electron_temperature > 0.0, f"Ensure your temperature is positive."
        assert mass_density > 0.0, f"Ensure your mass density is positive."
        assert len(charge_states) == len(
            partial_densities
        ), f"The number of species and corresponding charge states do not match."
        self.number_of_states = len(partial_densities)
        self.mass_density = mass_density * g_per_cm3_TO_kg_per_m3
        self.electron_temperature = electron_temperature * eV_TO_K
        self.ion_temperature = ion_temperature * eV_TO_K
        if not np.isclose(electron_temperature, ion_temperature, rtol=1.0e-4):
            warnings.warn(f"You have set non-equilibrium conditions. Make sure the models called allow this.")
        self.partial_densities = partial_densities
        self.elements = elements
        self.charge_states = charge_states
        self.models = models

        self.tau_array = np.linspace(0, 1 / (electron_temperature), 1000)

        if user_defined_inputs is not None:
            keys = user_defined_inputs.keys()
            self.ipd_eV = user_defined_inputs["ipd"] if "ipd" in keys else None
            self.user_defined_lfc = user_defined_inputs["lfc"] if "lfc" in keys else None
            self.ion_core_radii = (
                np.array(user_defined_inputs["ion_core_radii"]) * BOHR_RADIUS
                if "ion_core_radii" in keys
                else np.full(self.number_of_states, None)
            )
            self.csd_parameters = (
                user_defined_inputs["csd_parameters"]
                if "csd_parameters" in keys
                else np.full(self.number_of_states, None)
            )
            self.csd_core_charges = (
                user_defined_inputs["csd_core_charges"]
                if "csd_core_charges" in keys
                else np.full(self.number_of_states, None)
            )
            self.sec_core_power = user_defined_inputs["sec_core_power"] if "sec_core_power" in keys else None
            self.srr_sigma_parameter = (
                user_defined_inputs["srr_sigma_parameter"] if "srr_sigma_parameter" in keys else None
            )

            # Few options to check here for the Crowley constant
            crowley_force_constant = (
                user_defined_inputs["crowley_force_constant"] if "crowley_force_constant" in keys else None
            )
            if crowley_force_constant is None:
                self.crowley_force_constant = 0.9
            else:
                if isinstance(crowley_force_constant, (float, int)):
                    self.crowley_force_constant = crowley_force_constant
                else:
                    if crowley_force_constant.upper() in ["FLUID", "ION_SPHERE"]:
                        self.crowley_force_constant = 0.9
                    elif crowley_force_constant.upper() in ["FCC", "HCP"]:
                        self.crowley_force_constant = 0.99025
                    elif crowley_force_constant.upper() == "BCC":
                        self.crowley_force_constant = 1.01875
                    elif crowley_force_constant.upper() == "SC":
                        self.crowley_force_constant = 1.09189
                    else:
                        self.crowley_force_constant = 0.9
                        warnings.warn(
                            f"crowley_force_structure = {crowley_force_constant} not known!\n"
                            + f"Should be a number or 'FLUID', 'ION_SPHERE', 'FCC', 'HCP', 'BCC', or 'SC' "
                            + f"Treating as ion sphere (constant = 0.9)"
                        )

            assert hasattr(self.ion_core_radii, "__len__")
            assert hasattr(self.csd_core_charges, "__len__")
            assert hasattr(self.csd_parameters, "__len__")
        else:
            self.ipd_eV = None
            self.user_defined_lfc = None
            self.ion_core_radii = np.full(self.number_of_states, None)
            self.csd_parameters = np.full(self.number_of_states, None)
            self.csd_core_charges = np.full(self.number_of_states, None)
            self.sec_core_power = None
            self.srr_sigma_parameter = None
            self.crowley_force_constant = 0.9

        self.user_defined_inputs = user_defined_inputs

        self.states, self.overlord_state = self.get_mean_and_all_states(elements)

        # Run Variables
        self.enforce_fsum = enforce_fsum
        self.verbose = verbose
        self.save_to_json = save_to_json

        if save_to_json:
            assert output_file_name is not None, f"Please specify the output file name."
            _, extension = os.path.splitext(output_file_name)

            if not extension == ".json":
                output_file_name += ".json"
        self.output_file_name = output_file_name

        self.hnc_mix_fraction = hnc_mix_fraction
        self.hnc_delta = hnc_delta
        self.hnc_max_iterations = hnc_max_iterations

        # Some asserts, make sure people are paying attention
        if self.models.ii_potential == "SRR":
            assert (
                self.srr_sigma_parameter is not None and self.sec_core_power is not None
            ), "You forget to set key parameters for the SSR ii potential."
        if self.models.ii_potential == "CSD":
            assert any(x is not None for x in self.ion_core_radii) and any(
                x is not None for x in self.ion_core_radii
            ), "You forget to set key parameters for the CSD ii potential."
        if self.models.sf_model == "MSA":
            assert any(
                x is not None for x in self.ion_core_radii
            ), "You forgot to set the ion core radius for the MSA static structure factor approximation."
        if self.models.ii_potential == "SRR":
            assert any(
                x is not None for x in self.ion_core_radii
            ), "You forgot to set the ion core radius for the SRR potential."

    def get_mean_and_all_states(self, elements):
        """
        Initialize an xDave object.
        This will define a mean plasma state and individual states describing each species.
        Note that temperatures are kept consistent for each species, mass density is accounted for using the partial densities.

        Returns:
            states (array): array of states for each species
            overlord_state (PlasmaState): instance of class PlasmaState containing the mean variables
        """
        states = []
        Z_mean = 0.0
        AN_mean = 0.0
        amu_mean = 0.0

        atomic_masses, atomic_numbers = get_atomic_data_for_all_elements(elements)

        sum_term = np.sum(self.partial_densities * atomic_masses)
        for i in range(0, self.number_of_states):

            x = self.partial_densities[i]
            if x == 0:
                warnings.warn(f"Trying to initialize state with zero fractional density. Skipping this one!")
                # remove corresponding element from the list and subtract the number of states to ensure consistency
                np.delete(self.elements, i)
                np.delete(self.charge_states, i)
                self.number_of_states -= 1
                continue

            Z = self.charge_states[i]

            assert Z >= 0.0, f"Trying to set the charge state negative. This is not allowed."
            Z_mean += x * Z
            ni = x * self.mass_density / sum_term
            AN = atomic_numbers[i]
            AN_mean += x * AN
            amu = atomic_masses[i] / ATOMIC_MASS_UNIT  # this is also dumb!
            amu_mean += x * amu
            binding_energies = get_binding_energies_from_element(AN, Z)

            state = PlasmaState(
                electron_temperature=self.electron_temperature,
                ion_temperature=self.ion_temperature,
                mass_density=x * self.mass_density,
                ion_number_density=ni,
                charge_state=Z,
                binding_energies=binding_energies,
                atomic_mass=amu,
                atomic_number=AN,
                ion_core_radius=self.ion_core_radii[i],
                sec_power=self.sec_core_power,
                csd_core_charge=self.csd_core_charges[i],
                csd_parameter=self.csd_parameters[i],
                srr_sigma=self.srr_sigma_parameter,
            )
            states.append(state)

        overlord_state = PlasmaState(
            electron_temperature=self.electron_temperature,
            ion_temperature=self.ion_temperature,
            mass_density=self.mass_density,
            charge_state=Z_mean,
            atomic_mass=amu_mean,
            atomic_number=AN_mean,
            binding_energies=np.array([]),
        )
        self.ocp_flag = len(states) < 2  # (len(np.unique(atomic_numbers)) < len(states)) and len(states) <= 2
        return np.array(states), overlord_state

    ## ------------------------ ##
    ## -- Main run functions -- ##
    ## ------------------------ ##

    def run(self, w, k=None, angle=None, beam_energy=None, mode="DYNAMIC"):
        """
        Main run function.
        Depending on the run model chosen (STATIC/DYNAMIC), this will return either the full DSF calculation
        or the static case.


        Parameters:
            k (float/ array): array or single value of scattering wavenumbers in units of a_B^{-1}
            w (array): array of points in the energy grid, in units of eV.
            angle (float): scattering angle in degrees, optional.
            beam_energy: energy of the probe beam in units of eV, optional.
            mode (str): run mode, either DYNAMIC or STATIC.

        Returns:
            array: total bound-free DSF in units of 1/eV
            array: total free-free DSF in units of 1/eV
            float: Rayleigh Weight, dimensionless
            array: Contributions to the ff DSF by each species in units of 1/eV (non-sensical, for completeness only)
            array: Contributions to the bf DSF by each species in units of 1/eV

        Returns:
            array: array of k values in units of a_B^{-1}
            array: array of static structure factors for each species, shape is determined by the number of elements, non-dimensional
            float: Rayleigh Weight, non-dimensional
            array: array of the screening cloud for each species, non-dimensional
            array: array of the form factors for each species, non-dimensional

        """

        if self.verbose:
            self._print_logo()

        if angle is None:
            assert k is not None, f"You have to set either the angle or the scattering wave number."
            if beam_energy is None:
                beam_energy = 8.0e3
                print(f"Assuming beam energy of 8 keV.") if self.verbose else None
            angle = calculate_angle(q=k, energy=beam_energy)
        elif k is None:
            assert angle is not None, f"You have to set either the angle or the scattering wave number."
            assert beam_energy is not None, f"If you set an angle, you also need to specify the beam energy."
            k = calculate_q(angle=angle, energy=beam_energy)

        k_SI = k / BOHR_RADIUS
        w_SI = w * eV_TO_J

        if mode == "DYNAMIC":
            return self._run_dynamic_mode(k=k_SI, w=w_SI)
        elif mode == "STATIC":
            return self._run_static_mode(k=k_SI)

    def _bf_norm(self, w, ff, bf, k):
        """
        Calculate normalization on the BF factor to enforce the f-sum rule.
        This is currently not being used.
        """
        k /= BOHR_RADIUS
        f_sum = k**2 / 2
        F_tot, F_wff, F_wbf = self.get_itcf(tau=self.tau_array, w=w, ff=ff, bf=bf)
        dF_ff = np.polyfit(self.tau_array[:3], F_wff[:3], deg=1)[0]
        dF_bf = np.polyfit(self.tau_array[:3], F_wbf[:3], deg=1)[0]
        A = -1 / dF_bf * (f_sum + dF_ff)
        return A

    def _run_dynamic_mode(self, k, w):
        """
        Main dynamic function. Internally, everything is run in SI units.

        Parameters:
            k (float): scattering wavenumber in units of a_B^{-1}
            w (array): array of energies in eV

        Returns:
            array: total bound-free DSF in units of 1/eV
            array: total free-free DSF in units of 1/eV
            float: Rayleigh Weight, dimensionless
            array: Contributions to the ff DSF by each species in units of 1/eV (non-sensical, for completeness only)
            array: Contributions to the bf DSF by each species in units of 1/eV
        """

        if self.user_defined_lfc is None:
            lfc_kernel = LFC(state=self.overlord_state)
            lfc = lfc_kernel.calculate_lfc(k=k, w=w, model=self.models.lfc_model)
            print(f"Calculated LFC={lfc}") if self.verbose else None
        else:
            lfc = self.user_defined_lfc
            print(f"Using user-defined LFC={lfc}") if self.verbose else None

        if self.ipd_eV is not None:
            state_ipds = np.full(len(self.states), self.ipd_eV * eV_TO_J)
            if self.verbose:
                print(f"Applying user-defined input IPD: {self.ipd_eV}")
        else:
            state_ipds = get_ipd(
                plasma=self,
                state=self.overlord_state,
                model=self.models.ipd_model,
                user_defined_ipd=self.ipd_eV,
                crowley_force_constant=self.crowley_force_constant,
            )

        ff = FreeFreeDSF(state=self.overlord_state)
        if self.overlord_state.charge_state > 0:
            ff_dsf = ff.get_dsf(k=k, w=w, lfc=lfc, model=self.models.polarisation_model)
        else:
            ff_dsf = np.zeros_like(w)

        ff_tot = ff_dsf * self.overlord_state.charge_state

        bf_tot = np.zeros_like(w)
        ff_i = np.zeros((len(self.states), len(w)))
        bf_i = np.zeros((len(self.states), len(w)))

        if self.verbose:
            print(f"Mean charge state = {self.overlord_state.charge_state}.")

        for i in range(0, len(self.states)):
            state: PlasmaState = self.states[i]
            x = self.partial_densities[i]
            ipd = state_ipds[i]
            if self.verbose:
                print(f"\nRunning state {i} with Z={state.charge_state} and x={x}")

                if self.ipd_eV is None:
                    print(f"Calculated IPD for state {i}={ipd * J_TO_eV} eV\n")

            binding_energies = state.binding_energies * eV_TO_J

            ff_i[i] = x * ff_dsf

            Eb = binding_energies - ipd

            if np.any(ipd < binding_energies[binding_energies < 0.0]):
                warnings.warn(
                    f"IPD {ipd * J_TO_eV} is larger than the binding energy of state {i}: {binding_energies[binding_energies < 0.]* J_TO_eV}. Consider increasing your ionization degree. The bound-free feature is being set to zero."
                )

            bf = BoundFreeDSF(state=state)
            bf_dsf = bf.get_dsf(ZA=state.atomic_number, Zb=state.Zb, k=k, w=w, Eb=Eb, model=self.models.bf_model)
            bf_tot += x * bf_dsf
            bf_i[i] = x * bf_dsf

        # Calculate the Rayleigh weight
        if self.ocp_flag:
            wr_kernel = OCPRayleighWeight(
                overlord_state=self.overlord_state, state=self.states[0], verbose=self.verbose
            )
            rayleigh_weight = wr_kernel.get_rayleigh_weight(
                k=k,
                lfc=lfc,
                sf_model=self.models.sf_model,
                ii_potential=self.models.ii_potential,
                ee_potential=self.models.ee_potential,
                ei_potential=self.models.ei_potential,
                screening_model=self.models.screening_model,
                bridge_function=self.models.bridge_function,
                hnc_max_iterations=self.hnc_max_iterations,
                hnc_mix_fraction=self.hnc_mix_fraction,
                hnc_delta=self.hnc_delta,
                return_full=False,
            )
        else:
            wr_kernel = MCPRayleighWeight(overlord_state=self.overlord_state, states=self.states, verbose=self.verbose)
            rayleigh_weight = wr_kernel.get_rayleigh_weight(
                k=k,
                lfc=lfc,
                sf_model=self.models.sf_model,
                ii_potential=self.models.ii_potential,
                ee_potential=self.models.ee_potential,
                ei_potential=self.models.ei_potential,
                screening_model=self.models.screening_model,
                hnc_mix_fraction=self.hnc_mix_fraction,
                hnc_max_iterations=self.hnc_max_iterations,
                hnc_delta=self.hnc_delta,
                return_full=False,
            )

        dsf = ff_tot + bf_tot

        if self.enforce_fsum:
            if self.verbose:
                print(f"You are currently enforcing a normalization based on the f-sum rule.")
            bf_i /= self.overlord_state.Zb
            bf_tot /= self.overlord_state.Zb
            ff_i /= self.overlord_state.charge_state
            ff_tot /= self.overlord_state.charge_state
            rayleigh_weight /= self.overlord_state.charge_state

        # convert everything to cgs before returning results
        bf_tot /= J_TO_eV
        ff_tot /= J_TO_eV
        dsf /= J_TO_eV
        ff_i /= J_TO_eV
        bf_i /= J_TO_eV

        if self.save_to_json:
            bf_data = {f"species_{i}": row.tolist() for i, row in enumerate(bf_i)}
            result_dict = dict(
                {
                    "w": list(w * J_TO_eV),
                    "lfc": float(lfc),
                    "ipd": ipd * J_TO_eV,
                    "ff": list(ff_tot),
                    "bf": {
                        "tot": list(bf_tot),
                        "bf_i": bf_data,
                    },
                    "dsf": list(dsf),
                    "WR": list(rayleigh_weight),
                    "Sii": None,
                    "qs": None,  # make sure the screening cloud and form factors are being passed down, same for the static structure factor(s)
                    "fs": None,
                }
            )
            self.save_dynamic(fname=self.output_file_name, k=k * BOHR_RADIUS, results=result_dict)

        return (bf_tot, ff_tot, dsf, rayleigh_weight, ff_i, bf_i)

    def _run_static_mode(self, k):
        """
        Main static run function. Internally, everything is run in SI units.

        Parameters:
            k (array): array of scattering wavenumbers in units of a_B^{-1}

        Returns:
            array: array of k values in units of a_B^{-1}
            array: array of static structure factors for each species, shape is determined by the number of elements, non-dimensional
            float: Rayleigh Weight, non-dimensional
            array: array of the screening cloud for each species, non-dimensional
            array: array of the form factors for each species, non-dimensional
        """

        lfc_kernel = LFC(state=self.overlord_state)
        lfc = lfc_kernel.calculate_lfc(k=k, w=0, model=self.models.lfc_model)

        if self.ocp_flag:
            wr_kernel = OCPRayleighWeight(
                overlord_state=self.overlord_state, state=self.states[0], verbose=self.verbose
            )
            _, Sab, rayleigh_weight, qs, fs = wr_kernel.get_rayleigh_weight(
                k=k,
                lfc=lfc,
                sf_model=self.models.sf_model,
                ii_potential=self.models.ii_potential,
                ee_potential=self.models.ee_potential,
                ei_potential=self.models.ei_potential,
                screening_model=self.models.screening_model,
                bridge_function=self.models.bridge_function,
                hnc_mix_fraction=self.hnc_mix_fraction,
                hnc_max_iterations=self.hnc_max_iterations,
                hnc_delta=self.hnc_delta,
                return_full=True,
            )

            Sab_tot = Sab

        else:
            wr_kernel = MCPRayleighWeight(overlord_state=self.overlord_state, states=self.states, verbose=self.verbose)

            _, Sab, rayleigh_weight, qs, fs = wr_kernel.get_rayleigh_weight(
                k=k,
                lfc=lfc,
                sf_model=self.models.sf_model,
                ii_potential=self.models.ii_potential,
                ee_potential=self.models.ee_potential,
                ei_potential=self.models.ei_potential,
                screening_model=self.models.screening_model,
                hnc_mix_fraction=self.hnc_mix_fraction,
                hnc_max_iterations=self.hnc_max_iterations,
                hnc_delta=self.hnc_delta,
                return_full=True,
            )

            Sab_tot = np.zeros_like(k)
            for n1 in range(0, self.number_of_states):
                for n2 in range(0, self.number_of_states):
                    Sab_tot += np.sqrt(self.partial_densities[n1] * self.partial_densities[n2]) * Sab[n1, n2, :]

        if self.save_to_json:
            nspecies = self.number_of_states
            sf_data = dict({})
            qs_data = dict({})
            fs_data = dict({})

            for n1 in range(0, nspecies):
                for n2 in range(0, nspecies):
                    Si = Sab[n1, n2, :]
                    new_row = dict(
                        {
                            f"S_{self.elements[n1]}{self.states[n1].charge_state:.0f}_{self.elements[n2]}{self.states[n2].charge_state:.0f}": list(
                                Si
                            )
                        }
                    )
                    sf_data.update(new_row)

                    if n1 == n2:
                        new_q = dict({f"q_{self.elements[n1]}{self.states[n1].charge_state:.0f}": list(qs[n1])})
                        qs_data.update(new_q)
                        new_f = dict({f"f_{self.elements[n1]}{self.states[n1].charge_state:.0f}": list(fs[n1])})
                        fs_data.update(new_f)

            result_dict = dict(
                {
                    "k": list(k / BOHR_RADIUS),
                    "lfc": list(lfc),
                    "WR": list(rayleigh_weight),
                    "Sii": sf_data,
                    "q": qs_data,
                    "f": fs_data,
                }
            )

            self.save_static(fname=self.output_file_name, results=result_dict)

        # Return outputs in cgs
        return k * BOHR_RADIUS, Sab, Sab_tot, rayleigh_weight, qs, fs, lfc

    def run_inelastic(self, w, k=None, angle=None, beam_energy=None):
        """
        Inelastic run function to ignore the rayleigh weight calculation.

        Parameters:
            k (float/ array): array or single value of scattering wavenumbers in units of a_B^{-1}
            w (array): array of points in the energy grid, in units of eV.
            angle (float): scattering angle in degrees, optional.
            beam_energy: energy of the probe beam in units of eV, optional.
            mode (str): run mode, either DYNAMIC or STATIC.

        Returns:
            array: total bound-free DSF in units of 1/eV
            array: total free-free DSF in units of 1/eV
            float: Rayleigh Weight, dimensionless
            array: Contributions to the ff DSF by each species in units of 1/eV (non-sensical, for completeness only)
            array: Contributions to the bf DSF by each species in units of 1/eV
        """

        if self.verbose:
            self._print_logo()

        if angle is None:
            assert k is not None, f"You have to set either the angle or the scattering wave number."
            if beam_energy is None:
                beam_energy = 8.0e3
                print(f"Assuming beam energy of 8 keV.") if self.verbose else None
            angle = calculate_angle(q=k, energy=beam_energy)
        elif k is None:
            assert angle is not None, f"You have to set either the angle or the scattering wave number."
            assert (
                beam_energy is not None
            ), f"If you set an angle, you also need to specify the beam energy. I can't read your fucking mind."
            k = calculate_q(angle=angle, energy=beam_energy)

        k_SI = k / BOHR_RADIUS
        w_SI = w * eV_TO_J

        if self.user_defined_lfc is None:
            lfc_kernel = LFC(state=self.overlord_state)
            lfc = lfc_kernel.calculate_lfc(k=k, w=w, model=self.models.lfc_model)
            print(f"Calculated LFC={lfc}") if self.verbose else None
        else:
            lfc = self.user_defined_lfc
            print(f"Using user-defined LFC={lfc}") if self.verbose else None

        if self.ipd_eV is not None:
            state_ipds = np.full(len(self.states), self.ipd_eV * eV_TO_J)
            if self.verbose:
                print(f"Applying user-defined input IPD: {self.ipd_eV}")
        else:
            state_ipds = get_ipd(
                plasma=self,
                state=self.overlord_state,
                model=self.models.ipd_model,
                user_defined_ipd=self.ipd_eV,
                crowley_force_constant=self.crowley_force_constant,
            )

        ff = FreeFreeDSF(state=self.overlord_state)
        ff_dsf = ff.get_dsf(k=k_SI, w=w_SI, lfc=lfc, model=self.models.polarisation_model)
        ff_tot = ff_dsf * self.overlord_state.charge_state

        bf_tot = np.zeros_like(w_SI)
        ff_i = np.zeros((len(self.states), len(w_SI)))
        bf_i = np.zeros((len(self.states), len(w_SI)))

        if self.verbose:
            print(f"Mean charge state = {self.overlord_state.charge_state}.")

        for i in range(0, len(self.states)):
            state: PlasmaState = self.states[i]
            x = self.partial_densities[i]
            ipd = state_ipds[i]
            if self.verbose:
                print(f"\nRunning state {i} with Z={state.charge_state} and x={x}\n")
                print(f"Calculated IPD for state {i}={ipd * J_TO_eV} eV\n")

            binding_energies = state.binding_energies * eV_TO_J

            ff_i[i] = x * ff_dsf

            Eb = binding_energies - ipd

            if np.any(ipd < binding_energies[binding_energies < 0.0]):
                warnings.warn(
                    f"IPD {ipd * J_TO_eV} is larger than the binding energy of state {i}: {binding_energies[binding_energies < 0.]* J_TO_eV}. Consider increasing your ionization degree. The bound-free feature is being set to zero."
                )

            bf = BoundFreeDSF(state=state)
            bf_dsf = bf.get_dsf(ZA=state.atomic_number, Zb=state.Zb, k=k_SI, w=w_SI, Eb=Eb, model=self.models.bf_model)
            bf_tot += x * bf_dsf
            bf_i[i] = x * bf_dsf

        if self.enforce_fsum:
            bf *= self._bf_norm(w=w, ff=ff, bf=bf, k=k)

        dsf = ff_tot + bf_tot
        return bf_tot / J_TO_eV, ff_tot / J_TO_eV, dsf / J_TO_eV, ff_i / J_TO_eV, bf_i / J_TO_eV

    def run_elastic(self, k, w):
        """

        Elastic run function to calculate the Rayleigh weight only for a wave number.

        Parameters:
            k (float/ array): array or single value of scattering wavenumbers in units of a_B^{-1}
            w (array): array of points in the energy grid, in units of eV.

        Returns:
            array: array of k values in units of a_B^{-1}
            array: array of static structure factors for each species, shape is determined by the number of elements, non-dimensional
            float: Rayleigh Weight, non-dimensional
            array: array of the screening cloud for each species, non-dimensional
            array: array of the form factors for each species, non-dimensional
        """
        k_value = k / BOHR_RADIUS
        omega_array = w.copy() * eV_TO_J

        lfc_kernel = LFC(state=self.overlord_state)
        lfc = lfc_kernel.calculate_lfc(k=k_value, w=omega_array, model=self.models.lfc_model)

        # Calculate the Rayleigh weight
        if self.ocp_flag:
            wr_kernel = OCPRayleighWeight(state=self.overlord_state, verbose=self.verbose)
            rayleigh_weight = wr_kernel.get_rayleigh_weight(
                k=k_value,
                lfc=lfc,
                sf_model=self.models.sf_model,
                ii_potential=self.models.ii_potential,
                ee_potential=self.models.ee_potential,
                ei_potential=self.models.ei_potential,
                screening_model=self.models.screening_model,
                return_full=False,
            )
        else:
            wr_kernel = MCPRayleighWeight(overlord_state=self.overlord_state, states=self.states, verbose=self.verbose)
            rayleigh_weight = wr_kernel.get_rayleigh_weight(
                k=k_value,
                lfc=lfc,
                sf_model=self.models.sf_model,
                ii_potential=self.models.ii_potential,
                ee_potential=self.models.ee_potential,
                ei_potential=self.models.ei_potential,
                screening_model=self.models.screening_model,
                return_full=False,
            )
        return rayleigh_weight

    ## --------------------- ##
    ## -- Post-processing -- ##
    ## --------------------- ##

    def convolve_with_sif(
        self, omega, bf, ff, dsf, Wr, beam_energy, type="GAUSSIAN", fwhm=10, source_energy=None, source=None
    ):
        """
        Convolve DSF with a source instrument function. You can either specify an analytic type (Gaussian only for now) or input your own
        using the inputs.

        Parameters
            omega (array): energy grid in units of eV
            dsf (array): dynamic structure factor in units of 1/eV
            Wr (float): rayleigh weight describing the elastic feature
            beam_energy (float): energy of the probe beam in units of eV
            type (float, optional): specifies the type of SIF, either analytic or USER_DEFINED
            fwhm (float): defines the forward-half-width-maximum of the analytic SIF, only applied if type is analytic
            source_energ (array): energy grid corresponding to the source in units of eV
            source (array): source intensity in arbitrary units

        Returns:
            array: energy grid of the output spectrum in units of eV
            array: convolved inelastic component spectrum in arbitrary units
            array: convolved elastic component spectrum in arbitrary units
            array: convolved spectrum in arbitrary units
        """

        spec_energy = beam_energy - omega

        if type == "GAUSSIAN":
            assert fwhm is not None
            sigma = fwhm / 2.355
            source = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(omega**2) / (2 * sigma**2))
            source_energy = spec_energy
        elif type == "USER_DEFINED":
            assert source_energy is not None, f"If you want to use a user-defined sif, you need to define it."
            assert source is not None, f"If you want to use a user-defined sif, you need to define it."

        # Safety check on the variables
        if spec_energy[1] - spec_energy[0] < 0:
            spec_energy = spec_energy[::-1]
            flip_spec_ene = True
        else:
            flip_spec_ene = False

        if omega[1] - omega[0] > 0:
            om = omega[::-1]
            S = dsf[::-1]
            S_inel = bf[::-1] + ff[::-1]
        else:
            om = omega
            S = dsf
            S_inel = bf + ff

        if source_energy[1] - source_energy[0] < 0:
            source_energy = source_energy[::-1]
            source_spectrum = source[::-1]
        else:
            source_energy = source_energy
            source_spectrum = source

        spectrum = np.zeros_like(spec_energy)
        inelastic = np.zeros_like(spec_energy)
        source_spectrum /= np.sum(source_spectrum)  # Normalise for convolution

        for i, Ei in enumerate(source_energy):
            Bi = source_spectrum[i]
            scttr_spc = S * (1.0 - om / Ei) ** 2
            # scttr_spc_inel = S_inel * (1.0 - om / Ei) ** 2
            scttr_ene = Ei - om
            spectrum += np.interp(x=spec_energy, xp=scttr_ene, fp=scttr_spc) * Bi
            # inelastic += np.interp(x=spec_energy, xp=scttr_ene, fp=scttr_spc_inel) * Bi

        new_source = np.interp(x=spec_energy, xp=source_energy, fp=source_spectrum)
        new_source /= np.sum(new_source)
        inelastic = spectrum.copy()
        elastic = new_source * Wr / (spec_energy[1] - spec_energy[0])
        spectrum += elastic
        if flip_spec_ene:
            spec_energy = spec_energy[::-1]
            spectrum = spectrum[::-1]
            inelastic = inelastic[::-1]
            elastic = elastic[::-1]
        return spec_energy, inelastic, elastic, spectrum

    def get_itcf(self, w, ff, bf, tau=None):
        """
        Apply the double-sided Laplace transform to the DSF to obtain the imaginary-time correlation function (ITCF).

        Parameters
            omega (array): energy grid in units of eV
            ff (array): free-free dsf in units of 1/eV
            bf (array): bound-free dsf in units of 1/eV
            tau (array): tau-grid to perform the Laplace transform over, optional.

        Returns
            array: tau-grid
            array: total inelastic ITCF
            array: free-free ITCF
            array: bound-free ITCF
        """
        if tau is None:
            tau = self.tau_array
        return laplace(tau=tau, E=w, wff=ff, wbf=bf)

    def get_static_structure_factors(self, w, ff, bf):
        """
        Integrate the dynamic structure factor over the whole energy grid to obtain a static structure factor.

        Parameters
            omega (array): energy grid in units of eV
            ff (array): free-free dsf in units of 1/eV
            bf (array): bound-free dsf in units of 1/eV

        Returns
            float: bound-free static structure factor
            float: free-free static structure factor
        """
        bf_static = np.trapezoid(bf, w)
        ff_static = np.trapezoid(ff, w)
        return bf_static, ff_static

    ## -------------------------------- ##
    ## -- Saving and loading outputs -- ##
    ## -------------------------------- ##

    def save_dynamic(self, fname, k, results):

        setup_dict = dict(
            {
                "xDave Version": "0.0.1a0",  # automate this
                "Time": str(datetime.today().strftime("%Y-%m-%d %H:%M:%S")),
                "models": self.models.toJSON(),
                "mode": {
                    "run_mode": "dynamic",
                    "k": k,
                    # "probe_energy" : probe_energy,
                    # "angle": calculate_angle(q=k * BOHR_RADIUS, energy=probe_energy),
                },
                "hnc_variables": {
                    "mix_fraction": self.hnc_mix_fraction,
                    "delta": self.hnc_delta,
                    "max_iterations": self.hnc_max_iterations,
                },
                "user_defined_inputs": self.user_defined_inputs,
            }
        )

        plasma_parameters_dict = dict(
            {
                "electron_temperature": self.overlord_state.electron_temperature * K_TO_eV,
                "ion_temperature": self.overlord_state.ion_temperature * K_TO_eV,
                "mass_density": self.overlord_state.mass_density * kg_per_m3_TO_g_per_cm3,
                "mean_charge_state": self.overlord_state.charge_state,
                "mean_atomic_number": self.overlord_state.atomic_number,
                "mean_atomic_mass": self.overlord_state.atomic_mass * kg_TO_amu,
                "material": {
                    "elements": list(self.elements),
                    "charge_states": list(self.charge_states),
                    "partial_densities": list(self.partial_densities),
                    "binding_energies": {
                        f"species_{i}": list(self.states[i].binding_energies) for i in range(0, self.number_of_states)
                    },
                },
                "bound_electron_number_density": self.overlord_state.bound_electron_number_density * per_m3_TO_per_cm3,
                "free_electron_number_density": self.overlord_state.free_electron_number_density * per_m3_TO_per_cm3,
                "free_electron_parameters": {
                    "rs": -1,
                    "theta": -1,
                    "fermi_wavenumber": -1,
                    "fermi_energy": -1,
                    "chemical_potential": -1,
                    "debye_inverse_screening_length": -1,
                },  # TODO(HB): add these once they're stored in the plasma state
                "ion_ion_coupling": -1,  # add Gamma_ii
                "electron_ion_coupling": -1,  # add Gamma_ei
            }
        )
        # results_dict = results
        run_info_dict = dict(
            {
                "run_time": -1,
                "Sii": {  # TODO(HB): add these values from HNC solver
                    "convergence": True,
                    "iterations": -1,
                },
            }
        )
        output_dict = dict(
            {
                "setup": setup_dict,
                "plasma_parameters": plasma_parameters_dict,
                "results": results,
                "run_info": run_info_dict,
            }
        )

        with open(fname, "w") as fp:
            json.dump(output_dict, fp)

    def save_static(self, fname, results):
        setup_dict = dict(
            {
                "xDave Version": "0.0.1a0",  # automate this
                "Time": str(datetime.today().strftime("%Y-%m-%d %H:%M:%S")),
                "models": self.models.toJSON(),
                "mode": {
                    "run_mode": "static",
                    # "probe_energy" : probe_energy,
                    # "angle": calculate_angle(q=k * BOHR_RADIUS, energy=probe_energy),
                },
                "hnc_variables": {
                    "mix_fraction": self.hnc_mix_fraction,
                    "delta": self.hnc_delta,
                    "max_iterations": self.hnc_max_iterations,
                },
                "user_defined_inputs": self.user_defined_inputs,
            }
        )
        plasma_parameters_dict = dict(
            {
                "electron_temperature": self.overlord_state.electron_temperature * K_TO_eV,
                "ion_temperature": self.overlord_state.ion_temperature * K_TO_eV,
                "mass_density": self.overlord_state.mass_density * kg_per_m3_TO_g_per_cm3,
                "mean_charge_state": self.overlord_state.charge_state,
                "mean_atomic_number": self.overlord_state.atomic_number,
                "mean_atomic_mass": self.overlord_state.atomic_mass * kg_TO_amu,
                "material": {
                    "elements": list(self.elements),
                    "charge_states": list(self.charge_states),
                    "partial_densities": list(self.partial_densities),
                },
                "bound_electron_number_density": self.overlord_state.bound_electron_number_density * per_m3_TO_per_cm3,
                "free_electron_number_density": self.overlord_state.free_electron_number_density * per_m3_TO_per_cm3,
                "free_electron_parameters": {
                    "rs": -1,
                    "theta": -1,
                    "fermi_wavenumber": -1,
                    "fermi_energy": -1,
                    "chemical_potential": -1,
                    "debye_inverse_screening_length": -1,
                },  # TODO(HB): add these once they're stored in the plasma state
                "ion_ion_coupling": -1,  # add Gamma_ii
                "electron_ion_coupling": -1,  # add Gamma_ei
            }
        )
        # results_dict = results
        run_info_dict = dict(
            {
                "run_time": -1,
                "Sii": {  # TODO(HB): add these values from HNC solver
                    "convergence": True,
                    "iterations": -1,
                },
            }
        )
        output_dict = dict(
            {
                "setup": setup_dict,
                "plasma_parameters": plasma_parameters_dict,
                "results": results,
                "run_info": run_info_dict,
            }
        )

        with open(fname, "w") as fp:
            json.dump(output_dict, fp)

    def load_result_from_json(self, fname):
        with open(fname) as f:
            data = json.load(f)

        return data

    ## ------------------- ##
    ## -- Miscellaneous -- ##
    ## ------------------- ##

    def _print_logo(self):
        print(
            "\n"
            r"    ___   ___   _____ "
            "\n"
            r"__ _|   \ /_\ \ / / __|"
            "\n"
            r"\ \ / |) / _ \ V /| _| "
            "\n"
            r"/_\_\___/_/ \_\_/ |___|"
            "\n".center(20)
        )

from plasma_state import PlasmaState, get_rho_T_from_rs_theta, get_fractions_from_Z, get_fractions_from_Z_partial
from models import ModelOptions
from freefree_dsf import FreeFreeDSF
from boundfree_dsf import BoundFreeDSF
from rayleigh_weight import OCPRayleighWeight, MCPRayleighWeight
from lfc import LFC
from ipd import get_ipd

from utils import (
    calculate_q,
    laplace,
    get_atomic_mass_for_element,
    get_binding_energies_from_elements,
    get_emission_lines_for_element,
    calculate_angle,
    get_mcss_wr_from_status_file,
    load_mcss_result,
)
from unit_conversions import *
from constants import *

import numpy as np
from scipy.signal import fftconvolve
import warnings

import scipy.stats as stats
import matplotlib.pyplot as plt

import os

warnings.filterwarnings("ignore", category=FutureWarning)


class xDave:

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
        user_defined_inputs: dict = None,
    ):
        assert np.sum(partial_densities) == 1.0, f"Fractional densities do not add up 1. Try again sucker."
        self.number_of_states = len(partial_densities)
        self.mass_density = mass_density
        self.electron_temperature = electron_temperature
        self.ion_temperature = ion_temperature
        self.partial_densities = partial_densities
        self.elements = elements
        self.charge_states = charge_states
        self.models = models

        self.enforce_fsum = enforce_fsum
        self.tau_array = np.linspace(0, 1 / (electron_temperature * K_TO_eV), 1000)

        self.states, self.overlord_state = self.initialize()

        self.ipd_eV = None
        self.user_defined_lfc = None
        self.ion_core_radius = None

        if user_defined_inputs is not None:
            keys = user_defined_inputs.keys()
            if "ipd" in keys:
                self.ipd_eV = user_defined_inputs["ipd"]
            if "lfc" in keys:
                self.user_defined_lfc = user_defined_inputs["lfc"]
            if "ion_core_radius" in keys:
                self.ion_core_radius = user_defined_inputs["ion_core_radius"]

    def initialize(self):
        states = []
        Z_mean = 0.0
        AN_mean = 0.0
        ANs = []
        amu_mean = 0.0
        for i in range(0, self.number_of_states):
            element = self.elements[i]
            amu, AN = get_atomic_mass_for_element(element)
            ANs.append(AN)
            x = self.partial_densities[i]
            Z = self.charge_states[i]
            assert Z <= AN, f"Ionization degree for state {i} is larger than the atomic number {AN}. Check your setup."
            binding_energies = get_binding_energies_from_elements(AN)
            state = PlasmaState(
                electron_temperature=self.electron_temperature,
                ion_temperature=self.ion_temperature,
                mass_density=x * self.mass_density,
                charge_state=Z,
                binding_energies=binding_energies,
                atomic_mass=amu,
                atomic_number=AN,
            )
            states.append(state)
            Z_mean += x * Z
            AN_mean += x * AN
            amu_mean += x * amu

        overlord_state = PlasmaState(
            electron_temperature=self.electron_temperature,
            ion_temperature=self.ion_temperature,
            mass_density=self.mass_density,
            charge_state=Z_mean,
            atomic_mass=amu_mean,
            atomic_number=AN_mean,
            binding_energies=np.array([]),
        )
        self.ocp_flag = (len(np.unique(ANs)) < len(states)) and len(states) <= 2
        return np.array(states), overlord_state

    def _bf_norm(self, w, ff, bf, k):
        k /= BOHR_RADIUS
        f_sum = k**2 / 2
        F_tot, F_wff, F_wbf = self.get_itcf(tau=self.tau_array, w=w, ff=ff, bf=bf)
        dF_ff = np.polyfit(self.tau_array[:3], F_wff[:3], deg=1)[0]
        dF_bf = np.polyfit(self.tau_array[:3], F_wbf[:3], deg=1)[0]
        A = -1 / dF_bf * (f_sum + dF_ff)
        return A

    def run(self, k, w):

        self._print_logo()

        lfc_kernel = LFC(state=self.overlord_state)
        lfc = lfc_kernel.calculate_lfc(k=k, w=w, model=self.models.lfc_model)
        print(f"Calculated LFC={lfc}")

        if self.ipd_eV is not None:
            ipd = self.ipd_eV * eV_TO_J
            print(f"Applying user-defined input IPD: {self.ipd_eV}")
        else:
            ipd = get_ipd(state=self.overlord_state, model=self.models.ipd_model, user_defined_ipd=self.ipd_eV)
            print(f"Calculated IPD={ipd * J_TO_eV} eV")

        ff = FreeFreeDSF(state=self.overlord_state)
        ff_dsf = ff.get_dsf(k=k, w=w, lfc=lfc, model=self.models.polarisation_model)
        # the factor of Z/AN is needed to match MCSS results, I will need to figure out where it comes from
        ff_tot = ff_dsf * self.overlord_state.charge_state / self.overlord_state.atomic_number

        bf_tot = np.zeros_like(w)
        ff_i = np.zeros((len(self.states), len(w)))
        bf_i = np.zeros((len(self.states), len(w)))

        print(f"Mean charge state = {self.overlord_state.charge_state}.")

        for i in range(0, len(self.states)):
            state = self.states[i]
            x = self.partial_densities[i]
            print(f"\nRunning state {i} with Z={state.charge_state} and x={x}\n")
            binding_energies = state.binding_energies * eV_TO_J

            ff_i[i] = x * ff_dsf

            Eb = binding_energies - ipd

            if np.any(np.abs(ipd) > (np.abs(binding_energies[binding_energies < 0.0]))):
                warnings.warn(
                    f"IPD {ipd * J_TO_eV} is larger than the binding energy of state {i}: {binding_energies[binding_energies < 0.]* J_TO_eV}. Consider increasing your ionization degree. The bound-free feature is being set to zero."
                )

            bf = BoundFreeDSF(state=state)
            bf_dsf = bf.get_dsf(ZA=state.atomic_number, Zb=state.Zb, k=k, w=w, Eb=Eb, model=self.models.bf_model)
            bf_tot += x * bf_dsf  # * state.charge_state  # / state.atomic_number
            bf_i[i] = x * bf_dsf  # * state.charge_state  # / state.atomic_number

            # This is where the HNC stuff will have to go eventually
            # WR = self.rayleigh_weight

        # Calculate the Rayleigh weight
        if self.ocp_flag:
            wr_kernel = OCPRayleighWeight(state=self.overlord_state, ion_core_radius=1.0 * BOHR_RADIUS)
            rayleigh_weight = wr_kernel.get_rayleigh_weight(
                k=k,
                sf_model=self.models.static_structure_factor_approximation,
                ii_potential=self.models.ii_potential,
                bridge_function=self.models.bridge_function,
                screening="None",
            )
        else:
            wr_kernel = MCPRayleighWeight(
                overlord_state=self.overlord_state, states=self.states, ion_core_radius=1.0 * BOHR_RADIUS
            )
            rayleigh_weight = wr_kernel.get_rayleigh_weight(
                k=k,
                lfc=lfc,
                sf_model=self.models.static_structure_factor_approximation,
                ii_potential=self.models.ii_potential,
                ee_potential=self.models.ee_potential,
                ei_potential=self.models.ei_potential,
                screening_model=self.models.screening_model,
                return_full=False,
            )

        if self.enforce_fsum:
            bf *= self._bf_norm(w=w, ff=ff, bf=bf, k=k)

        # Divide by the atomic number to be consistent with the ff component
        bf_tot /= self.overlord_state.atomic_number
        dsf = ff_tot + bf_tot
        return bf_tot, ff_tot, dsf, rayleigh_weight, ff_i, bf_i

    def run_static_mode(self, k):
        lfc_kernel = LFC(state=self.overlord_state)
        lfc = lfc_kernel.calculate_lfc(k=k, w=0, model=self.models.lfc_model)
        print(f"Calculated LFC={lfc}")

        if self.ocp_flag:
            wr_kernel = OCPRayleighWeight(state=self.overlord_state, ion_core_radius=1.0 * BOHR_RADIUS)
            return wr_kernel.get_rayleigh_weight(
                k=k,
                sf_model=self.models.static_structure_factor_approximation,
                ii_potential=self.models.ii_potential,
                bridge_function=self.models.bridge_function,
                screening="HARD_CORE",
            )

        wr_kernel = MCPRayleighWeight(
            overlord_state=self.overlord_state, states=self.states, ion_core_radius=1.0 * BOHR_RADIUS
        )
        k, Sab, rayleigh_weight, qs, fs = wr_kernel.get_rayleigh_weight(
            k=k,
            lfc=lfc,
            sf_model=self.models.static_structure_factor_approximation,
            ii_potential=self.models.ii_potential,
            ee_potential=self.models.ee_potential,
            ei_potential=self.models.ei_potential,
            screening_model=self.models.screening_model,
            return_full=True,
        )
        return k, Sab, rayleigh_weight, qs, fs

    def convolve_with_sif(self, sif, bf, ff, WR):
        tot_dsf = bf + ff
        inelastic = fftconvolve(tot_dsf, sif, mode="same")  # + WR * sif
        elastic = WR * sif
        spectrum = inelastic + elastic
        return inelastic, elastic, spectrum

    def get_itcf(self, w, ff, bf, tau=None):
        if tau is None:
            tau = self.tau_array
        return laplace(tau=self.tau_array, E=w, wff=ff, wbf=bf)

    def save_result(self, fname, dirname, w, tau, ff, bf, dsf, F_inel, F_bf, F_ff, mode="all"):
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        output_file = os.path.join(dirname, fname + ".csv")
        output_file_itcf = os.path.join(dirname, fname + "_itcf.csv")
        if mode == "all":
            np.savetxt(
                output_file,
                np.transpose(np.array([w, ff, bf, dsf])),
                delimiter=",",
                header="w[J] FF[1/J] BF[1/J] Inel[1/J]",
            )
            np.savetxt(
                output_file_itcf,
                np.transpose(np.array([tau, F_ff, F_bf, F_inel])),
                delimiter=",",
                header="tau[1/eV] F_FF F_BF F_Inel",
            )
        print(f"Saving output to file {fname}")

    def _print_logo(self):
        # print(
        #     ".----------------.  .----------------.  .----------------.  .----------------.\n"
        #     "| .--------------. || .--------------. || .--------------. || .--------------. |\n"
        #     "| |  ____  ____  | || |  _______     | || |  _________   | || |    _______   | |  \n"
        #     "| | |_  _||_  _| | || | |_   __ \    | || | |  _   _  |  | || |   /  ___  |  | |  \n"
        #     "| |   \ \  / /   | || |   | |__) |   | || | |_/ | | \_|  | || |  |  (__ \_|  | |  \n"
        #     "| |    > `' <    | || |   |  __ /    | || |     | |      | || |   '.___`-.   | |  \n"
        #     "| |  _/ /'`\ \_  | || |  _| |  \ \_  | || |    _| |_     | || |  |`\____) |  | |  \n"
        #     "| | |____||____| | || | |____| |___| | || |   |_____|    | || |  |_______.'  | |  \n"
        #     "| |              | || |              | || |              | || |              | |  \n"
        #     "| '--------------' || '--------------' || '--------------' || '--------------' |  \n"
        #     "' '----------------'  '----------------'  '----------------'  '----------------'  \n"
        # )
        print("\n -------------------------------- \n xDAVE C\n --------------------------------\n")


## ----------------------------------------- ##
## ----------------- TESTS ----------------- ##
## ----------------------------------------- ##


def test_setup():
    elements = np.array(["H", "H", "C", "C"])
    rho = 1 * g_per_cm3_TO_kg_per_m3
    T = 70 * eV_TO_K

    partial_densities = np.array([0.15, 0.15, 0.34, 0.36])
    charge_states = np.array([0.0, 1.0, 3, 4])
    user_defined_inputs = None

    models = ModelOptions(polarisation_model="NUMERICAL", bf_model="SCHUMACHER", lfc_model="NONE", ipd_model="NONE")

    omega_array = np.linspace(-1000, 1500, 1000) * eV_TO_J

    k = 8 / ang_TO_m
    q = k * BOHR_RADIUS
    beam_energy = 9.0e3
    angle = calculate_angle(q=q, energy=beam_energy)
    print(f"Running at q={q}, E={beam_energy} -> angle={angle}")

    # Load values from MCSS output files
    mcss_fn = f"mcss_tests/mixed_species_tests/mcss_mixed_species_test_ch_angle={angle:.2f}"
    rayleigh_weight = get_mcss_wr_from_status_file(mcss_fn + "_status.txt")
    mcss_En, mcss_wff, mcss_wbf, mcss_ff, mcss_bf, mcss_el = load_mcss_result(mcss_fn + ".csv")
    mcss_ipd = -3.3087805e001  # eV
    sif = np.zeros_like(omega_array)

    kernel = xDave(
        models=models,
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        elements=elements,
        partial_densities=partial_densities,
        charge_states=charge_states,
        user_defined_inputs=user_defined_inputs,
    )

    bf_tot, ff_tot, dsf, WR, ff_i, bf_i = kernel.run(k=k, w=omega_array)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    mcss_norm = 0.3 * 1 + (1 - 0.3) * 6
    ax = axes[0, 0]
    ax.set_title("Total DSF")
    ax.plot(omega_array * J_TO_eV, bf_tot / J_TO_eV, label="BF")
    ax.plot(omega_array * J_TO_eV, ff_tot / J_TO_eV, label="FF")
    ax.plot(omega_array * J_TO_eV, dsf / J_TO_eV, label="Tot")
    ax.plot(mcss_En, (mcss_wbf + mcss_wff) / mcss_norm, lw=2, ls="dashed", c="black", label="MCSS")
    # ax.plot(mcss_En, (mcss_wbf + mcss_wff), lw=2, c="black", ls="dashed", label="MCSS")
    ax.legend()

    ax = axes[1, 0]
    ax.set_title("FF contributions")
    ax.plot(omega_array * J_TO_eV, ff_i[0] / J_TO_eV, label="H0: FF")
    ax.plot(omega_array * J_TO_eV, ff_i[1] / J_TO_eV, label="H1: FF")
    ax.plot(omega_array * J_TO_eV, ff_i[2] / J_TO_eV, label="C3: FF")
    ax.plot(omega_array * J_TO_eV, ff_i[3] / J_TO_eV, label="C4: FF")
    ax.plot(omega_array * J_TO_eV, ff_tot / J_TO_eV, label="Tot FF")
    ax.plot(mcss_En, mcss_wff / mcss_norm, lw=2, c="black", ls="dashed", label="MCSS")
    # ax.plot(mcss_En, mcss_wff, lw=2, c="black", ls="solid", label="MCSS")
    ax.legend()

    ax = axes[1, 1]
    ax.set_title("BF contributions")
    ax.plot(omega_array * J_TO_eV, bf_i[0] / J_TO_eV, label="H0: BF")
    ax.plot(omega_array * J_TO_eV, bf_i[1] / J_TO_eV, label="H1: BF")
    ax.plot(omega_array * J_TO_eV, bf_i[2] / J_TO_eV, label="C3: BF")
    ax.plot(omega_array * J_TO_eV, bf_i[3] / J_TO_eV, label="C4: BF")
    ax.plot(omega_array * J_TO_eV, bf_tot / J_TO_eV, label="Tot BF")
    ax.plot(mcss_En, mcss_wbf / mcss_norm, lw=2, c="black", ls="dashed", label="MCSS")
    # ax.plot(mcss_En, mcss_wbf, lw=2, c="black", ls="solid", label="MCSS")
    ax.legend()

    sif = stats.norm.pdf(omega_array, 0, 2 * eV_TO_J)
    sif /= np.max(sif)
    WR *= J_TO_eV

    inelastic, elastic, spectrum = kernel.convolve_with_sif(sif=sif, bf=bf_tot, ff=ff_tot, WR=WR)

    ax = axes[0, 1]
    ax.set_title("Spectrum")
    ax.plot(omega_array * J_TO_eV, inelastic / np.max(spectrum), label="inel", ls="-.")
    ax.plot(omega_array * J_TO_eV, elastic / np.max(spectrum), label="el", ls="-.")
    ax.plot(omega_array * J_TO_eV, spectrum / np.max(spectrum), label="tot", ls="-.")
    ax.legend()
    ax.set_xlim(-800, 900)

    plt.tight_layout()
    plt.show()
    fig.savefig(
        f"ch_test_T={T*K_TO_eV}_rho={rho*kg_per_m3_TO_g_per_cm3}_Z={kernel.overlord_state.charge_state}.pdf", dpi=200
    )


def test_be():
    rs = 3
    theta = 1
    # atomic_mass = 9.0121831  # amu
    Z_mean = 3.73
    rho = 22.0 * g_per_cm3_TO_kg_per_m3
    T = 150 * eV_TO_K

    beam_energy = 9.0e3  # * eV_TO_J
    angles = np.array([13, 30, 45, 60, 80, 100, 120, 140, 160])
    ks = calculate_q(angle=angles, energy=beam_energy) / BOHR_RADIUS
    omega_array = np.linspace(-800, 1400, 1000) * eV_TO_J

    WR = 0.1

    models = ModelOptions(
        polarisation_model="NUMERICAL", bf_model="SCHUMACHER", lfc_model="DORNHEIM_ESA", ipd_model="STEWART_PYATT"
    )
    Z1, Z2, x1, x2 = get_fractions_from_Z(Z=Z_mean)
    xs = np.array([x1, x2])

    elements = np.array(["Be", "Be"])
    charge_states = np.array([Z1, Z2])

    user_defined_inputs = {"ipd": 0.0, "lfc": 1.0, "ion_core_radius": 2.0}

    xdave = xDave(
        models=models,
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        elements=elements,
        partial_densities=xs,
        charge_states=charge_states,
        user_defined_inputs=user_defined_inputs,
    )

    k = 8 / ang_TO_m
    q = k * BOHR_RADIUS

    bf_tot, ff_tot, dsf, WR, _, _ = xdave.run(k=k, w=omega_array)

    print(f"Calculate Rayleigh weight: {WR}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 10))

    ax = axes[0]
    ax.plot(omega_array * J_TO_eV, bf_tot / J_TO_eV, label="BF")
    ax.plot(omega_array * J_TO_eV, ff_tot / J_TO_eV, label="FF")
    ax.plot(omega_array * J_TO_eV, dsf / J_TO_eV, label="Tot")
    ax.legend()

    sif = stats.norm.pdf(omega_array, 0, 2 * eV_TO_J)
    sif /= np.max(sif)
    WR *= J_TO_eV

    inelastic, elastic, spectrum = xdave.convolve_with_sif(sif=sif, bf=bf_tot, ff=ff_tot, WR=WR)

    ax = axes[1]
    ax.plot(omega_array * J_TO_eV, inelastic / np.max(spectrum), label="inel", ls="-.")
    ax.plot(omega_array * J_TO_eV, elastic / np.max(spectrum), label="el", ls="-.")
    ax.plot(omega_array * J_TO_eV, spectrum / np.max(spectrum), label="tot", ls="-.")
    ax.legend()
    ax.set_xlim(-800, 750)

    plt.show()
    fig.savefig(f"beryllium_test_rs={rs}_theta={theta}_Z={Z_mean}_q={q:.2f}.pdf", dpi=200)


if __name__ == "__main__":

    #     ##TODO(Hannah):
    #     ## check that the mass density and number of electrons is being handled correctly across all states
    #     ## compare against MCSS and PIMC for this set of conditions
    #     ## Add IPD model: DONE
    #     ## Clean up bf call (arguments are a bit messy): DONE
    #     ## Start calculating things like kF, EF, omega_p, etc. for the plasma state upon initialisation to avoid extra computation
    #     ## Start timing and looking at how much this scales with number of points
    #     ## I should move away from defining states by their mass density (problematic when you have mixed species) and just look at electron number density... probably a lot easier to split up: STILL THINKING ABOUT THIS
    # test_setup()
    test_be()

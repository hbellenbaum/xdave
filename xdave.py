from plasma_state import PlasmaState, get_rho_T_from_rs_theta, get_fractions_from_Z
from models import ModelOptions
from freefree_dsf import FreeFreeDSF
from boundfree_dsf import BoundFreeDSF
from lfc import LFC
from ipd import get_ipd

from utils import (
    calculate_q,
    laplace,
    get_atomic_mass_for_element,
    get_binding_energies_from_elements,
    get_emission_lines_for_element,
)
from unit_conversions import *
from constants import *

import numpy as np
from scipy.signal import fftconvolve
import warnings

import scipy.stats as stats
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)


class Setup:

    def __init__(
        self,
        mass_density: float,
        electron_temperature: float,
        ion_temperature: float,
        elements: np.array,
        partial_densities: np.array,
        charge_states: np.array,
        user_defined_inputs: dict,
    ):
        assert np.sum(partial_densities) == 1.0, f"Fractional densities do not add up 1. Try again sucker."
        self.number_of_states = len(partial_densities)
        self.mass_density = mass_density
        self.electron_temperature = electron_temperature
        self.ion_temperature = ion_temperature
        self.partial_densities = partial_densities
        self.elements = elements
        self.charge_states = charge_states

        # Not sure yet what the best way of handling this is yet, but I am hoping to include a dict of the optional inputs (LFC, IPD, etc.)
        # to pass onto the state object
        self.user_defined_inputs = user_defined_inputs

        self.states, self.overlord_state = self.initialize()

    def initialize(self):

        states = []
        Z_mean = 0.0
        AN_mean = 0.0
        amu_mean = 0.0
        for i in range(0, self.number_of_states):
            element = self.elements[i]
            amu, AN = get_atomic_mass_for_element(element)
            x = self.partial_densities[i]
            Z = self.charge_states[i]
            assert Z <= AN, f"Ionization degree for state {i} is larger than the atomic number{AN}. Check your setup."
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
            electron_temperature=state.electron_temperature,
            ion_temperature=state.ion_temperature,
            mass_density=state.mass_density,
            charge_state=Z_mean,
            atomic_mass=amu_mean,
            atomic_number=AN_mean,
            binding_energies=np.array([]),
        )
        return np.array(states), overlord_state


class xDave:

    def __init__(
        self,
        models: ModelOptions,
        overlord_state: PlasmaState,
        states: np.array,
        fractions: np.array,
        # binding_energies: np.array,
        rayleigh_weight: float,
        sif: np.array,
        ipd: float = None,
    ):
        self.models = models
        self.states = states
        self.fractions = fractions
        # self.binding_energies = binding_energies
        self.rayleigh_weight = rayleigh_weight
        self.sif = sif
        self.ipd_eV = ipd

        self.overlord_state = overlord_state

    def run(self, k, w):

        self._print_logo()

        ff_tot = np.zeros_like(w)
        bf_tot = np.zeros_like(w)

        lfc_kernel = LFC(models=self.models, state=self.overlord_state)
        lfc = lfc_kernel.calculate_lfc(k=k, w=w, model=self.models.lfc_model)
        print(f"Calculated LFC={lfc}")

        if self.ipd_eV is not None:
            ipd = self.ipd_eV * eV_TO_J
            print(f"Applying user-defined input IPD: {self.ipd_eV}")
        else:
            ipd = get_ipd(state=self.overlord_state, model=self.models.ipd_model, user_defined_ipd=self.ipd_eV)
            print(f"Calculated IPD={ipd * J_TO_eV} eV")

        ff_i = np.zeros((len(self.states), len(w)))
        bf_i = np.zeros((len(self.states), len(w)))

        for i in range(0, len(self.states)):
            state = self.states[i]
            x = self.fractions[i]
            print(f"\nRunning state {i} with Z={state.charge_state} and x={x}\n")
            binding_energies = state.binding_energies * eV_TO_J

            ff = FreeFreeDSF(state=state)
            ff_dsf = ff.get_dsf(k=k, w=w, lfc=lfc, model=self.models.polarisation_model)
            ff_tot += x * ff_dsf
            ff_i[i] = x * ff_dsf

            Eb = binding_energies - ipd

            if np.any(np.abs(ipd) > (np.abs(binding_energies[binding_energies < 0.0]))):
                warnings.warn(
                    f"IPD {ipd * J_TO_eV} is larger than the binding energy of state {i}: {binding_energies[binding_energies < 0.]* J_TO_eV}. Consider increasing your ionization degree. The bound-free feature is being set to zero."
                )

            bf = BoundFreeDSF(state=state)
            bf_dsf = bf.get_dsf(ZA=state.atomic_number, Zb=state.Zb, k=k, w=w, Eb=Eb, model=self.models.bf_model)
            bf_tot += x * bf_dsf
            bf_i[i] = x * bf_dsf

            # This is where the HNC stuff will have to go eventually
            WR = self.rayleigh_weight

        dsf = ff_tot + bf_tot
        return bf_tot, ff_tot, dsf, WR, ff_i, bf_i

    def convolve_with_sif(self, sif, bf, ff, WR):
        tot_dsf = bf + ff
        inelastic = fftconvolve(tot_dsf, sif, mode="same")  # + WR * sif
        elastic = WR * sif
        spectrum = inelastic + elastic
        return inelastic, elastic, spectrum

    def get_itcf(self, tau, w, ff, bf):
        return laplace(tau=tau, E=w, wff=ff, wbf=bf)

    def _print_logo(self):
        # ' .----------------.  .----------------.  .----------------.  .----------------. '
        # '| .--------------. || .--------------. || .--------------. || .--------------. |'
        # '| |  ____  ____  | || |  _______     | || |  _________   | || |    _______   | |  '
        # '| | |_  _||_  _| | || | |_   __ \    | || | |  _   _  |  | || |   /  ___  |  | |  '
        # '| |   \ \  / /   | || |   | |__) |   | || | |_/ | | \_|  | || |  |  (__ \_|  | |  '
        # '| |    > `' <    | || |   |  __ /    | || |     | |      | || |   '.___`-.   | |  '
        # '| |  _/ /'`\ \_  | || |  _| |  \ \_  | || |    _| |_     | || |  |`\____) |  | |  '
        # '| | |____||____| | || | |____| |___| | || |   |_____|    | || |  |_______.'  | |  '
        # '| |              | || |              | || |              | || |              | |  '
        # '| '--------------' || '--------------' || '--------------' || '--------------' |  '
        # '' '----------------'  '----------------'  '----------------'  '----------------'  '
        print("\n -------------------------------- \n xDAVE C\n --------------------------------\n")


def test_setup():
    elements = np.array(["H", "H", "C", "C"])
    rho = 1 * g_per_cm3_TO_kg_per_m3
    T = 70 * eV_TO_K

    partial_densities = np.array([0.0, 0.3, 0.34, 0.36])
    charge_states = np.array([0.0, 1.0, 3, 4])
    user_defined_inputs = None

    models = ModelOptions(
        polarisation_model="NUMERICAL", bf_model="SCHUMACHER", lfc_model="DORNHEIM_ESA", ipd_model="STEWART_PYATT"
    )
    # binding_energies = np.array([-13.6, -130]) * eV_TO_J

    c_emission_lines = get_emission_lines_for_element(element="C")

    setup = Setup(
        mass_density=rho,
        electron_temperature=T,
        ion_temperature=T,
        # models=models,
        elements=elements,
        partial_densities=partial_densities,
        charge_states=charge_states,
        user_defined_inputs=user_defined_inputs,
    )
    states = setup.states
    # print(states)
    rayleigh_weight = 0.3
    omega_array = np.linspace(-1000, 1500, 1000) * eV_TO_J

    sif = np.zeros_like(omega_array)

    kernel = xDave(
        models=models,
        states=states,
        fractions=partial_densities,
        rayleigh_weight=rayleigh_weight,
        overlord_state=setup.overlord_state,
        ipd=None,
        sif=sif,
    )

    k = 8 / ang_TO_m
    q = k * BOHR_RADIUS

    bf_tot, ff_tot, dsf, WR, ff_i, bf_i = kernel.run(k=k, w=omega_array)
    fig, axes = plt.subplots(1, 4, figsize=(14, 10))

    ax = axes[0]
    ax.set_title("Total DSF")
    for key, value in c_emission_lines.items():
        print(key, value)
        # if value > 0.0:
        ax.axvline(x=-value, label=f"{key}", ls=":")
        # print("{} is of the type {} and is located at {}".format(key, value["type"], value["locatie"]))
    ax.plot(omega_array * J_TO_eV, bf_tot / J_TO_eV, label="BF")
    ax.plot(omega_array * J_TO_eV, ff_tot / J_TO_eV, label="FF")
    ax.plot(omega_array * J_TO_eV, dsf / J_TO_eV, label="Tot")
    ax.legend()

    ax = axes[1]
    ax.set_title("FF contributions")
    ax.plot(omega_array * J_TO_eV, ff_i[0] / J_TO_eV, label="H0: FF")
    ax.plot(omega_array * J_TO_eV, ff_i[1] / J_TO_eV, label="H1: FF")
    ax.plot(omega_array * J_TO_eV, ff_i[2] / J_TO_eV, label="C3: FF")
    ax.plot(omega_array * J_TO_eV, ff_i[3] / J_TO_eV, label="C4: FF")
    ax.plot(omega_array * J_TO_eV, ff_tot / J_TO_eV, label="Tot FF")
    ax.legend()

    ax = axes[2]
    ax.set_title("BF contributions")
    ax.plot(omega_array * J_TO_eV, bf_i[0] / J_TO_eV, label="H0: BF")
    ax.plot(omega_array * J_TO_eV, bf_i[1] / J_TO_eV, label="H1: BF")
    ax.plot(omega_array * J_TO_eV, bf_i[2] / J_TO_eV, label="C3: BF")
    ax.plot(omega_array * J_TO_eV, bf_i[3] / J_TO_eV, label="C4: BF")
    ax.plot(omega_array * J_TO_eV, bf_tot / J_TO_eV, label="Tot BF")
    ax.legend()

    sif = stats.norm.pdf(omega_array, 0, 2 * eV_TO_J)
    sif /= np.max(sif)
    WR *= J_TO_eV

    inelastic, elastic, spectrum = kernel.convolve_with_sif(sif=sif, bf=bf_tot, ff=ff_tot, WR=WR)

    ax = axes[3]
    ax.set_title("Spectrum")
    ax.plot(omega_array * J_TO_eV, inelastic / np.max(spectrum), label="inel", ls="-.")
    ax.plot(omega_array * J_TO_eV, elastic / np.max(spectrum), label="el", ls="-.")
    ax.plot(omega_array * J_TO_eV, spectrum / np.max(spectrum), label="tot", ls="-.")
    ax.legend()
    ax.set_xlim(-800, 900)

    plt.tight_layout()
    plt.show()
    fig.savefig(
        f"ch_test_T={T*K_TO_eV}_rho={rho*kg_per_m3_TO_g_per_cm3}_Z={setup.overlord_state.charge_state}.pdf", dpi=200
    )


def test_be():
    rs = 3
    theta = 1
    atomic_mass = 9.0121831  # amu
    Z_mean = 3.73
    # rho, T = get_rho_T_from_rs_theta(rs=rs, theta=theta, atomic_mass=atomic_mass)
    # rho *= g_per_cm3_TO_kg_per_m3
    # T *= eV_TO_K
    rho = 22.0 * g_per_cm3_TO_kg_per_m3
    T = 150 * eV_TO_K

    state_B3 = PlasmaState(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_state=3.0,
        atomic_mass=atomic_mass,
        atomic_number=4,
    )

    state_B4 = PlasmaState(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_state=4.0,
        atomic_mass=atomic_mass,
        atomic_number=4,
    )
    ##TODO(Hannah):
    ## check that the mass density and number of electrons is being handled correctly across all states
    ## compare against MCSS and PIMC for this set of conditions
    ## Add IPD model
    ## Clean up bf call (arguments are a bit messy)
    ## Start calculating things like kF, EF, omega_p, etc. for the plasma state upon initialisation to avoid extra computation
    ## Start timing and looking at how much this scales with number of points
    ## I should move away from defining states by their mass density (problematic when you have mixed species) and just look at electron number density... probably a lot easier to split up

    beam_energy = 9.0e3  # * eV_TO_J
    angles = np.array([13, 30, 45, 60, 80, 100, 120, 140, 160])
    ks = calculate_q(angle=angles, energy=beam_energy) / BOHR_RADIUS
    omega_array = np.linspace(-800, 1400, 1000) * eV_TO_J

    WR = 0.1

    models = ModelOptions(
        polarisation_model="NUMERICAL", bf_model="SCHUMACHER", lfc_model="DORNHEIM_ESA", ipd_model="STEWART_PYATT"
    )
    x1, x2 = get_fractions_from_Z(Z=Z_mean)
    xs = np.array([x1, x2])

    setup = Setup(
        mass_density=rho,
        electron_temperature=T,
        ion_temperature=T,
        models=models,
        partial_densities=xs,
        binding_energies=np.array([-111.5]) * eV_TO_J,
        atomic_weights=np.array([4, 4]),
        charge_states=np.array([3, 4]),
    )

    print(
        f"Running fractions: {x1} for charge {state_B3.charge_state}\n"
        f"and {x2} for charge {state_B4.charge_state}\n"
    )
    xdave = xDave(
        models=models,
        states=np.array([state_B3, state_B4]),
        fractions=xs,
        rayleigh_weight=WR,
        sif=np.zeros_like(omega_array),
    )

    # q = ks[-1] * BOHR_RADIUS
    # k = q / BOHR_RADIUS
    k = 8 / ang_TO_m
    q = k * BOHR_RADIUS

    bf_tot, ff_tot, dsf, Wr = xdave.run(k=k, w=omega_array)

    fig, axes = plt.subplots(1, 2, figsize=(14, 10))

    # plt.figure()
    ax = axes[0]
    ax.plot(omega_array * J_TO_eV, bf_tot / J_TO_eV, label="BF")
    ax.plot(omega_array * J_TO_eV, ff_tot / J_TO_eV, label="FF")
    ax.plot(omega_array * J_TO_eV, dsf / J_TO_eV, label="Tot")
    ax.legend()
    # plt.show()

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

    test_setup()

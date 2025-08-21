from plasma_state import PlasmaState, get_rho_T_from_rs_theta, get_fractions_from_Z
from models import ModelOptions
from freefree_dsf import FreeFreeDSF
from boundfree_dsf import BoundFreeDSF
from lfc import LFC

from utils import calculate_q, laplace
from unit_conversions import *
from constants import *

import numpy as np
from scipy.signal import fftconvolve


class xDave:

    def __init__(
        self,
        models: ModelOptions,
        states: np.array,
        fractions: np.array,
        binding_energies: np.array,
        rayleigh_weight: float,
        sif: np.array,
    ):
        self.models = models
        self.states = states
        self.fractions = fractions
        self.binding_energies = binding_energies
        self.rayleigh_weight = rayleigh_weight
        self.sif = sif

    def run(self, k, w):

        self._print_logo()

        ff_tot = np.zeros_like(w)
        bf_tot = np.zeros_like(w)

        # ff_dsfs = np.zeros_like(self.states)

        for i in range(0, len(self.states)):
            state = self.states[i]
            x = self.fractions[i]
            lfc_kernel = LFC(models=self.models, state=state)
            lfc = lfc_kernel.calculate_lfc(k=k, w=w, model=self.models.lfc_model)
            print(f"State {i}: LFC = {lfc}")

            ff = FreeFreeDSF(state=state, models=self.models)
            ff_dsf = ff.get_dsf(k=k, w=w, lfc=lfc)
            # ff_dsfs[i] = ff_dsf
            ff_tot += x * ff_dsf
            # print(ff_dsfs[i])

            bf = BoundFreeDSF(state=state, models=self.models)
            bf_dsf = bf.get_dsf(ZA=state.atomic_number, Zb=state.Zb, k=k, w=w, Eb=self.binding_energies)
            bf_tot += x * bf_dsf

            # This is where the HNC stuff will have to go eventually
            WR = self.rayleigh_weight

        dsf = ff_tot + bf_tot
        return bf_tot, ff_tot, dsf, WR

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


def main():
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    Z_mean = 0.51

    rs = 3
    theta = 1
    atomic_mass = 1.00784
    rho, T = get_rho_T_from_rs_theta(rs=rs, theta=theta, atomic_mass=atomic_mass)
    rho *= g_per_cm3_TO_kg_per_m3
    T *= eV_TO_K

    models = ModelOptions(polarisation_model="NUMERICAL_RPA", bf_model="SCHUMACHER")

    state_bf = PlasmaState(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_state=0.0,
        atomic_mass=atomic_mass,
        atomic_number=1,
    )

    beam_energy = 9.0e3
    angles = np.array([13, 30, 45, 60, 80, 100, 120, 140, 160])
    ks = calculate_q(angle=angles, energy=beam_energy) / BOHR_RADIUS
    omega_array = np.linspace(-100, 100, 500) * eV_TO_J

    binding_energies = (
        np.array(
            [
                -13.6,
            ]
        )
        * eV_TO_J
    )

    k = ks[0]

    bf_kernel = BoundFreeDSF(state=state_bf, models=models)
    bf_dsf = bf_kernel.get_dsf(ZA=1.0, Zb=state_bf.Zb, k=k, w=omega_array, Eb=binding_energies)

    state_ff = PlasmaState(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_state=1.0,
        atomic_mass=atomic_mass,
        atomic_number=1,
    )
    ff_kernel = FreeFreeDSF(state=state_ff, models=models)
    ff_dsf = ff_kernel.get_dsf(k=k, w=omega_array, lfc=0.0)

    tot_dsf = 0.5 * ff_dsf + 0.5 * bf_dsf

    WR = 1.2

    sif = stats.norm.pdf(omega_array, 0, 2 * eV_TO_J)
    sif /= np.max(sif)
    # plt.figure()
    # plt.plot(omega_array, sif)
    # plt.show()

    # bf_sif = np.convolve(sif, bf_dsf, mode="same")
    # spectrum = np.convolve(tot_dsf, sif, mode="same")
    inelastic = fftconvolve(tot_dsf, sif, mode="same")  # + WR * sif
    elastic = WR * sif * J_TO_eV
    spectrum = inelastic + elastic

    # print(bf_dsf)
    # plt.figure()
    fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    ax.plot(omega_array * J_TO_eV, 0.5 * bf_dsf / J_TO_eV, label="BF", c="crimson", ls="-.")
    ax.plot(omega_array * J_TO_eV, 0.5 * ff_dsf / J_TO_eV, label="FF", c="navy", ls="-.")
    ax.plot(omega_array * J_TO_eV, tot_dsf / J_TO_eV, label="TOT", c="darkgreen", ls="-.")
    # ax.plot(omega_array * J_TO_eV, sif / J_TO_eV, label="SIF", c="black", ls="-.")
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.set_ylabel(r"DSF [1/eV]")
    ax.legend()

    ax = axes[1]
    ax.plot(omega_array * J_TO_eV, sif / np.max(sif), label="SIF", ls="-.")
    ax.plot(omega_array * J_TO_eV, inelastic / np.max(spectrum), label="Inelastic", ls="-.")
    ax.plot(omega_array * J_TO_eV, elastic / np.max(spectrum), label="Elastic", ls="-.")
    ax.plot(omega_array * J_TO_eV, spectrum / np.max(spectrum), label="Spectrum", ls="-.")
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.set_ylabel(r"Signal [arb. units]")
    # ax.plot(omega_array, bf_sif / np.max(bf_sif), label="BF", ls="solid")
    ax.legend()
    fig.suptitle(f"Hydrogen at rs={rs}, theta={theta}, Z=0.5")
    plt.show()

    # return


if __name__ == "__main__":

    import scipy.stats as stats
    import matplotlib.pyplot as plt

    rs = 3
    theta = 1
    atomic_mass = 1.00784
    Z_mean = 0.51
    rho, T = get_rho_T_from_rs_theta(rs=rs, theta=theta, atomic_mass=atomic_mass)
    rho *= g_per_cm3_TO_kg_per_m3
    T *= eV_TO_K

    state_H0 = PlasmaState(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_state=0.0,
        atomic_mass=atomic_mass,
        atomic_number=1,
    )

    state_H1 = PlasmaState(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_state=1.0,
        atomic_mass=atomic_mass,
        atomic_number=1,
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
    omega_array = np.linspace(-300, 300, 500) * eV_TO_J

    WR = 1.2

    models = ModelOptions(polarisation_model="NUMERICAL", bf_model="SCHUMACHER", lfc_model="DORNHEIM_ESA")
    x1, x2 = get_fractions_from_Z(Z=Z_mean)
    xs = np.array([x1, x2])

    print(
        f"Running fractions: {x1} for charge {state_H0.charge_state}\n"
        f"and {x2} for charge {state_H1.charge_state}\n"
    )
    xdave = xDave(
        models=models,
        states=np.array([state_H0, state_H1]),
        fractions=xs,
        binding_energies=np.array([-13.6]) * eV_TO_J,
        rayleigh_weight=WR,
        sif=np.zeros_like(omega_array),
    )

    bf_tot, ff_tot, dsf, Wr = xdave.run(k=ks[0], w=omega_array)

    plt.figure()
    plt.plot(omega_array * J_TO_eV, bf_tot / J_TO_eV, label="BF")
    plt.plot(omega_array * J_TO_eV, ff_tot / J_TO_eV, label="FF")
    plt.plot(omega_array * J_TO_eV, dsf / J_TO_eV, label="Tot")
    plt.legend()
    plt.show()

    sif = stats.norm.pdf(omega_array, 0, 2 * eV_TO_J)
    sif /= np.max(sif)
    WR *= J_TO_eV

    inelastic, elastic, spectrum = xdave.convolve_with_sif(sif=sif, bf=bf_tot, ff=ff_tot, WR=WR)

    plt.figure()
    plt.plot(omega_array * J_TO_eV, inelastic / np.max(spectrum), label="inel", ls="-.")
    plt.plot(omega_array * J_TO_eV, elastic / np.max(spectrum), label="el", ls="-.")
    plt.plot(omega_array * J_TO_eV, spectrum / np.max(spectrum), label="tot", ls="-.")
    plt.legend()
    plt.show()

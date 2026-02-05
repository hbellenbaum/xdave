from xdave import *
from xdave.plasma_state import get_fractions_from_Z_partial, get_fractions_from_Z
from xdave.utils import (
    calculate_q,
    load_mcss_result,
    get_mcss_wr_from_status_file,
    load_mcss_result_ar,
    load_mcss_result_ar_3species,
)
from xdave.unit_conversions import K_TO_eV, kg_per_m3_TO_g_per_cm3

import numpy as np
import matplotlib.pyplot as plt
import os


def test_sif():
    T = 80.0  # eV
    rho = 3.5  # g/cc
    ZC = 4.5  # also ZC=2.5, 3.5, 4.0, 4.5
    ZH = 1.0
    xH = 0.2
    angle = 120  # degrees
    beam_energy = 8.5e3  # eV
    q = calculate_q(angle=angle, energy=beam_energy)
    print(f"Running at q={q:.3f}")

    elements = np.array(["H", "C", "C"])

    Zmin, Zmax, xmin, xmax = get_fractions_from_Z_partial(ZC, x0=xH)
    partial_densities = np.array([xH, xmin, xmax])
    charge_states = np.array([ZH, Zmin, Zmax])
    user_defined_inputs = dict()

    models = ModelOptions(
        polarisation_model="DANDREA_FIT",
        bf_model="SCHUMACHER",
        lfc_model="NONE",
        ipd_model="NONE",
        screening_model="DEBYE_HUCKEL",
    )

    k = q  # 1/aB

    omega_array = np.arange(-4000, 4000, 1.0)  # eV

    kernel = xDave(
        models=models,
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_states=charge_states,
        elements=elements,
        partial_densities=partial_densities,
        user_defined_inputs=None,
        enforce_fsum=False,
        save_to_json=True,
    )

    mcss_norm = 1  # kernel.overlord_state.atomic_number

    bf_tot, ff_tot, dsf, WR, ff_i, bf_i = kernel.run(k=k, w=omega_array)

    if False:
        plt.figure()
        plt.plot(omega_array, bf_tot, c="magenta", label="BF")
        plt.plot(omega_array, ff_tot, c="limegreen", label="FF")
        plt.plot(omega_array, dsf, c="orange", label="DSF")
        plt.legend()
        plt.show()

    energy1, inelastic1, elastic1, spectrum1 = kernel.convolve_with_sif(
        omega=omega_array, bf=bf_tot, ff=ff_tot, dsf=dsf, Wr=WR, beam_energy=beam_energy, type="GAUSSIAN", fwhm=26
    )

    def asym_gaussian(x, A, x0, sigma_left, sigma_right, baseline=0.0):
        """Credit: W. Martin :)"""
        x = np.asarray(x)
        xprime = x - x0
        sigma = np.where(x < x0, sigma_left, sigma_right)
        return xprime, baseline + A * np.exp(-0.5 * (xprime / sigma) ** 2)

    x = omega_array + beam_energy
    A = 0.025
    x0 = beam_energy
    sigma_left = 13
    sigma_right = 13
    ene_user_defined, sif_user_defined = asym_gaussian(x=x, A=A, x0=x0, sigma_left=sigma_left, sigma_right=sigma_right)
    energy2, inelastic2, elastic2, spectrum2 = kernel.convolve_with_sif(
        omega=omega_array,
        bf=bf_tot,
        ff=ff_tot,
        dsf=dsf,
        Wr=WR,
        beam_energy=beam_energy,
        type="USER_DEFINED",
        source_energy=ene_user_defined + beam_energy,
        source=sif_user_defined,
    )

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(energy1, spectrum1, label="GAUSSIAN", ls="-.", c="navy")
    axes[0].plot(energy2, spectrum2, label="USER_DEFINED", ls="--", c="crimson")
    axes[0].legend()
    axes[0].set_xlabel("Energy [eV]")
    axes[0].set_ylabel("Spectrum [ ]")

    axes[1].plot(energy1, elastic1, label="GAUSSIAN", ls="-.", c="navy")
    axes[1].plot(
        ene_user_defined + beam_energy,
        sif_user_defined,
        label="USER_DEFINED",
        ls="--",
        c="crimson",
    )
    axes[1].set_xlabel("Energy [eV]")
    axes[1].set_ylabel("SIF [ ]")
    axes[1].set_xlim(beam_energy - 100, beam_energy + 100)
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def check_convolution():

    T = 30  # eV
    rho = 1.2  # g/cc
    Z = 1.5
    elements = np.array(["C", "C"])
    Zmin, Zmax, xmin, xmax = get_fractions_from_Z(Z)
    charge_states = np.array([Zmin, Zmax])
    partial_densities = np.array([xmin, xmax])

    models = ModelOptions()
    beam_energy = 8.5e3  # eV
    angle = 120
    q = calculate_q(angle=angle, energy=beam_energy)  # inverse aB
    omega_array = np.arange(-800, 1000, 10.0)  # eV

    output_file_name = os.path.join(os.path.dirname(__file__), f"ch_run_T={T:.0f}")

    kernel = xDave(
        models=models,
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_states=charge_states,
        elements=elements,
        partial_densities=partial_densities,
        user_defined_inputs=None,
        enforce_fsum=False,
        verbose=True,
        save_to_json=True,
        output_file_name=output_file_name,
    )

    bf_tot, ff_tot, dsf, WR, ff_i, bf_i = kernel.run(k=q, w=omega_array, mode="DYNAMIC")

    energy1, inelastic1, elastic1, spectrum1 = kernel.convolve_with_sif(
        omega=omega_array, bf=bf_tot, ff=ff_tot, dsf=dsf, Wr=WR, beam_energy=beam_energy, type="GAUSSIAN", fwhm=26
    )

    area_tot = np.trapezoid(spectrum1, energy1)
    area_inel = np.trapezoid(inelastic1, energy1)
    print(
        f"\nTotal integrated area: {area_tot}\n Total inelastic integrated area: {area_inel}\n Diff = {abs(area_tot - area_inel)}\n WR = {WR[0]}\n"
    )

    plt.figure()
    plt.plot(energy1, inelastic1, c="navy", label="Inel")
    plt.plot(energy1, elastic1, c="crimson", label="El")
    plt.plot(energy1, spectrum1, c="green", label="Tot")
    plt.legend()
    plt.show()


def test_full_spectrum():
    T = 80.0  # eV
    rho = 3.5  # g/cc
    ZC = 4.5  # also ZC=2.5, 3.5, 4.0, 4.5
    ZH = 1.0
    xH = 0.2
    angle = 120  # degrees
    beam_energy = 8.5e3  # eV
    q = calculate_q(angle=angle, energy=beam_energy)
    print(f"Running at q={q:.3f}")

    elements = np.array(["H", "C", "C"])

    Zmin, Zmax, xmin, xmax = get_fractions_from_Z_partial(ZC, x0=xH)
    partial_densities = np.array([xH, xmin, xmax])
    charge_states = np.array([ZH, Zmin, Zmax])
    user_defined_inputs = dict({"ipd": -10, "lfc": 0.0})

    models = ModelOptions(
        polarisation_model="DANDREA_FIT",
        bf_model="SCHUMACHER",
        lfc_model="NONE",
        ipd_model="STEWART_PYATT",
        screening_model="DEBYE_HUCKEL",
    )

    k = q  # 1/aB

    omega_array = np.arange(-4000, 4000, 1.0)  # eV

    output_file_name = os.path.join(os.path.dirname(__file__), f"ch_run_T={T:.0f}")

    kernel = xDave(
        models=models,
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_states=charge_states,
        elements=elements,
        partial_densities=partial_densities,
        user_defined_inputs=None,
        enforce_fsum=False,
        verbose=True,
        save_to_json=True,
        output_file_name=output_file_name,
    )

    bf_tot, ff_tot, dsf, WR, ff_i, bf_i = kernel.run(k=k, w=omega_array, mode="DYNAMIC")

    # data = kernel.load_result_from_json(fname=kernel.output_file_name)
    # print(data["setup"]["user_defined_inputs"])
    # print(data["plasma_parameters"]["electron_temperature"])

    k = np.linspace(0.1, 15, 1000)
    k, Sab, _, WR, qs, fs, lfc = kernel.run(k=k, w=0.0, mode="STATIC")


def compare_full_spectrum_mcss():
    T = 155.5  # eV
    rho = 30.0  # g/cc
    Z = 3.5  #
    angle = 75  # degrees, also can run 120
    beam_energy = 9.0e3  # eV
    q = calculate_q(angle=angle, energy=beam_energy)
    print(f"Running at q={q:.3f}")

    fn = os.path.join(
        os.path.dirname(__file__),
        f"comparison_data/mcss_comparisons/be_runs_T={T:.2f}_rho={rho:.2f}/mcss_run_be_T={T:.2f}_rho={rho:.2f}_Z={Z}_angle={angle:.0f}_full.csv",
    )
    status_fn = os.path.join(
        os.path.dirname(__file__),
        f"comparison_data/mcss_comparisons/be_runs_T={T:.2f}_rho={rho:.2f}/mcss_run_be_T={T:.2f}_rho={rho:.2f}_Z={Z}_angle={angle:.0f}_full_status.txt",
    )
    En_mcss, wff_mcss, wbf_mcss, ff_mcss, bf_mcss, el_mcss = load_mcss_result(filename=fn)
    WR_mcss = get_mcss_wr_from_status_file(status_file=status_fn)

    mcss_norm = 1

    # hard-coded for Z=3.5
    elements = np.array(["Be", "Be"])
    partial_densities = np.array([0.5, 0.5])
    charge_states = np.array([3, 4])

    models = ModelOptions(polarisation_model="NUMERICAL", bf_model="SCHUMACHER", lfc_model="NONE", ipd_model="NONE")
    k = q  # 1/aB

    omega_array = np.linspace(-2000, 2000, 20000)  # eV
    kernel = xDave(
        models=models,
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        elements=elements,
        charge_states=charge_states,
        partial_densities=partial_densities,
        user_defined_inputs=None,
        enforce_fsum=False,
    )

    bf_tot, ff_tot, dsf, WR, ff_i, bf_i = kernel.run(k=k, w=omega_array)
    ff_tot[np.isnan(ff_tot)] = 0.0

    print(f"MCSS WR = {WR_mcss}\nxDave WR = {WR}")

    # plot results
    fig, axes = plt.subplots(1, 3, figsize=(16, 16))

    ax = axes[0]
    ax.set_title("Total DSF")
    ax.plot(omega_array, dsf, label="Inel", ls="-.", c="magenta")
    ax.plot(En_mcss, (wbf_mcss + wff_mcss) / mcss_norm, ls=":", c="purple", label="MCSS / AN")
    ax.legend()
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.set_ylabel(r"DSF [1/eV]")

    ax = axes[1]
    ax.set_title("FF DSF")
    ax.plot(omega_array, ff_tot, label="FF", ls="--", c="orange")
    ax.plot(En_mcss, wff_mcss / mcss_norm, c="navy", ls=":", label="MCSS: ff")
    ax.legend()
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.set_ylabel(r"DSF [1/eV]")

    ax = axes[2]
    ax.set_title("BF DSF")
    ax.plot(omega_array, bf_tot, label="BF", ls="solid", c="dodgerblue")
    ax.plot(En_mcss, wbf_mcss / mcss_norm, c="brown", ls=":", label="MCSS: bf")
    ax.legend()
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.set_ylabel(r"DSF [1/eV]")

    plt.show()
    plt.close()

    spec_energy, inelastic, elastic, spectrum = kernel.convolve_with_sif(
        omega=omega_array, bf=bf_tot, ff=ff_tot, dsf=dsf, Wr=WR_mcss, beam_energy=beam_energy, type="GAUSSIAN", fwhm=1
    )

    fig2, axes = plt.subplots(1, 2, figsize=(14, 14))
    ax = axes[0]
    ax.plot(En_mcss + beam_energy, ff_mcss + bf_mcss, label="MCSS: inel", ls="-.", c="orange")
    ax.plot(spec_energy, inelastic, label="xDave: inel", ls=":", c="crimson")
    ax.plot(En_mcss + beam_energy, el_mcss, label="MCSS: el", ls="-.", c="dodgerblue")
    ax.plot(spec_energy, elastic, label="xDave: el", ls=":", c="navy")
    ax.legend()
    ax.set_xlabel(r"$E$ [eV]")
    ax.set_ylabel(r"I [arb. u.]")

    diff1 = inelastic - np.interp(x=spec_energy, xp=En_mcss + beam_energy, fp=ff_mcss + bf_mcss)
    diff1 = np.abs(diff1)
    diff2 = elastic - np.interp(x=spec_energy, xp=En_mcss + beam_energy, fp=el_mcss)
    diff2 = np.abs(diff2)
    ax = axes[1]
    ax.plot(spec_energy, diff1, label="Diff: inel", c="crimson", ls="solid")
    ax.plot(spec_energy, diff2, label="Diff: el", c="navy", ls="solid")
    ax.legend()
    ax.set_xlabel(r"$E$ [eV]")
    ax.set_ylabel(r"Abs. diff.")

    plt.show()
    plt.close()


if __name__ == "__main__":
    # test_sif()
    # test_full_spectrum()
    # check_convolution()
    compare_full_spectrum_mcss()

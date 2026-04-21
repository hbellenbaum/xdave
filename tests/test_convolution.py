from xdave import *
from xdave.plasma_state import get_fractions_from_Z
from xdave.utils import calculate_q

import numpy as np
import matplotlib.pyplot as plt
import os


def test_analytic_convolution():

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
    diff = abs(area_tot) - abs(area_inel)
    assert np.isclose(
        diff, WR, atol=1.0e-6
    ), f"Analytic convolution does not seem to work. Difference between the inelastic and total area under the spectrum is not equal to the Rayleigh weight: {diff - WR[0]}"


def test_user_defined_convolution():
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
    diff = abs(area_tot) - abs(area_inel)
    assert np.isclose(
        diff, WR, atol=1.0e-6
    ), f"User-defined convolution does not seem to work. Difference between the inelastic and total area under the spectrum is not equal to the Rayleigh weight: {diff - WR[0]}"

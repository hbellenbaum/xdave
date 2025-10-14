r"""
Quick interface to run Spectrum produced here with the x-ray tracing code HEART.

For more information, see: https://gitlab.com/heart-ray-tracing/HEART
"""

import sys

sys.path.insert(1, "/home/bellen85/code/dev/xdave/xdave")

from HEART import Spectrometer
from HEART.standards.standard_spectrometers import *
from xdave import *

import matplotlib.pyplot as plt


def create_xdave_spectrum(plot=False):
    plt.style.use("~/Desktop/resources/plotting/poster.mplstyle")

    T = 155.5
    rho = 30.0
    Z = 2.5
    # q = 4.0
    angle = 60
    beam_energy = 9.0e3
    q = calculate_q(angle=angle, energy=beam_energy)
    print(f"Running at q={q:.3f}")

    elements = np.array(["Be", "Be"])
    rho *= g_per_cm3_TO_kg_per_m3
    T *= eV_TO_K

    Z_min, Z_max, frac_min, frac_max = get_fractions_from_Z(Z=Z)

    partial_densities = np.array([frac_min, frac_max])
    charge_states = np.array([Z_min, Z_max])

    models = ModelOptions(polarisation_model="NUMERICAL", bf_model="SCHUMACHER", lfc_model="NONE", ipd_model="NONE")
    k = q / BOHR_RADIUS

    omega_array = np.arange(-4000, 4000, 0.5) * eV_TO_J

    kernel = xDave(
        models=models,
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        elements=elements,
        charge_states=charge_states,
        partial_densities=partial_densities,
        user_defined_inputs=None,
    )

    bf_tot, ff_tot, dsf, WR, ff_i, bf_i = kernel.run(k=k, w=omega_array)
    ff_tot[np.isnan(ff_tot)] = 0.0  # for now
    print(f"Calculated Rayleigh weight = {WR}")

    WR *= J_TO_eV

    sif_fwhm = 10 * eV_TO_J
    P_inelastic, P_elastic, spectrum = kernel.convolve_with_sif(
        omega=omega_array, bf=bf_tot, ff=ff_tot, WR=WR, type="GAUSSIAN", fwhm=sif_fwhm
    )

    if plot:
        plt.figure(figsize=(8, 6))
        plt.semilogy(omega_array * J_TO_eV, P_inelastic / np.max(spectrum), label="inel", ls="-.", c="dodgerblue")
        plt.semilogy(omega_array * J_TO_eV, P_elastic / np.max(spectrum), label="el", ls="-.", c="magenta")
        plt.semilogy(omega_array * J_TO_eV, spectrum / np.max(spectrum), label="tot", ls="-.", c="orange")
        plt.legend()
        plt.ylim(1.0e-10, 1.5)
        plt.show()

    return omega_array, P_inelastic, P_elastic, spectrum


def test():
    ## everything is in keV or mm
    Eb = 8  # keV

    output_dir = f"/home/bellen85/code/dev/xdave/heart_outputs"
    fname = "test_heart_output"

    energy, P_inel, P_el, intensity = create_xdave_spectrum()
    photon_energy_keV = energy * J_TO_eV * 1.0e-3 + Eb

    ## this is where the spectrum will go...
    # photon_energies = np.array([Eb])
    # intensity = np.array([1.0])
    spect = Spectrometer(output_dir=output_dir, output_file_name=fname, silence=False)
    setup_EuXFEL_vonHamos(spectrometer=spect, central_energy_keV=Eb)
    spect.add_source_profile(photon_energies_keV=photon_energy_keV, intensity=intensity)
    spect.ray_trace(Nphotons=1.0e7)
    im = spect.detector.detector_image


if __name__ == "__main__":
    test()
    # create_xdave_spectrum()

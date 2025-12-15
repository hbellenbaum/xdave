from xdave.constants import *
from xdave.utils import calculate_angle, calculate_q
from xdave.unit_conversions import *
from xdave.plasma_state import PlasmaState, get_rho_T_from_rs_theta
from xdave.models import ModelOptions
from xdave.freefree_dsf import FreeFreeDSF
from xdave.boundfree_dsf import BoundFreeDSF
from xdave.ii_ff import PaulingShermanIonicFormFactor
from xdave.lfc import LFC
from xdave.potentials import *
from xdave import *
from xdave.screening_cloud import ScreeningCloud
from xdave.constants import BOHR_RADIUS
from xdave.static_sf import OCPStaticStructureFactor, MCPStaticStructureFactor


import numpy as np
import matplotlib.pyplot as plt

import os


def update_ff_results(model, w, k_bohr, dsf, fn):
    file = fn + f"_k={k_bohr:.1f}_model={model}.csv"
    np.savetxt(file, np.array([w, dsf]).T)
    print(f"Updating FF test for model = {model}: \nfile = {file}")


def test_ff_version():
    print(f"\n Testing FF")
    Te = 50 * eV_TO_K
    rho = 10.0 * g_per_cm3_TO_kg_per_m3
    atomic_number = 4
    atomic_mass = 1.0
    beam_energy = 9.0e3
    charge_state = 3.0

    ZA = atomic_number
    Zb = atomic_number - charge_state

    angles = np.array([13, 30, 45, 60, 80, 100, 120, 140, 160])
    ks = calculate_q(angle=angles, energy=beam_energy) / BOHR_RADIUS
    omega_array = np.linspace(-450, 800, 10) * eV_TO_J
    binding_energies = np.array([-111.5, -111.5, -111.5]) * eV_TO_J
    state = PlasmaState(
        electron_temperature=Te,
        ion_temperature=Te,
        mass_density=rho,
        charge_state=charge_state,
        atomic_mass=atomic_mass,
        atomic_number=atomic_number,
        binding_energies=binding_energies,
    )

    output_dir = os.path.join(os.path.dirname(__file__), "xdave_results/ff/")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    fn = output_dir + f"dsf_check_be_T={Te/eV_TO_K:.0f}_rho={rho/g_per_cm3_TO_kg_per_m3:.0f}_Z={charge_state}"

    for i in range(len(ks)):
        k = ks[i]
        k_bohr = k * BOHR_RADIUS
        angle = np.round(calculate_angle(q=k * BOHR_RADIUS, energy=beam_energy))  # angles[i]
        kernel = FreeFreeDSF(state=state)
        dsf_lindhard = kernel.get_dsf(k=k, w=omega_array, lfc=0.0, model="LINDHARD")
        # update_ff_results(model="LINDHARD", w=omega_array, k_bohr=k_bohr, dsf=dsf_lindhard, fn=fn)
        dsf_lindhard_save = np.genfromtxt(fn + f"_k={k_bohr:.1f}_model=LINDHARD.csv", delimiter=" ")

        dsf_rpa = kernel.get_dsf(k=k, w=omega_array, lfc=0.0, model="NUMERICAL")
        # update_ff_results(model="RPA", w=omega_array, k_bohr=k_bohr, dsf=dsf_rpa, fn=fn)
        dsf_rpa_save = np.genfromtxt(fn + f"_k={k_bohr:.1f}_model=RPA.csv", delimiter=" ")

        dsf_fit = kernel.get_dsf(k=k, w=omega_array, lfc=0.0, model="DANDREA_FIT")
        # update_ff_results(model="Fit", w=omega_array, k_bohr=k_bohr, dsf=dsf_fit, fn=fn)
        dsf_fit_save = np.genfromtxt(fn + f"_k={k_bohr:.1f}_model=Fit.csv", delimiter=" ")

        # plt.figure()
        # plt.scatter(omega_array, dsf_fit, c="red")
        # plt.scatter(dsf_fit_save[:, 0], dsf_fit_save[:, 1], c="orange")
        # plt.show()

        if False:
            # I am currently ignoring Mermin because it is not working yet
            dsf_mermin = kernel.get_dsf(k=k, w=omega_array, lfc=0.0, model="MERMIN")
            update_ff_results(model="MERMIN", w=omega_array, k_bohr=k_bohr, dsf=dsf_lindhard, fn=fn)
            dsf_mermin_save = np.genfromtxt(fn + f"_k={k_bohr:.1f}_model=MERMIN.csv", delimiter=" ")

        rtol = 1.0e-4

        assert np.isclose(dsf_lindhard, dsf_lindhard_save[:, 1], rtol=rtol).all(
            axis=-1
        ), f"Lindhard FF model has failed the test at k={k_bohr:.1f} 1/aB."
        assert np.isclose(dsf_rpa, dsf_rpa_save[:, 1], rtol=rtol).all(
            axis=-1
        ), f"Numerical RPA FF model has failed the test at k={k_bohr:.1f} 1/aB."
        assert np.isclose(dsf_fit, dsf_fit_save[:, 1], rtol=rtol).all(
            axis=-1
        ), f"Dandrea RPA Fit FF model has failed the test at k={k_bohr:.1f} 1/aB."
        # if np.isclose(dsf_mermin, dsf_mermin_save[:, 1], rtol=1.0e-2).all(axis=-1):
        #     print(f"Mermin FF model has failed the test at k={k_bohr:.1f} 1/aB.")


def update_bf_results(model, w, k_bohr, dsf, fn):
    file = fn + f"_k={k_bohr:.1f}_model={model}.csv"
    np.savetxt(file, np.array([w, dsf]).T)
    print(f"Updating BF test for model = {model}: \nfile = {file}")


def test_bf_version():
    Te = 110 * eV_TO_K
    rho = 10.0 * g_per_cm3_TO_kg_per_m3
    atomic_number = 4
    atomic_mass = 1.0
    beam_energy = 9.0e3
    charge_state = 3

    ZA = atomic_number
    Zb = atomic_number - charge_state

    angles = np.array([13, 30, 45, 60, 80, 100, 120, 140, 160])
    ks = calculate_q(angle=angles, energy=beam_energy) / BOHR_RADIUS
    omega_array = np.linspace(-450, 800, 10) * eV_TO_J
    binding_energies = np.array([-111.5, -111.5, -111.5]) * eV_TO_J
    state = PlasmaState(
        electron_temperature=Te,
        ion_temperature=Te,
        mass_density=rho,
        charge_state=charge_state,
        atomic_mass=atomic_mass,
        atomic_number=atomic_number,
        binding_energies=binding_energies,
    )

    colors = ["red", "green", "blue", "orange", "gray", "black", "yellow", "magenta", "purple"]

    output_dir = os.path.join(os.path.dirname(__file__), "xdave_results/bf/")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    fn = output_dir + f"dsf_check_be_T={Te/eV_TO_K:.0f}_rho={rho/g_per_cm3_TO_kg_per_m3:.0f}_Z={charge_state}"

    for i in range(len(ks)):
        k = ks[i]
        k_bohr = k * BOHR_RADIUS
        angle = np.round(calculate_angle(q=k * BOHR_RADIUS, energy=beam_energy))  # angles[i]
        # print(f"Running for k={k * BOHR_RADIUS} 1/aB and angle={angle}")

        kernel = BoundFreeDSF(state=state)
        dsf = kernel.get_dsf(ZA=ZA, Zb=Zb, Eb=binding_energies, w=omega_array, k=k, model="SCHUMACHER")
        # update_bf_results(model="IA", w=omega_array, k_bohr=k_bohr, dsf=dsf, fn=fn)
        dsf_ia_save = np.genfromtxt(fn + f"_k={k_bohr:.1f}_model=IA.csv", delimiter=" ")
        dsf_hr = kernel.get_dsf(ZA=ZA, Zb=Zb, Eb=binding_energies, w=omega_array, k=k, model="HR_CORRECTION")
        # update_bf_results(model="HR", w=omega_array, k_bohr=k_bohr, dsf=dsf_hr, fn=fn)
        dsf_hr_save = np.genfromtxt(fn + f"_k={k_bohr:.1f}_model=HR.csv", delimiter=" ")
        dsf_tr = kernel.get_dsf(ZA=ZA, Zb=Zb, Eb=binding_energies, w=omega_array, k=k, model="TRUNCATED_IA")
        # update_bf_results(model="trIA", w=omega_array, k_bohr=k_bohr, dsf=dsf_tr, fn=fn)
        dsf_tr_save = np.genfromtxt(fn + f"_k={k_bohr:.1f}_model=trIA.csv", delimiter=" ")

        assert np.isclose(dsf, dsf_ia_save[:, 1], rtol=1.0e-2).all(
            axis=-1
        ), f"Impulse approximation BF model has failed the test at k={k_bohr:.1f} 1/aB."
        assert np.isclose(dsf_hr, dsf_hr_save[:, 1], rtol=1.0e-2).all(
            axis=-1
        ), f"Holm-Ribberfors correction to the IA BF model has failed the test at k={k_bohr:.1f} 1/aB."
        assert np.isclose(dsf_tr, dsf_tr_save[:, 1], rtol=1.0e-2).all(
            axis=-1
        ), f"truncated IA BF model has failed the test at k={k_bohr:.1f} 1/aB."


def update_ff_file(fn, ks, ff, element, Z_b):
    arr = np.array([ks, ff]).T
    file = fn + f"form_factor_{element}{Z_b}.txt"
    np.savetxt(file, arr, header="k ff")
    print(f"Updating form factor results: file = {file}")


def test_formfactor_version():

    ks = np.linspace(0.01, 10, 10) / BOHR_RADIUS

    ff_C3 = PaulingShermanIonicFormFactor().calculate_form_factor(Z=6, Z_b=3, k=ks)
    ff_B2 = PaulingShermanIonicFormFactor().calculate_form_factor(Z=4, Z_b=2, k=ks)
    ff_H0 = PaulingShermanIonicFormFactor().calculate_form_factor(Z=1, Z_b=1, k=ks)

    fn = os.path.join(os.path.dirname(__file__), "xdave_results/form_factors/")
    if not os.path.exists(fn):
        os.mkdir(fn)
    # update_ff_file(fn, ks, ff_H0, element="H", Z_b=1 - 1)
    # update_ff_file(fn, ks, ff_B2, element="B", Z_b=4 - 2)
    # update_ff_file(fn, ks, ff_C3, element="C", Z_b=6 - 3)
    res_H0 = np.genfromtxt(fn + f"form_factor_H0.txt", skip_header=1)
    res_B2 = np.genfromtxt(fn + f"form_factor_B2.txt", skip_header=1)
    res_C3 = np.genfromtxt(fn + f"form_factor_C3.txt", skip_header=1)

    assert np.isclose(ff_H0, res_H0[:, 1]).all(), f"Form factor model failed for H0."
    assert np.isclose(ff_B2, res_B2[:, 1]).all(), f"Form factor model failed for B2."
    assert np.isclose(ff_C3, res_C3[:, 1]).all(), f"Form factor model failed for C3."


def update_lfc_files(ks, fn, lfcs_dornheim, lfcs_interp, lfcs_ui, lfcs_gv, lfcs_farid):
    arr = np.array([ks, lfcs_dornheim, lfcs_interp, lfcs_ui, lfcs_gv, lfcs_farid]).T
    np.savetxt(fn, arr, header="ks Dornheim Interp UI GV Farid")
    print(f"Updating LFC file {fn}")


def test_lfc_version():
    AN = 1
    amu = 1

    Ts = np.linspace(10, 150, 5) * eV_TO_K
    rhos = np.linspace(1, 10, 5) * g_per_cm3_TO_kg_per_m3

    Zi = 1

    ks = np.linspace(0.01, 10, 10) / BOHR_RADIUS

    for T in Ts:

        for rho in rhos:
            state = PlasmaState(
                electron_temperature=T,
                ion_temperature=T,
                mass_density=rho,
                charge_state=Zi,
                atomic_mass=amu,
                atomic_number=AN,
                binding_energies=None,
            )
            kernel = LFC(state=state)
            lfcs_dornheim = kernel.calculate_lfc(k=ks, w=0.0, model="DORNHEIM_ESA")
            lfcs_interp = kernel.calculate_lfc(k=ks, w=0.0, model="PADE_INTERP")
            lfcs_ui = kernel.calculate_lfc(k=ks, w=0.0, model="UI")
            lfcs_gv = kernel.calculate_lfc(k=ks, w=0.0, model="GV")
            lfcs_farid = kernel.calculate_lfc(k=ks, w=0.0, model="FARID")

            output_dir = os.path.join(os.path.dirname(__file__), "xdave_results/lfc")
            # if not os.path.exists(output_dir):
            #     os.mkdir(output_dir)

            fn = os.path.join(output_dir, f"lfc_results_T={T/eV_TO_K:.0f}_rho={rho/g_per_cm3_TO_kg_per_m3:.1f}.csv")
            # update_lfc_files(ks, fn, lfcs_dornheim, lfcs_interp, lfcs_ui, lfcs_gv, lfcs_farid)
            res = np.genfromtxt(fn, delimiter=" ")

            assert np.isclose(
                lfcs_dornheim, res[:, 1]
            ).all(), f"Dornheim ESA has failed test for rho={rho/g_per_cm3_TO_kg_per_m3:.1f} and T={T/eV_TO_K:.1f}"
            assert np.isclose(
                lfcs_interp, res[:, 2]
            ).all(), f"Interp LFC has failed test for rho={rho/g_per_cm3_TO_kg_per_m3:.1f} and T={T/eV_TO_K:.1f}"
            assert np.isclose(
                lfcs_ui, res[:, 3]
            ).all(), (
                f"Utsumi-Ichimaru LFC has failed test for rho={rho/g_per_cm3_TO_kg_per_m3:.1f} and T={T/eV_TO_K:.1f}"
            )

            assert np.isclose(
                lfcs_gv, res[:, 4]
            ).all(), (
                f"Geldart-Vosko LFC has failed test for rho={rho/g_per_cm3_TO_kg_per_m3:.1f} and T={T/eV_TO_K:.1f}"
            )

            assert np.isclose(
                lfcs_farid, res[:, 5]
            ).all(), f"Farid LFC has failed test for rho={rho/g_per_cm3_TO_kg_per_m3:.1f} and T={T/eV_TO_K:.1f}"


def update_ii_files_k(ks, coulomb, yukawa, dh, csd, kelbg, deutsch, fn):
    arr = np.array([ks, coulomb, yukawa, dh, csd, kelbg, deutsch]).T
    np.savetxt(fn, arr, header="k Coulomb Yukawa DH CSD Kelbg Deutsch")
    print(f"Updating ii potentials in k-space: file={fn}")


def update_ii_files_r(rs, coulomb, yukawa, dh, srr, csd, kelbg, deutsch, fn):
    arr = np.array([rs, coulomb, yukawa, dh, srr, csd, kelbg, deutsch]).T
    np.savetxt(fn, arr, header="r Coulomb Yukawa DH SRR CSD Kelbg Deutsch")
    print(f"Updating ii potentials in r-space: file={fn}")


def test_ii_version():
    """
    Comparing the ion-ion potentials against a previous version to track changes.
    """
    n = 8192
    r0 = 0.5e-1 * BOHR_RADIUS  # [m]
    rf = 1.0e2 * BOHR_RADIUS  # [m]
    dr = (rf - r0) / n
    dk = np.pi / (n * dr)  # [1/m] as it should be [1/m],
    kf = r0 + n * dk
    rs = np.linspace(r0, rf, n)  # [m]
    ks = np.linspace(r0, kf, n)  # [1/m]
    Ti = 2 * eV_TO_K
    rho = 6.2 * g_per_cm3_TO_kg_per_m3  # g/cc

    Zi = 3
    atomic_weight = 26.9815384 * amu_TO_kg
    ni = rho / atomic_weight
    Rii = (3 / (4 * np.pi * ni)) ** (1 / 3)

    kappa_e = 1.24 / BOHR_RADIUS
    gamma_ii = 0.2 / BOHR_RADIUS
    ion_core_radius = 1 * ang_TO_m
    sigma_ii = 3.5
    sec_power = 6
    alpha = 2 / Rii

    print(f"\nTesting potentials in r-space")

    test_dh = debye_huckel_r(Qa=Zi, Qb=Zi, r=rs, alpha=gamma_ii, kappa_e=kappa_e)
    test_coulomb = coulomb_r(Qa=Zi, Qb=Zi, r=rs)
    test_csd = charge_switching_debye_r(
        Qa=Zi,
        Qb=Zi,
        r=rs,
        csd_parameter_a=gamma_ii,
        csd_parameter_b=gamma_ii,
        csd_core_charge_a=13,
        csd_core_charge_b=13,
        kappa_e=kappa_e,
    )
    test_srr = short_range_screening_r(
        Qa=Zi,
        Qb=Zi,
        r=rs,
        Ti=Ti,
        alpha=alpha,
        srr_core_power=sec_power,
        ion_core_radius=ion_core_radius,
        srr_sigma=sigma_ii,
        kappa_e=kappa_e,
    )
    test_yukawa = yukawa_r(Qa=Zi, Qb=Zi, r=rs, alpha=alpha)
    test_kelbg = kelbg_r(Qa=Zi, Qb=Zi, r=rs, alpha=alpha)
    test_deutsch = deutsch_r(Qa=Zi, Qb=Zi, r=rs, alpha=alpha)

    output_dir = os.path.join(os.path.dirname(__file__), "xdave_results/potentials")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    fn = os.path.join(
        output_dir,
        f"ii_potentials_r-space_results_T={Ti/eV_TO_K:.0f}_rho={rho/g_per_cm3_TO_kg_per_m3:.0f}_Zi={Zi}.csv",
    )
    # update_ii_files_r(rs, test_coulomb, test_yukawa, test_dh, test_srr, test_csd, test_kelbg, test_deutsch, fn)
    res = np.genfromtxt(fn, delimiter=" ", skip_header=1)

    assert np.isclose(test_coulomb, res[:, 1]).all(), f"Coulomb model has failed test."
    assert np.isclose(test_yukawa, res[:, 2]).all(), f"Yukawa model has failed test."
    assert np.isclose(test_dh, res[:, 3]).all(), f"Debye-Huckel model has failed test."
    assert np.isclose(test_srr, res[:, 4]).all(), f"SRR model has failed test."
    assert np.isclose(test_csd, res[:, 5]).all(), f"CSD model has failed test."
    assert np.isclose(test_kelbg, res[:, 6]).all(), f"Kelbg model has failed test."
    assert np.isclose(test_deutsch, res[:, 7]).all(), f"Deutsch model has failed test."

    print(f"\nTesting potentials in k-space")
    test_dh = debye_huckel_k(Qa=Zi, Qb=Zi, k=ks, alpha=gamma_ii, kappa_e=kappa_e)
    test_coulomb = coulomb_k(Qa=Zi, Qb=Zi, k=ks)
    test_csd = charge_switching_debye_k(
        Qa=Zi,
        Qb=Zi,
        k=ks,
        alpha=alpha,
        csd_parameter_a=gamma_ii,
        csd_parameter_b=gamma_ii,
        csd_core_charge_a=13,
        csd_core_charge_b=13,
        kappa_e=kappa_e,
    )

    test_yukawa = yukawa_k(Qa=Zi, Qb=Zi, k=ks, alpha=alpha)
    test_kelbg = kelbg_k(Qa=Zi, Qb=Zi, k=ks, alpha=alpha)
    test_deutsch = deutsch_k(Qa=Zi, Qb=Zi, k=ks, alpha=alpha)

    output_dir = os.path.join(os.path.dirname(__file__), "xdave_results/potentials")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    fn = os.path.join(
        output_dir,
        f"ii_potentials_k-space_results_T={Ti/eV_TO_K:.0f}_rho={rho/g_per_cm3_TO_kg_per_m3:.0f}_Zi={Zi}.csv",
    )
    # update_ii_files_k(ks, test_coulomb, test_yukawa, test_dh, test_csd, test_kelbg, test_deutsch, fn)
    res = np.genfromtxt(fn, delimiter=" ", skip_header=1)

    assert np.isclose(test_coulomb, res[:, 1]).all(), f"Coulomb model has failed test."
    assert np.isclose(test_yukawa, res[:, 2]).all(), f"Yukawa model has failed test."
    assert np.isclose(test_dh, res[:, 3]).all(), f"Debye-Huckel model has failed test."
    assert np.isclose(test_csd, res[:, 4]).all(), f"CSD model has failed test."
    assert np.isclose(test_kelbg, res[:, 5]).all(), f"Kelbg model has failed test."
    assert np.isclose(test_deutsch, res[:, 6]).all(), f"Deutsch model has failed test."


def update_ei_files_k(ks, coulomb, yukawa, hc, sc, fn):
    arr = np.array([ks, coulomb, yukawa, hc, sc]).T
    np.savetxt(fn, arr, header="k Coulomb Yukawa Hard-core Soft-core")
    print(f"Updating ei potentials in k-space: file={fn}")


def update_ei_files_r(rs, coulomb, yukawa, fn):
    arr = np.array([rs, coulomb, yukawa]).T
    np.savetxt(fn, arr, header="r Coulomb Yukawa")
    print(f"Updating ei potentials in r-space: file={fn}")


def test_ei_version():
    """
    Comparing the electron-ion potentials against a previous version to track changes.
    """
    n = 8192
    r0 = 0.5e-1 * BOHR_RADIUS  # [m]
    rf = 1.0e2 * BOHR_RADIUS  # [m]
    dr = (rf - r0) / n
    dk = np.pi / (n * dr)  # [1/m] as it should be [1/m],
    kf = r0 + n * dk
    rs = np.linspace(r0, rf, n)  # [m]
    ks = np.linspace(r0, kf, n)  # [1/m]
    Ti = 2 * eV_TO_K
    rho = 6.2 * g_per_cm3_TO_kg_per_m3  # g/cc

    Zi = 3
    atomic_weight = 26.9815384 * amu_TO_kg
    ni = rho / atomic_weight
    Rii = (3 / (4 * np.pi * ni)) ** (1 / 3)

    kappa_e = 1.24 / BOHR_RADIUS
    gamma_ii = 0.2 / BOHR_RADIUS
    ion_core_radius = 1 * ang_TO_m
    sigma_ii = 3.5
    sec_power = 6
    alpha = 2 / Rii

    print(f"\nTesting potentials in r-space")

    test_coulomb = coulomb_r(Qa=Zi, Qb=Zi, r=rs)
    test_yukawa = yukawa_r(Qa=Zi, Qb=Zi, r=rs, alpha=alpha)

    output_dir = os.path.join(os.path.dirname(__file__), "xdave_results/potentials")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    fn = os.path.join(
        output_dir,
        f"ei_potentials_r-space_results_T={Ti/eV_TO_K:.0f}_rho={rho/g_per_cm3_TO_kg_per_m3:.0f}_Zi={Zi}.csv",
    )
    # update_ei_files_r(rs, test_coulomb, test_yukawa, fn)
    res = np.genfromtxt(fn, delimiter=" ", skip_header=1)

    assert np.isclose(test_coulomb, res[:, 1]).all(), f"Coulomb model has failed test."
    assert np.isclose(test_yukawa, res[:, 2]).all(), f"Yukawa model has failed test."

    print(f"\nTesting potentials in k-space")
    test_coulomb = coulomb_k(Qa=Zi, Qb=Zi, k=ks)
    test_yukawa = yukawa_k(Qa=Zi, Qb=Zi, k=ks, alpha=alpha)
    test_hc = hard_core_ei_k(Qa=Zi, Qb=-1, k=ks, sigma_c=sigma_ii)
    test_sc = soft_core_ei_k(Qa=Zi, k=ks, rcore=ion_core_radius, n=sec_power)

    output_dir = os.path.join(os.path.dirname(__file__), "xdave_results/potentials")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    fn = os.path.join(
        output_dir,
        f"ei_potentials_k-space_results_T={Ti/eV_TO_K:.0f}_rho={rho/g_per_cm3_TO_kg_per_m3:.0f}_Zi={Zi}.csv",
    )
    # update_ei_files_k(ks, test_coulomb, test_yukawa, test_hc, test_sc, fn)
    res = np.genfromtxt(fn, delimiter=" ", skip_header=1)

    assert np.isclose(test_coulomb, res[:, 1]).all(), f"Coulomb model has failed test."
    assert np.isclose(test_yukawa, res[:, 2]).all(), f"Yukawa model has failed test."
    assert np.isclose(test_sc, res[:, 4]).all(), f"Soft-core model has failed test."
    assert np.isclose(test_hc, res[:, 3]).all(), f"Hard-core model has failed test."


def update_screening_file(fn, element, ks, qs, model):
    file = fn + f"{element}_{model}.txt"
    np.savetxt(file, np.array([ks, qs]).T, header="k qs")
    print(f"Updating screening file for {element} for model: {model}")


def test_screening_cloud_version():
    ZH = 1
    ZC = 4
    T = 10.0  # eV
    rho = 5.84  # 5.84

    models = ModelOptions(
        polarisation_model="NUMERICAL",
        bf_model="SCHUMACHER",
        lfc_model="NONE",
        ipd_model="NONE",
        ee_potential="COULOMB",
        ei_potential="COULOMB",
        ii_potential="YUKAWA",
    )

    elements = np.array(["H", "C"])
    partial_densities = np.array([0.5, 0.5])
    charge_states = np.array([ZH, ZC])

    kernel = xDave(
        mass_density=rho,
        electron_temperature=T,
        ion_temperature=T,
        elements=elements,
        partial_densities=partial_densities,
        charge_states=charge_states,
        models=models,
    )

    ei_potential_model = models.ei_potential
    ee_potential_model = models.ee_potential

    k = np.linspace(1.0e-2 / BOHR_RADIUS, 10 / BOHR_RADIUS, 10)

    screening_H = ScreeningCloud(state=kernel.states[0], overlord_state=kernel.overlord_state)
    f_fws_H = screening_H.get_screening_cloud(
        k=k, screening_model="FINITE_WAVELENGTH", ei_potential=ei_potential_model, ee_potential=ee_potential_model
    )
    f_dh_H = screening_H.get_screening_cloud(
        k=k, screening_model="DEBYE_HUCKEL", ei_potential=ei_potential_model, ee_potential=ee_potential_model
    )

    screening_C = ScreeningCloud(state=kernel.states[1], overlord_state=kernel.overlord_state)
    f_fws_C = screening_C.get_screening_cloud(
        k=k, screening_model="FINITE_WAVELENGTH", ei_potential=ei_potential_model, ee_potential=ee_potential_model
    )
    f_dh_C = screening_C.get_screening_cloud(
        k=k, screening_model="DEBYE_HUCKEL", ei_potential=ei_potential_model, ee_potential=ee_potential_model
    )

    output_dir = os.path.join(os.path.dirname(__file__), "xdave_results/screening")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    fn = os.path.join(output_dir, f"CH_screening_test_T={T:.0f}_rho={rho:.0f}_")

    # update_screening_file(fn=fn, element="C", ks=k, qs=f_fws_C, model="fws")
    dat_C_FWS = np.genfromtxt(fn + f"C_fws.txt", skip_header=1)
    # update_screening_file(fn=fn, element="C", ks=k, qs=f_dh_C, model="dh")
    dat_C_DH = np.genfromtxt(fn + f"C_dh.txt", skip_header=1)
    # update_screening_file(fn=fn, element="H", ks=k, qs=f_fws_H, model="fws")
    dat_H_FWS = np.genfromtxt(fn + f"H_fws.txt", skip_header=1)
    # update_screening_file(fn=fn, element="H", ks=k, qs=f_dh_H, model="dh")
    dat_H_DH = np.genfromtxt(fn + f"H_dh.txt", skip_header=1)

    atol = 1.0e-4
    assert np.isclose(f_fws_C, dat_C_FWS[:, 1], atol=atol).all(), f"FW screening test failed for Carbon."
    assert np.isclose(f_dh_C, dat_C_DH[:, 1], atol=atol).all(), f"DH screening test failed for Carbon."
    assert np.isclose(f_fws_H, dat_H_FWS[:, 1], atol=atol).all(), f"FW screening test failed for Hydrogen."
    assert np.isclose(f_dh_H, dat_H_DH[:, 1], atol=atol).all(), f"DH screening test failed for Hydrogen."


def update_sf_file_ocp(fn, material, ks, S):
    arr = np.array([ks, S]).T
    file = fn + f"static_sf_{material}.txt"
    np.savetxt(file, arr, header="ks S")


def update_sf_file_mcp(fn, material, ks, S):

    arr = np.array([ks, S[0, 0], S[0, 1], S[1, 1]]).T
    file = fn + f"static_sf_{material}.txt"
    np.savetxt(file, arr, header="ks S11 S12 S22")


def test_sf_version():
    ni = 5.0e22 * per_cm3_TO_per_m3
    ZC = 2
    ZH = 1
    T = 8  # eV
    nC = 5.0e22 * per_cm3_TO_per_m3
    nH = nC

    mC = 12.011 * amu_TO_kg
    mH = 1.008 * amu_TO_kg

    rho_C = nC * mC * kg_per_m3_TO_g_per_cm3
    rho_CH = 1.2

    fn = os.path.join(os.path.dirname(__file__), "xdave_results/static_sf/")
    if not os.path.exists(fn):
        os.mkdir(fn)

    # set up kernel for CH case
    models = ModelOptions()
    elements = np.array(["C"])
    charge_states = np.array([ZC])
    partial_densities_C = np.array([1.0])
    code_kernel_C = xDave(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho_C,
        elements=elements,
        partial_densities=partial_densities_C,
        charge_states=charge_states,
        models=models,
    )

    mix_fraction = 0.8
    ks = np.linspace(0.5, 10, 1000) / BOHR_RADIUS
    sf_C = OCPStaticStructureFactor(
        state=code_kernel_C.states[0], mix_fraction=mix_fraction, max_iterations=15000, delta=1.0e-12, verbose=False
    )
    Sab_C = sf_C.get_ii_static_structure_factor(k=ks, sf_model="HNC", pseudo_potential="DEBYE_HUCKEL")
    # update_sf_file_ocp(fn=fn, material="C", ks=ks, S=Sab_C)
    Sab_C_save = np.genfromtxt(fn + "static_sf_C.txt", delimiter=" ", skip_header=1)

    # multi-component
    # set up kernel for CH case
    models = ModelOptions()
    elements = np.array(["H", "C"])
    charge_states = np.array([ZH, ZC])
    partial_densities_CH = np.array([0.5, 0.5])
    code_kernel_CH = xDave(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho_CH,
        elements=elements,
        partial_densities=partial_densities_CH,
        charge_states=charge_states,
        models=models,
    )
    mix_fraction = 0.9
    sf_CH = MCPStaticStructureFactor(
        overlord_state=code_kernel_CH.overlord_state,
        states=code_kernel_CH.states,
        mix_fraction=mix_fraction,
        max_iterations=15000,
        delta=1.0e-10,
        verbose=False,
    )
    Sab_CH = sf_CH.get_ab_static_structure_factor(k=ks, sf_model="HNC", pseudo_potential="DEBYE_HUCKEL")
    # update_sf_file_mcp(fn=fn, material="CH", ks=ks, S=Sab_CH)
    Sab_CH_save = np.genfromtxt(fn + "static_sf_CH.txt", delimiter=" ", skip_header=1)

    rtol = 1.0e-3
    assert np.isclose(Sab_C, Sab_C_save[:, 1], rtol=rtol).all(), f"OCP structure factor for Carbon failed."
    assert np.isclose(
        Sab_CH[0, 0], Sab_CH_save[:, 1], rtol=rtol
    ).all(), f"OCP structure factor for CH failed for HH component."
    assert np.isclose(
        Sab_CH[0, 1], Sab_CH_save[:, 2], rtol=rtol
    ).all(), f"OCP structure factor for CH failed for CH component."
    assert np.isclose(
        Sab_CH[1, 1], Sab_CH_save[:, 3], rtol=rtol
    ).all(), f"OCP structure factor for CH failed for CC component."

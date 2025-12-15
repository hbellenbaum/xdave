from xdave.unit_conversions import ang_TO_m, eV_TO_K, g_per_cm3_TO_kg_per_m3, amu_TO_kg, per_cm3_TO_per_m3, J_TO_Ryd
from xdave.constants import BOLTZMANN_CONSTANT, BOHR_RADIUS, DIRAC_CONSTANT
from xdave.potentials import *

import numpy as np
import matplotlib.pyplot as plt
import os


THIS_DIR = os.path.dirname(__file__)


def test_ii_potentials():
    # Warm dense aluminium
    # DH: kappa_e = 1.24 1/aB
    # CSD: gamma_ii = 0.2 1/aB
    # SRR: ri_core = 1 1/Ang
    # SRR: sigma_ii = 3.5
    # SRR: n = 6

    n = 8192
    r0 = 0.5e-1 * BOHR_RADIUS  # [m]
    rf = 1.0e2 * BOHR_RADIUS  # [m]
    dr = (rf - r0) / n
    dk = np.pi / (n * dr)  # [1/m] as it should be [1/m],
    kf = r0 + n * dk
    rs = np.linspace(r0, rf, n)  # [m]
    ks = np.linspace(r0, kf, n)  # [1/m]
    Ti = 1.75 * eV_TO_K
    rho = 6.26 * g_per_cm3_TO_kg_per_m3  # g/cc

    Zi = 3
    Zf = 3
    atomic_weight = 26.9815384 * amu_TO_kg
    ni = rho / atomic_weight
    Rii = (3 / (4 * np.pi * ni)) ** (1 / 3)

    beta = 1 / (BOLTZMANN_CONSTANT * Ti)

    kappa_e = 1.24 / BOHR_RADIUS
    gamma_ii = 0.2 / BOHR_RADIUS
    ion_core_radius = 1 * ang_TO_m
    sigma_ii = 3.5
    sec_power = 6
    alpha = 2 / Rii

    test_dh = debye_huckel_r(Qa=Zi, Qb=Zi, r=rs, alpha=gamma_ii, kappa_e=kappa_e)
    test_coulomb_Z1 = coulomb_r(Qa=3, Qb=3, r=rs)
    test_coulomb_Z2 = coulomb_r(Qa=13, Qb=13, r=rs)
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
    print(beta * test_srr)
    test_yukawa = yukawa_r(Qa=Zi, Qb=Zi, r=rs, alpha=alpha)
    test_kelbg = kelbg_r(Qa=Zi, Qb=Zi, r=rs, alpha=alpha)
    test_deutsch = deutsch_r(Qa=Zi, Qb=Zi, r=rs, alpha=alpha)

    test_dir = "comparison_data/potentials/mcss_manual"
    dat_coulomb_Z1 = np.genfromtxt(os.path.join(THIS_DIR, test_dir, f"Fig15a_Coulomb_Zi_3.csv"), delimiter=",")
    dat_coulomb_Z2 = np.genfromtxt(os.path.join(THIS_DIR, test_dir, f"Fig15a_Coulomb_Zi_13.csv"), delimiter=",")
    dat_csd = np.genfromtxt(os.path.join(THIS_DIR, test_dir, f"Fig15a_CSD.csv"), delimiter=",")
    dat_dh = np.genfromtxt(os.path.join(THIS_DIR, test_dir, f"Fig15a_DH.csv"), delimiter=",")
    dat_srr = np.genfromtxt(os.path.join(THIS_DIR, test_dir, f"Fig15a_SRR.csv"), delimiter=",")

    plt.figure()
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(dat_coulomb_Z1[:, 0], dat_coulomb_Z1[:, 1], label="Coulomb Zi=3", marker="x", color="red")
    plt.scatter(dat_coulomb_Z2[:, 0], dat_coulomb_Z2[:, 1], label="Coulomb Zi=13", marker="o", color="red")
    plt.scatter(dat_csd[:, 0], dat_csd[:, 1], label="CSD", marker="*", color="orange")
    plt.scatter(dat_dh[:, 0], dat_dh[:, 1], label="DH", marker="*", color="blue")
    plt.scatter(dat_srr[:, 0], dat_srr[:, 1], label="SRR", marker="*", color="purple")

    plt.plot(rs / BOHR_RADIUS, beta * test_coulomb_Z1, label="Coulomb Zi=3", color="red", ls="solid")
    plt.plot(rs / BOHR_RADIUS, beta * test_coulomb_Z2, label="Coulomb Zi=13", color="red", ls="-.")
    plt.plot(rs / BOHR_RADIUS, beta * test_dh, label="DH", color="blue", ls="-.")
    plt.plot(rs / BOHR_RADIUS, beta * test_csd, label="CSD", color="orange", ls="-.")
    plt.plot(rs / BOHR_RADIUS, beta * test_srr, label="SRR", color="purple", ls="-.")
    plt.plot(rs / BOHR_RADIUS, beta * test_deutsch, label="Deutsch", color="darkgreen", ls="-.")
    plt.plot(rs / BOHR_RADIUS, beta * test_yukawa, label="Yukawa", color="black", ls="-.")
    plt.plot(rs / BOHR_RADIUS, beta * test_kelbg, label="Kelbg", color="magenta", ls="-.")

    plt.xlim(1.0e-2, 1.0e2)
    plt.ylim(1.0e-3, 1.0e6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compare_ei_potentials():

    T = 1.0e5  # K
    ni = 1.23e23  # cm^{-3}
    Zi = 2
    ni *= per_cm3_TO_per_m3
    Rii = (3 / (4 * np.pi * ni)) ** (1 / 3)
    alpha = 2 / Rii
    mi = 2 * amu_TO_kg
    lambda_ei = DIRAC_CONSTANT / np.sqrt(BOLTZMANN_CONSTANT * T * mi)

    n = 8192
    r0 = 0.5e-1 * BOHR_RADIUS  # [m]
    rf = 1.0e2 * BOHR_RADIUS  # [m]
    dr = (rf - r0) / n
    dk = np.pi / (n * dr)  # [1/m] as it should be [1/m],
    kf = r0 + n * dk
    rs = np.linspace(r0, rf, n)  # [m]
    ks = np.linspace(r0, kf, n)  # [1/m]

    coulomb_ei = ei_coulomb_r(Qa=Zi, r=rs) * J_TO_Ryd
    yukawa_ei = ei_yukawa_r(Qa=Zi, r=rs, alpha=alpha) * J_TO_Ryd
    deutsch_ei = deutsch_r(Qa=Zi, Qb=-1, r=rs, alpha=alpha) * J_TO_Ryd
    kelbg_ei = kelbg_r(Qa=Zi, Qb=-1, r=rs, alpha=alpha) * J_TO_Ryd
    kk_ei = klimontovich_kraeft_r(Qa=Zi, r=rs, T=T, lambda_ei=lambda_ei) * J_TO_Ryd

    rs /= Rii

    plt.figure()
    plt.plot(rs, np.abs(coulomb_ei), c="black", ls="-.", label="xDave: Coulomb")
    plt.plot(rs, np.abs(yukawa_ei), c="blue", ls="-.", label="xDave: Yukawa")
    plt.plot(rs, np.abs(kelbg_ei), c="red", ls="-.", label="xDave: Kelbg")
    plt.plot(rs, np.abs(kk_ei), c="green", ls="-.", label="xDave: Klimontovich-Kraeft")
    plt.plot(rs, np.abs(deutsch_ei), c="orange", ls="-.", label="xDave: Deutsch")
    plt.legend()
    plt.xlim(0, 5)
    plt.ylim(0, 10)
    plt.ylabel(r"$U_{ei}(r)$ [ryd]")
    plt.xlabel(r"$r$ [$d_i$]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_ii_potentials()
    compare_ei_potentials()

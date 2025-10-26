import sys

sys.path.insert(1, "/home/bellen85/code/dev/xdave/xdave")

from unit_conversions import ang_TO_m, eV_TO_K, g_per_cm3_TO_kg_per_m3, amu_TO_kg
from constants import BOLTZMANN_CONSTANT, VACUUM_PERMITTIVITY, BOHR_RADIUS, ELEMENTARY_CHARGE
from plasma_state import PlasmaState
from static_sf import OCPStaticStructureFactor, MCPStaticStructureFactor

from potentials import *
from xdave import xDave
from models import ModelOptions

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

    # T = 1.0e5  # K
    # ni = 1.23e23 * per_cm3_TO_per_m3  # cm^-3
    Zi = 3
    Zf = 3
    atomic_weight = 26.9815384 * amu_TO_kg
    ni = rho / atomic_weight
    Rii = (3 / (4 * np.pi * ni)) ** (1 / 3)

    beta = 1 / (BOLTZMANN_CONSTANT * Ti)

    # Rii = (3 / (4 * np.pi * ni)) ** (1 / 3)
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


def test_ei_potentials():
    test_dir = "comparison_data/potentials/Wuensch_Thesis"
    dat_coulomb = np.genfromtxt(os.path.join(THIS_DIR, test_dir, f"Fig4-6_Coulomb.csv"), delimiter=",")
    dat_deutsch = np.genfromtxt(os.path.join(THIS_DIR, test_dir, f"Fig4-6_Deutsch.csv"), delimiter=",")
    dat_kelbg = np.genfromtxt(os.path.join(THIS_DIR, test_dir, f"Fig4-6_Kelbg.csv"), delimiter=",")
    dat_kk = np.genfromtxt(os.path.join(THIS_DIR, test_dir, f"Fig4-6_KK.csv"), delimiter=",")

    plt.figure()
    plt.scatter(dat_coulomb[:, 0], dat_coulomb[:, 1], marker="x", c="black", label="Coulomb")
    plt.scatter(dat_deutsch[:, 0], dat_deutsch[:, 1], marker="x", c="blue", label="Deutsch")
    plt.scatter(dat_kelbg[:, 0], dat_kelbg[:, 1], marker="x", c="red", label="Kelbg")
    plt.scatter(dat_kk[:, 0], dat_kk[:, 1], marker="x", c="green", label="KK")
    plt.legend()
    plt.ylabel(r"$U_{ei}(r)$ [ryd]")
    plt.xlabel(r"$r$ [$d_i$]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_ii_potentials()
    # test_ei_potentials()

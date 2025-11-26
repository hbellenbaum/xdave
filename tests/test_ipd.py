from xdave.unit_conversions import (
    eV_TO_K,
    kg_per_m3_TO_g_per_cm3,
    g_per_cm3_TO_kg_per_m3,
    J_TO_eV,
    K_TO_eV,
)
from xdave.ipd import get_ipd
from xdave import PlasmaState

import numpy as np
import matplotlib.pyplot as plt
import os


THIS_DIR = os.path.dirname(__file__)

# TODO(HB): find a better test!


def test_CrowleyFig1():
    """
    Compare IPD models to data extracted from Crowley et al., High Energy Density Phys. 13 (2014) Fig. 1
    Somewhat confusing test, since I'm not actually sure about the conditions.
    """

    element = "Al"
    AN = 13
    amu = 26.9815384  # * amu_TO_kg
    Te = 180 * eV_TO_K
    rho = 2.699 * g_per_cm3_TO_kg_per_m3

    Zis = np.linspace(1, AN, 50)

    ipds_sp = []
    ipds_ek = []
    ipds_crowley = []
    ipds_is = []

    for Zi in Zis:
        state = PlasmaState(
            electron_temperature=Te,
            ion_temperature=Te,
            mass_density=rho,
            charge_state=Zi,
            atomic_mass=amu,
            atomic_number=AN,
            binding_energies=None,
        )
        ipds_sp.append(get_ipd(state=state, model="STEWART_PYATT") * J_TO_eV)
        ipds_ek.append(get_ipd(state=state, model="ECKER_KROLL") * J_TO_eV)
        ipds_crowley.append(get_ipd(state=state, model="CROWLEY") * J_TO_eV)
        ipds_is.append(get_ipd(state=state, model="ION_SPHERE") * J_TO_eV)

    data_path = "tests/comparison_data/ipd/"
    sp_data = np.genfromtxt(data_path + f"Crowley2014_Fig1_SP.csv", delimiter=",")
    ek_data = np.genfromtxt(data_path + f"Crowley2014_Fig1_EK.csv", delimiter=",")
    crowley_data = np.genfromtxt(data_path + f"Crowley2014_Fig1_Crowley.csv", delimiter=",")
    exp_data = np.genfromtxt(data_path + f"Crowley2014_Fig1_Exp.csv", delimiter=",")

    plt.figure(figsize=(14, 10))
    plt.plot(Zis, ipds_sp, label="SP", ls="-.", c="navy")
    plt.scatter(AN - sp_data[:, 0], sp_data[:, 1], marker="x", c="navy", label=f"Crowley Fig 1: SP")
    plt.plot(Zis, ipds_ek, label="EK", ls="-.", c="crimson")
    plt.scatter(AN - ek_data[:, 0], ek_data[:, 1], marker="<", c="crimson", label=f"Crowley Fig 1: EK")
    plt.plot(Zis, ipds_crowley, label="Crowley", ls="-.", c="green")
    plt.scatter(AN - crowley_data[:, 0], crowley_data[:, 1], marker="*", c="green", label=f"Crowley Fig 1: Crowley")
    plt.plot(Zis, ipds_is, label="IS", ls="-.", c="orange")
    plt.scatter(AN - exp_data[:, 0], exp_data[:, 1], marker="o", c="black", label=f"Crowley Fig 1: Exp")
    plt.xlabel(r"$Z_i$")
    plt.ylabel(r"$\Delta_{IPD}$ [eV]")
    plt.title(f"Al IPD calculations: T={Te * K_TO_eV:.2f} eV, rho={rho * kg_per_m3_TO_g_per_cm3:.2f} g/cc")
    plt.legend()
    plt.show()


def compare_all():
    return


if __name__ == "__main__":
    compare_all()
    test_CrowleyFig1()

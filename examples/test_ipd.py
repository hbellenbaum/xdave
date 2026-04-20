from xdave.unit_conversions import (
    eV_TO_K,
    kg_per_m3_TO_g_per_cm3,
    g_per_cm3_TO_kg_per_m3,
    J_TO_eV,
    K_TO_eV,
)
from xdave.ipd import get_ipd
from xdave import PlasmaState, xDave, ModelOptions

import numpy as np
import matplotlib.pyplot as plt
import os


THIS_DIR = os.path.dirname(__file__)

# TODO(HB): find a better test!
# plt.style.use("~/Desktop/resources/plotting/poster.mplstyle")


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
        kernel = xDave(
            electron_temperature=Te,
            mass_density=rho,
            ion_temperature=Te,
            elements=np.array(["Al"]),
            partial_densities=np.array([1.0]),
            charge_states=np.array([Zi]),
            models=ModelOptions(),
        )
        ipds_sp.append(get_ipd(plasma=kernel, state=state, model="STEWART_PYATT") * J_TO_eV)
        ipds_ek.append(get_ipd(plasma=kernel, state=state, model="ECKER_KROLL") * J_TO_eV)
        ipds_crowley.append(get_ipd(plasma=kernel, state=state, model="CROWLEY") * J_TO_eV)
        ipds_is.append(get_ipd(plasma=kernel, state=state, model="ION_SPHERE") * J_TO_eV)

    data_path = "examples/comparison_data/ipd/"
    sp_data = np.genfromtxt(data_path + f"Crowley2014_Fig1_SP.csv", delimiter=",")
    ek_data = np.genfromtxt(data_path + f"Crowley2014_Fig1_EK.csv", delimiter=",")
    crowley_data = np.genfromtxt(data_path + f"Crowley2014_Fig1_Crowley.csv", delimiter=",")
    exp_data = np.genfromtxt(data_path + f"Crowley2014_Fig1_Exp.csv", delimiter=",")

    plt.figure(figsize=(14, 10))
    plt.plot(Zis, ipds_sp, label="SP", ls="-.", c="navy")
    plt.scatter(sp_data[:, 0], sp_data[:, 1], marker="x", c="navy", label=f"Crowley Fig 1: SP")
    plt.plot(Zis, ipds_ek, label="EK", ls="-.", c="crimson")
    plt.scatter(ek_data[:, 0], ek_data[:, 1], marker="<", c="crimson", label=f"Crowley Fig 1: EK")
    plt.plot(Zis, ipds_crowley, label="Crowley", ls="-.", c="green")
    plt.scatter(crowley_data[:, 0], crowley_data[:, 1], marker="*", c="green", label=f"Crowley Fig 1: Crowley")
    plt.plot(Zis, ipds_is, label="IS", ls="-.", c="orange")
    plt.scatter(exp_data[:, 0], exp_data[:, 1], marker="o", c="black", label=f"Crowley Fig 1: Exp")
    plt.xlabel(r"$Z_i$")
    plt.ylabel(r"$\Delta_{IPD}$ [eV]")
    plt.title(f"Al IPD calculations: T={Te * K_TO_eV:.2f} eV, rho={rho * kg_per_m3_TO_g_per_cm3:.2f} g/cc")
    plt.legend()
    plt.show()


def update_ipd_files(Zis, fn, ipd_sp, ipd_ek, ipd_crowley, ipd_is, ipd_dh):
    arr = np.array([Zis, ipd_sp, ipd_ek, ipd_crowley, ipd_is, ipd_dh]).T
    np.savetxt(fn, arr, header="Zi SP EK Crowley IS DH")
    print(f"Updating IPD file {fn}")


def test_version():
    AN = 13
    amu = 26.9815384  # * amu_TO_kg
    Te = 180 * eV_TO_K
    rho = 2 * 2.7 * g_per_cm3_TO_kg_per_m3

    Ts = np.linspace(10, 150, 10) * eV_TO_K

    Zis = np.linspace(1, AN, 50)

    for T in Ts:
        ipds_sp = []
        ipds_ek = []
        ipds_crowley = []
        ipds_is = []
        ipds_dh = []

        for Zi in Zis:
            state = PlasmaState(
                electron_temperature=T,
                ion_temperature=T,
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
            ipds_dh.append(get_ipd(state=state, model="DEBYE_HUCKEL") * J_TO_eV)

        output_dir = os.path.join(os.path.dirname(__file__), "xdave_results/ipd")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        fn = os.path.join(output_dir, f"IPD_results_Al_T={T/eV_TO_K:.0f}_rho={rho/g_per_cm3_TO_kg_per_m3:.0f}.csv")
        # update_ipd_files(Zis, fn, ipds_sp, ipds_ek, ipds_crowley, ipds_is, ipds_dh)
        res = np.genfromtxt(fn, skip_header=1, delimiter=" ")

        if not np.isclose(ipds_sp, res[:, 1]).all():
            print(f"Stewart-Pyatt model has failed test at T={T/eV_TO_K:.0f}")
        if not np.isclose(ipds_ek, res[:, 2]).all():
            print(f"Ecker-Kroell model has failed test at T={T/eV_TO_K:.0f}")
        if not np.isclose(ipds_crowley, res[:, 3]).all():
            print(f"Crowley model has failed test at T={T/eV_TO_K:.0f}")
        if not np.isclose(ipds_is, res[:, 4]).all():
            print(f"Ion sphere model has failed test at T={T/eV_TO_K:.0f}")
        if not np.isclose(ipds_dh, res[:, 5]).all():
            print(f"Debye-Hueckel model has failed test at T={T/eV_TO_K:.0f}")


if __name__ == "__main__":
    # compare_all()
    # test_version()
    test_CrowleyFig1()

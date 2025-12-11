from xdave.plasma_state import get_fractions_from_Z
from xdave.utils import calculate_angle

from xdave import *

import numpy as np
import matplotlib.pyplot as plt


def check_inelastic():
    T = 21.7  # eV
    rho = 2.2  # two times solid density [g/cc]
    Z_C = 4.5
    AN = 6
    Zb = AN - Z_C

    Zmin, Zmax, xmin, xmax = get_fractions_from_Z(Z=Z_C)
    elements = np.array(["C", "C"])  # np.array(["C"])  #
    charge_states = np.array([Zmin, Zmax])  # np.array([Z_C])  #
    partial_densities = np.array([xmin, xmax])  # np.array([1.0])  #

    models = ModelOptions(
        ei_potential="YUKAWA",
        ii_potential="YUKAWA",
        ee_potential="COULOMB",
        polarisation_model="NUMERICAL",
        sf_model="HNC",
        lfc_model="NONE",
        ipd_model="NONE",
        bf_model="SCHUMACHER",
        screening_model="FINITE_WAVELENGTH",
    )

    kernel = xDave(
        mass_density=rho,
        electron_temperature=T,
        ion_temperature=T,
        elements=elements,
        charge_states=charge_states,
        partial_densities=partial_densities,
        models=models,
        enforce_fsum=False,
        user_defined_inputs=None,
    )

    w = np.linspace(-500, 6000, 5000)

    beam_energy = 20.0e3

    ks = np.linspace(0.1, 15, 50)
    bf_statics = []
    ff_statics = []
    angles = []
    for i in range(0, len(ks)):
        bf_tot, ff_tot, dsf, rayleigh_weight, ff_i, bf_i = kernel.run(
            w=w, k=ks[i], beam_energy=beam_energy, mode="DYNAMIC"
        )
        angles.append(calculate_angle(q=ks[i], energy=beam_energy))

        bf_static, ff_static = kernel.get_static_structure_factors(w=w, bf=bf_tot, ff=ff_tot)
        bf_statics.append(bf_static)
        ff_statics.append(ff_static)

    # print(angles)

    plt.figure()
    plt.scatter(ks, bf_statics, marker="*", c="magenta", label="BF")
    plt.scatter(ks, ff_statics, marker="x", c="purple", label="FF")
    plt.axhline(kernel.overlord_state.charge_state, c="navy", ls="--", label=r"$Z$")
    plt.axhline(Zb, c="darkgreen", ls="--", label=r"$Z_b$")
    plt.axhline(kernel.overlord_state.atomic_number, c="crimson", ls="--", label="AN")
    plt.axhline(1.0, c="gray", ls="--", label="1.0")
    plt.xlabel(r"$k$ [$a_B^{-1}$]")
    plt.ylabel(r"SF")
    plt.title(f"enforce_fsum={kernel.enforce_fsum}")
    plt.legend()
    plt.show()


def check_elastic():
    T = 21.7  # eV
    rho = 2.2  # two times solid density [g/cc]
    Z_C = 3.0
    AN = 6
    Zb = AN - Z_C

    Zmin, Zmax, xmin, xmax = get_fractions_from_Z(Z=Z_C)
    elements = np.array(["C", "C"])  # np.array(["C"])  #
    charge_states = np.array([Zmin, Zmax])  # np.array([Z_C])  #
    partial_densities = np.array([xmin, xmax])  # np.array([1.0])  #

    models = ModelOptions(
        polarisation_model="NUMERICAL",
        ei_potential="YUKAWA",
        ii_potential="YUKAWA",
        bf_model="SCHUMACHER",
        lfc_model="NONE",
        ipd_model="NONE",
        screening_model="DEBYE_HUCKEL",
    )

    kernel = xDave(
        mass_density=rho,
        electron_temperature=T,
        ion_temperature=T,
        elements=elements,
        charge_states=charge_states,
        partial_densities=partial_densities,
        models=models,
        enforce_fsum=False,
        user_defined_inputs=None,
    )

    w = np.linspace(-500, 6000, 5000)

    beam_energy = 9.0e3
    ks = np.linspace(0.01, 10, 1000)
    angle = 75
    _, Sab, Sab_tot, rayleigh_weight, qs, fs, lfc = kernel.run(w=w, k=ks, beam_energy=beam_energy, mode="STATIC")

    fig, axes = plt.subplots(1, 4, figsize=(14, 10))
    ax = axes[0]
    ax.plot(ks, Sab[0, 0, :], label="xDave: 11", ls="-.", c="dodgerblue")
    ax.plot(ks, Sab[0, 1, :], label="xDave: 12", ls="-.", c="magenta")
    ax.plot(ks, Sab[1, 1, :], label="xDave: 22", ls="-.", c="limegreen")
    ax.plot(ks, Sab_tot, label="xDave: tot", ls="-.", c="black")
    ax.legend()
    ax = axes[1]
    ax.plot(ks, rayleigh_weight, label="xDave", ls="-.", c="lightgreen")
    ax.plot(ks, rayleigh_weight / kernel.overlord_state.charge_state, label="xDave / Zf", ls="-.", c="dodgerblue")
    ax.plot(ks, rayleigh_weight / kernel.overlord_state.atomic_number, label="xDave  AN", ls="-.", c="magenta")
    ax.legend()
    ax = axes[2]
    ax.plot(ks, qs[0], label=r"xDave $q_1$", ls="-.", c="lightgreen")
    ax.plot(ks, qs[1], label=r"xDave $q_2$", ls="-.", c="dodgerblue")
    ax.axhline(kernel.overlord_state.charge_state, c="gray", ls=":")
    ax.legend()
    ax = axes[3]
    ax.plot(ks, fs[0], label=r"xDave $f_1$", ls="-.", c="lightgreen")
    ax.plot(ks, fs[1], label=r"xDave $f_2$", ls="-.", c="dodgerblue")
    ax.axhline(kernel.overlord_state.charge_state, c="gray", ls=":")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    check_inelastic()
    check_elastic()

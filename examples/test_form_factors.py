from xdave.ii_ff import PaulingShermanIonicFormFactor
from xdave.unit_conversions import *
from xdave.constants import *


import numpy as np
import matplotlib.pyplot as plt
import os


def test_carbon_ff():
    file_paths = os.path.join(os.path.dirname(__file__), "comparison_data/form_factors/")
    c0 = np.loadtxt(file_paths + "fk_c0.txt", skiprows=2)
    c1 = np.loadtxt(file_paths + "fk_c1.txt", skiprows=2)
    c2 = np.loadtxt(file_paths + "fk_c2.txt", skiprows=2)
    c3 = np.loadtxt(file_paths + "fk_c3.txt", skiprows=2)
    c4 = np.loadtxt(file_paths + "fk_c4.txt", skiprows=2)
    c5 = np.loadtxt(file_paths + "fk_c5.txt", skiprows=2)

    AN = 6
    ks = np.linspace(0.01, 10, 100) / BOHR_RADIUS

    xdave_c0 = PaulingShermanIonicFormFactor().calculate_form_factor(Z=AN, Z_b=6, k=ks)
    xdave_c1 = PaulingShermanIonicFormFactor().calculate_form_factor(Z=AN, Z_b=5, k=ks)
    xdave_c2 = PaulingShermanIonicFormFactor().calculate_form_factor(Z=AN, Z_b=4, k=ks)
    xdave_c3 = PaulingShermanIonicFormFactor().calculate_form_factor(Z=AN, Z_b=3, k=ks)
    xdave_c4 = PaulingShermanIonicFormFactor().calculate_form_factor(Z=AN, Z_b=2, k=ks)
    xdave_c5 = PaulingShermanIonicFormFactor().calculate_form_factor(Z=AN, Z_b=1, k=ks)

    ks *= BOHR_RADIUS

    plt.figure()
    plt.plot(ks, xdave_c0, label="xDave: C0", ls="-.", c="navy")
    plt.plot(ks, xdave_c1, label="xDave: C1", ls="-.", c="dodgerblue")
    plt.plot(ks, xdave_c2, label="xDave: C2", ls="-.", c="lightgreen")
    plt.plot(ks, xdave_c3, label="xDave: C3", ls="-.", c="orange")
    plt.plot(ks, xdave_c4, label="xDave: C4", ls="-.", c="crimson")
    plt.plot(ks, xdave_c5, label="xDave: C5", ls="-.", c="magenta")
    plt.plot(c0[:, 0], c0[:, 1], label="Test data: C0", ls="--", c="navy", alpha=0.4)
    plt.plot(c1[:, 0], c1[:, 1], label="Test data: C1", ls="--", c="dodgerblue", alpha=0.4)
    plt.plot(c2[:, 0], c2[:, 1], label="Test data: C2", ls="--", c="lightgreen", alpha=0.4)
    plt.plot(c3[:, 0], c3[:, 1], label="Test data: C3", ls="--", c="orange", alpha=0.4)
    plt.plot(c4[:, 0], c4[:, 1], label="Test data: C4", ls="--", c="crimson", alpha=0.4)
    plt.plot(c5[:, 0], c5[:, 1], label="Test data: C5", ls="--", c="magenta", alpha=0.4)
    plt.legend()
    plt.xlabel(r"$k$ [$1/a_B$]")
    plt.ylabel("Form factor")
    plt.xlim(0, 10)
    plt.show()

    rtol = 1e-1

    if not np.isclose(xdave_c0, np.interp(ks, c0[:, 0], c0[:, 1]), rtol=rtol).all(axis=-1):
        print(f"Form factor test failed for C0.")
    if not np.isclose(xdave_c1, np.interp(ks, c1[:, 0], c1[:, 1]), rtol=rtol).all(axis=-1):
        print(f"Form factor test failed for C1.")
    if not np.isclose(xdave_c2, np.interp(ks, c2[:, 0], c2[:, 1]), rtol=rtol).all(axis=-1):
        print(f"Form factor test failed for C2.")
    if not np.isclose(xdave_c3, np.interp(ks, c3[:, 0], c3[:, 1]), rtol=rtol).all(axis=-1):
        print(f"Form factor test failed for C3.")
    if not np.isclose(xdave_c4, np.interp(ks, c4[:, 0], c4[:, 1]), rtol=rtol).all(axis=-1):
        print(f"Form factor test failed for C4.")
    if not np.isclose(xdave_c5, np.interp(ks, c5[:, 0], c5[:, 1]), rtol=rtol).all(axis=-1):
        print(f"Form factor test failed for C5.")


def update_ff_file(fn, ks, ff, element, Z_b):
    arr = np.array([ks, ff]).T
    file = fn + f"form_factor_{element}{Z_b}.txt"
    np.savetxt(file, arr, header="k ff")
    print(f"Updating form factor results: file = {file}")


def test_version():

    ks = np.linspace(0.01, 10, 100) / BOHR_RADIUS

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

    if not np.isclose(ff_H0, res_H0[:, 1]).all():
        print(f"Form factor model failed for H0.")
    if not np.isclose(ff_B2, res_B2[:, 1]).all():
        print(f"Form factor model failed for B2.")
    if not np.isclose(ff_C3, res_C3[:, 1]).all():
        print(f"Form factor model failed for C3.")


if __name__ == "__main__":
    test_carbon_ff()
    test_version()

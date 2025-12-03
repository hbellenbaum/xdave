from .constants import DIRAC_CONSTANT, SPEED_OF_LIGHT, BOHR_RADIUS, PI, ATOMIC_MASS_UNIT
from .unit_conversions import eV_TO_J

from scipy.fft import dst
import pandas as pd
import numpy as np

import math
import os

from collections import defaultdict

from importlib.resources import files


def read_mcss_output(filepath, start_line=0, end_line=96022):
    grouped_data = defaultdict(lambda: {"xnl": [], "Jnl": []})
    current_zb = None

    with open(filepath, "r") as f:
        lines = f.readlines()
        if end_line is not None:
            lines = lines[start_line:end_line]
        else:
            lines = lines[start_line:]

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 2:
            key, val = parts
            try:
                if key == "Zb":
                    current_zb = int(val)
                else:
                    val = float(val)
                    if key == "xnl" and current_zb is not None:
                        grouped_data[current_zb]["xnl"].append(val)
                    elif key == "Jnl" and current_zb is not None:
                        grouped_data[current_zb]["Jnl"].append(val)
            except ValueError:
                continue

    # build arrays for each Zb group
    results = {}
    for zb, values in grouped_data.items():
        pairs = list(zip(values["xnl"], values["Jnl"]))
        # pairs.sort(key=lambda x: x[0])
        results[zb] = np.array(pairs)

    return results


def calculate_q(angle, energy):
    # angle *= np.pi / 180.0
    angle_rad = angle * PI / 180
    E0 = energy * eV_TO_J
    q = 2 * E0 / (DIRAC_CONSTANT * SPEED_OF_LIGHT) * np.sin(angle_rad / 2)
    q *= BOHR_RADIUS
    return q


def calculate_q_SI(angle, energy):
    # angle *= np.pi / 180.0
    angle_rad = angle * PI / 180
    E0 = energy * eV_TO_J
    q = 2 * E0 / (DIRAC_CONSTANT * SPEED_OF_LIGHT) * np.sin(angle_rad / 2)
    q *= BOHR_RADIUS
    return q


def calculate_angle(q, energy):
    """
    Calculates an angle for the relevant q value.
    Input:
    - q: in Hartree units
    Returns:
    - angle: degrees
    """

    q_value = q / BOHR_RADIUS

    # convert energy from eV to J
    E0 = energy * eV_TO_J

    # small angle approximation: see Eqn. (9) in [2]
    K = DIRAC_CONSTANT * SPEED_OF_LIGHT * q_value / (2 * E0)
    angle = 2 * np.arcsin(K)

    # convert angle from radians to degrees
    angle *= 180 / np.pi

    # if np.isnan(angle):
    #     print(
    #         f"Attempted to calculate an angle, but either the wave number is too large or the beam energy is too small."
    #     )

    return angle


def load_mcss_result(filename):
    En, Es, lambda_s, wff, wbf, ff, bf, el, tot = np.genfromtxt(filename, skip_header=1, delimiter=",", unpack=True)
    return En[::-1], wff[::-1], wbf[::-1], ff, bf, el


def load_mcss_result_ar(filename, use_lfc_model=False):
    # Note: this only works for two-component systems
    if use_lfc_model:
        _, k, _, WR, f1, f2, q1, q2, S11, S12, S22, lfc = np.genfromtxt(
            filename, skip_header=1, delimiter=",", unpack=True
        )
    else:
        _, k, _, WR, f1, f2, q1, q2, S11, S12, S22 = np.genfromtxt(filename, skip_header=1, delimiter=",", unpack=True)
        lfc = np.zeros_like(k)
    return k, WR, f1, f2, q1, q2, S11, S12, S22, lfc


def load_mcss_result_ar_3species(filename, use_lfc_model=False):
    if use_lfc_model:
        _, k, _, WR, f1, f2, f3, q1, q2, q3, S11, S13, S12, S22, S23, S33, lfc = np.genfromtxt(
            filename, skip_header=1, delimiter=",", unpack=True
        )
    else:
        _, k, _, WR, f1, f2, f3, q1, q2, q3, S11, S13, S12, S22, S23, S33 = np.genfromtxt(
            filename, skip_header=1, delimiter=",", unpack=True
        )
        lfc = np.zeros_like(k)
    return k, WR, f1, f2, f3, q1, q2, q3, S11, S13, S12, S22, S23, S33, lfc


def get_mcss_wr_from_status_file(status_file):
    WR_message = "The calculated weight of the Rayleigh feature is:"
    # status_file = os.path.join(status_file)
    fr = open(status_file, "r")
    WR_mcss = None
    for line in fr.readlines():
        # print(line)
        if WR_message in line:
            WR_mcss = line.split(": ")[1]
    return float(WR_mcss)


def get_values_from_status_file(status_fn):

    return


HA_TO_eV = 27.211386  # hard-coded for now


def load_itcf_from_file(N, q_index=0, skiprows=0, data_path=None):
    """
    Load PIMC data, i.e. ITCF from Tobias' data set
    Note: no keys/ headers, all values are given in Hartree units

    Parameters
    ----------
    N: int
            number of particles in PIMC simulation to find correct file
    rs: float
            Wigner-Seitz radius of PIMC simulation
    theta: float
            non-dimensional temperature of PIMC simulation
    q_index: int
            index 0 to 100 for the chosen q-value
    skiprows: float
            default set to 0, don't change for the current file format
    data_path: str
            directory where data files are stored


    Returns
    ----------
    float
            q-value
    array
            array of tau values in 1/eV
    array
            array of ITCF values
    array
            array of statistical errors of the ITCF
    float
            proton-electron static structure factor corresponding to q
    float
            proton static structure factor corresponding to q
    float
            Rayleigh weight

    """

    path = data_path + f"/Tau_k_index{q_index}fermion_density_response_electron_{N}.res"
    data = np.loadtxt(path, skiprows=skiprows)

    # tau: first row in the ITCF file (for a given q) -> converted to 1/eV from 1/Hartree
    tau_array = data[:, 0] * 1 / HA_TO_eV
    # F: ITCF function for each tau (for a given q) -> directly compared to MCSS outputs
    itcf_array = data[:, 1]
    itcf_errors = data[:, 2]

    # hard-coded, taken from the corresponding lines (according to index) in the filename
    S_ei_from_file = np.loadtxt(
        os.path.join(
            data_path,
            f"fermion_proton_electron_static_structure_factor_{N}.res",
        ),
        usecols=1,
    )
    S_ei = S_ei_from_file[q_index]

    # load
    S_ii_from_file = np.loadtxt(
        os.path.join(data_path, f"fermion_proton_static_structure_factor_{N}.res"),
        usecols=1,
    )
    S_ii = S_ii_from_file[q_index]

    q_values = np.loadtxt(
        os.path.join(
            data_path,
            f"fermion_proton_electron_static_structure_factor_{N}.res",
        ),
        usecols=0,
    )
    q_value = q_values[q_index]

    # calculate Rayleigh weight
    WR = S_ei**2 / S_ii

    return q_value, tau_array, itcf_array, itcf_errors, S_ei, S_ii, WR


def laplace(tau, E, wff, wbf):
    """
    Laplace transform
    """

    F_wff = np.zeros(len(tau))
    F_wbf = np.zeros(len(tau))

    for i in range(0, len(tau)):

        kernel_wff = np.exp(-tau[i] * E) * wff  # * omega_factor
        kernel_wbf = np.exp(-tau[i] * E) * wbf  # * omega_factor
        F_wff[i] = np.trapz(kernel_wff, E)  # * omega_new[i]
        F_wbf[i] = np.trapz(kernel_wbf, E)  # * omega_new[i]

    F_tot_inel = F_wff + F_wbf
    return tau, F_tot_inel, F_wff, F_wbf


def get_atomic_data_for_all_elements(elements):

    nstates = len(elements)

    atomic_masses = np.zeros_like(elements, dtype=float)
    atomic_numbers = np.zeros_like(elements, dtype=float)
    for i in range(0, nstates):
        element = elements[i]
        amu, AN = get_atomic_mass_for_element(element)
        atomic_masses[i] = amu * ATOMIC_MASS_UNIT
        atomic_numbers[i] = AN

    return atomic_masses, atomic_numbers


def get_atomic_mass_for_element(e):
    """
    Load data from atomic data in folder xdave/data.
    """
    data_path = files("xdave") / "data" / "atomic_data.csv"

    ANs, elements, amus, _ = np.genfromtxt(
        data_path,
        delimiter=",",
        skip_header=1,
        dtype=None,
        unpack=True,
    )
    idx = np.where(elements == e)[0][0]
    atomic_weight = amus[idx]
    atomic_number = ANs[idx]
    return atomic_weight, atomic_number


def get_binding_energies_from_element(AN):
    # TODO(Hannah): this is a temporary fix until I get the file structure sorted out
    dat_file = os.path.dirname(__file__) + f"/data/binding_energies_xrdb.csv"
    df = pd.read_csv(dat_file)

    AN_col = df.columns[0]
    # Filter row for the given atomic number
    row = df[df[AN_col] == AN]

    if row.empty:
        return {}

    # Drop the AN column and any NaNs
    values = row.iloc[0].drop(labels=[AN_col]).fillna(0)
    return values.to_numpy() * (-1)


def get_emission_lines_for_element(element):
    df = pd.read_csv("data/emission_lines_table_1_2.csv")
    # Filter the row for the element
    row = df[df["Element"] == element]

    if row.empty:
        raise ValueError(f"Element '{element}' not found in the dataset.")

    row = row.iloc[0]  # Extract the single row as Series

    # Build dict of emission lines and energies
    emission_dict = {}
    for col in df.columns[1:]:  # skip "Element" column
        val = row[col]
        if pd.isna(val):
            val = 0  # or you could skip instead of setting to 0
        if pd.notna(val) and val != 0:
            emission_dict[col] = val

    return emission_dict


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return array[idx - 1], idx - 1
    else:
        return array[idx], idx


# ----------------------- #
# Fourier transform stuff #
# ----------------------- #


def forward_transform_fft(yr, r, k, dr, dk):
    """
    Fourier transform using the scipy.fft functions
    """
    n = len(r)

    weighted_yr = r[1:] * yr[1:]
    sum_vals = dst(weighted_yr, type=1)

    # Noramlize
    yk = np.zeros(n)
    yk[1:] = (2 * np.pi * dr / k[1:]) * sum_vals

    return yk


def inverse_transform_fft(yk, r, k, dr, dk):
    """
    Inverse fourier transform using the scipy.fft functions
    """
    n = len(r)

    weighted_yk = k[1:] * yk[1:]
    sum_vals = dst(weighted_yk, type=1)

    # Normalize
    yr = np.zeros(n)
    yr[1:] = (dk / (2 * np.pi) ** 2) * sum_vals / r[1:]

    # Extrapolate first point
    # Is this still necessary since I'm now doing this for the structure factor as well?
    yr[0] = 2 * yr[1] - yr[2]

    return yr


def forward_transform_fftn(yr, r, norm):
    """
    Fourier transform using the scipy.fft functions for matrixes
    """

    yk = np.zeros_like(yr)
    weighted_yr = yr[..., 1:].copy()
    weighted_yr *= r[1:]
    sum_vals = dst(weighted_yr, type=1, axis=-1)
    yk[..., 1:] = norm * sum_vals

    return yk


def inverse_transform_fftn(yk, k, norm):
    """
    Inverse fourier transform using the scipy.fft functions for matrixes
    I've tried to speed this up as much as I could, it is still slow af.
    """

    yr = np.zeros_like(yk)
    weighted_yk = yk[..., 1:].copy()
    weighted_yk *= k[1:]
    sum_vals = dst(weighted_yk, type=1, axis=-1)
    yr[..., 1:] = norm * sum_vals

    return yr


def spectral_convolution(spec_ene, omega, dsf, source_ene, source, Wr):
    """
    Tom's convolution :) I take no credit
    """
    spectrum = np.zeros_like(spec_ene)
    source /= np.sum(source)  # Normalise for convolution

    for ii, Ei in enumerate(source_ene):
        Bi = source[ii]
        momentum = (1.0 - omega / Ei) ** 2
        spectrum += np.interp(x=spec_ene, xp=Ei - omega, fp=dsf * momentum) * Bi

    # Now need to interpolate source on to the spectrum grid
    new_source = np.interp(x=spec_ene, xp=source_ene, fp=source)
    new_source /= np.sum(new_source)  # need to normalise

    # Apply Wr
    spectrum += new_source * Wr / (spec_ene[1] - spec_ene[0])

    return spectrum

import numpy as np
from constants import DIRAC_CONSTANT, SPEED_OF_LIGHT, BOHR_RADIUS, PI
from unit_conversions import eV_TO_J

from mendeleev import element
import pandas as pd
import os


def calculate_q(angle, energy):
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

    return angle


def load_mcss_result(filename):
    En, Es, lambda_s, wff, wbf, ff, bf, el, tot = np.genfromtxt(filename, skip_header=1, delimiter=",", unpack=True)
    return En[::-1], wff[::-1], wbf[::-1], ff, bf, el


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
    return F_tot_inel, F_wff, F_wbf


def get_atomic_mass_for_element(e):
    # y = element(int(AN))
    # amu = y.atomic_weight
    return element(str(e)).atomic_weight, element(str(e)).atomic_number


def get_binding_energies_from_elements(AN):
    dat_file = f"data/binding_energies_xrdb.csv"
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

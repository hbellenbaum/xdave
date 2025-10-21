import sys

sys.path.insert(1, "/home/bellen85/code/dev/xdave")

from mcss_tests.pymcss import *
from xdave import *
from unit_conversions import *

# from utils import calculate_angle


def run_be_sr_mode(T, rho, Z, angle, user_defined_ipd=0.0, user_defined_lfc=0.0, plot=False):
    THIS_DIR = os.path.dirname(__file__)
    mcss_dir = "~/code/mcss/mcss_ndtt/pro/mcss"
    mcss_executable = "mcss_60"  # "mcss_ndtt"  'mcss_51'

    Eb = 20e3
    # angle = calculate_angle(q=q, energy=Eb)

    # Z_min, Z_max = get_Z(Z)
    # x1, x2 = get_frac(Z=Z, Z_min=Z_min, Z_max=Z_max)

    results_dir = os.path.join(THIS_DIR, f"be_runs_T={T:.2f}_rho={rho:.2f}")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    params = MCSSParameters(
        mcss_executable=mcss_executable,
        mcss_dir=mcss_dir,
        deck_file=f"mcss_run_be_T={T:.2f}_rho={rho:.2f}_Z={Z}_angle={angle:.0f}",
        output_file=os.path.join(results_dir, f"mcss_run_be_T={T:.2f}_rho={rho:.2f}_Z={Z}_angle={angle:.0f}"),
        mass_density=rho,
        temperature=T,
        Zfs=np.array([Z]),
        ANs=np.array([4, 4]),
        fracs=np.array([1.0]),
        probe_energy=Eb,
        ipd=user_defined_ipd,
        ipd_model="USER_DEFINED",
        input_lfc=user_defined_lfc,
        bf_dsf_model="IMPULSE_SCHUMACHER",
        ee_pol_func_model="NUMERICAL_RPA",
        ee_lfc_model="NONE",
        deltaE=4000,
        ii_potential_model="DEBYE_HUCKEL",
        screen_cloud_model="FINITE_WAVELENGTH",
        ei_potential_model="EFFECTIVE_COULOMB",
        source_func_shape="GAUSSIAN",
        source_func_fwhm_ev=1.0,
        resolution=0.5,
        scattering_angle=angle,
    )
    En, wff, wbf, ff, bf, el = xrts_code_single(params=params)
    WR = get_mcss_wr_from_status_file(params.status_file_name)

    if plot:
        plt.figure()
        plt.plot(En, wff, label=f"ff")
        plt.plot(En, wbf, label=f"bf")
        plt.plot(En, wbf + wff, label=f"inel")
        plt.legend()
        plt.show()
        plt.close()

    return En, wff, wbf, ff, bf, el, WR


def run_c_sr_mode(T, rho, Z, angle, user_defined_ipd=0.0, user_defined_lfc=0.0, plot=False):
    THIS_DIR = os.path.dirname(__file__)
    mcss_dir = "~/code/mcss/mcss_ndtt/pro/mcss"
    mcss_executable = "mcss_60"  # "mcss_ndtt"  'mcss_51'

    Eb = 20e3

    results_dir = os.path.join(THIS_DIR, f"c_runs_T={T:.2f}_rho={rho:.2f}")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    params = MCSSParameters(
        mcss_executable=mcss_executable,
        mcss_dir=mcss_dir,
        deck_file=f"mcss_run_c_T={T:.2f}_rho={rho:.2f}_Z={Z}_angle={angle:.0f}",
        output_file=os.path.join(results_dir, f"mcss_run_c_T={T:.2f}_rho={rho:.2f}_Z={Z}_angle={angle:.0f}"),
        mass_density=rho,
        temperature=T,
        Zfs=np.array([Z]),
        ANs=np.array([6, 6]),
        fracs=np.array([1.0]),
        probe_energy=Eb,
        ipd=user_defined_ipd,
        ipd_model="USER_DEFINED",
        input_lfc=user_defined_lfc,
        bf_dsf_model="IMPULSE_SCHUMACHER",
        ee_pol_func_model="NUMERICAL_RPA",
        ee_lfc_model="NONE",
        deltaE=4000,
        ii_potential_model="EFFECTIVE_COULOMB",
        screen_cloud_model="FINITE_WAVELENGTH",
        ei_potential_model="EFFECTIVE_COULOMB",
        source_func_shape="GAUSSIAN",
        source_func_fwhm_ev=1.0,
        resolution=0.5,
        scattering_angle=angle,
    )
    En, wff, wbf, ff, bf, el = xrts_code_single(params=params)
    WR = get_mcss_wr_from_status_file(params.status_file_name)

    if plot:
        plt.figure()
        plt.plot(En, wff, label=f"ff")
        plt.plot(En, wbf, label=f"bf")
        plt.plot(En, wbf + wff, label=f"inel")
        plt.legend()
        plt.show()
        plt.close()

    return En, wff, wbf, ff, bf, el, WR


def run_ch_sr_mode(T, rho, xH, ZH, ZC, angle, user_defined_ipd=0.0, user_defined_lfc=0.0, plot=False):
    THIS_DIR = os.path.dirname(__file__)
    mcss_dir = "~/code/mcss/mcss_ndtt/pro/mcss"
    mcss_executable = "mcss_60"  # "mcss_ndtt"  'mcss_51'

    Eb = 20e3
    # angle = calculate_angle(q=q, energy=Eb)

    Z_min, Z_max = get_Z(ZC)
    x1, x2 = get_frac(Z=ZC, Z_min=Z_min, Z_max=Z_max, xlim=xH)

    results_dir = os.path.join(THIS_DIR, f"ch_runs_T={T:.2f}_rho={rho:.2f}")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    params = MCSSParameters(
        mcss_executable=mcss_executable,
        mcss_dir=mcss_dir,
        deck_file=f"mcss_run_ch_T={T:.2f}_rho={rho:.2f}_ZC={ZC}_angle={angle:.0f}",
        output_file=os.path.join(results_dir, f"mcss_run_ch_T={T:.2f}_rho={rho:.2f}_ZC={ZC}_angle={angle:.0f}"),
        mass_density=rho,
        temperature=T,
        Zfs=np.array([ZH, ZC]),
        ANs=np.array([1, 6, 6]),
        fracs=np.array([xH, x1, x2]),
        probe_energy=Eb,
        ipd=user_defined_ipd,
        ipd_model="USER_DEFINED",
        input_lfc=user_defined_lfc,
        bf_dsf_model="IMPULSE_SCHUMACHER",
        ee_pol_func_model="NUMERICAL_RPA",
        ee_lfc_model="NONE",
        deltaE=4000,
        ii_potential_model="DEBYE_HUCKEL",
        screen_cloud_model="FINITE_WAVELENGTH",
        ei_potential_model="EFFECTIVE_COULOMB",
        source_func_shape="GAUSSIAN",
        source_func_fwhm_ev=1.0,
        resolution=0.5,
        scattering_angle=angle,
    )
    En, wff, wbf, ff, bf, el = xrts_code_single3(params=params)
    WR = get_mcss_wr_from_status_file(params.status_file_name)

    if plot:
        plt.figure()
        plt.plot(En, wff, label=f"ff")
        plt.plot(En, wbf, label=f"bf")
        plt.plot(En, wbf + wff, label=f"inel")
        plt.legend()
        plt.show()
        plt.close()

    return En, wff, wbf, ff, bf, el, WR


def run_ch_ar_mode(T, rho, xH, ZH, ZC, angle, user_defined_ipd=0.0, user_defined_lfc=0.0, plot=False):
    THIS_DIR = os.path.dirname(__file__)
    mcss_dir = "~/code/mcss/mcss_ndtt/pro/mcss"
    mcss_executable = "mcss_60"  # "mcss_ndtt"  'mcss_51'

    Eb = 20e3
    # angle = calculate_angle(q=q, energy=Eb)

    # Z_min, Z_max = get_Z(ZC)
    # x1, x2 = get_frac(Z=ZC, Z_min=Z_min, Z_max=Z_max, xlim=xH)
    x2 = 1 - xH

    results_dir = os.path.join(THIS_DIR, f"ch_ar_runs_T={T:.2f}_rho={rho:.2f}")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    k_min = 0.1
    k_max = 13.3365

    params = MCSSParameters(
        mode_of_operation="XRTS_ANGULAR",
        mcss_executable=mcss_executable,
        mcss_dir=mcss_dir,
        deck_file=f"mcss_ar_run_ch_T={T:.2f}_rho={rho:.2f}_ZC={ZC}",
        output_file=os.path.join(results_dir, f"mcss_ar_run_ch_T={T:.2f}_rho={rho:.2f}_ZC={ZC}"),
        mass_density=rho,
        temperature=T,
        Zfs=np.array([ZH, ZC]),
        ANs=np.array([1, 6]),
        fracs=np.array([xH, x2]),
        probe_energy=Eb,
        ipd=user_defined_ipd,
        ipd_model="USER_DEFINED",
        input_lfc=user_defined_lfc,
        bf_dsf_model="IMPULSE_SCHUMACHER",
        ee_pol_func_model="NUMERICAL_RPA",
        ee_lfc_model="NONE",
        deltaE=4000,
        ii_potential_model="EFFECTIVE_COULOMB",
        screen_cloud_model="FINITE_WAVELENGTH",
        ei_potential_model="EFFECTIVE_COULOMB",
        source_func_shape="GAUSSIAN",
        source_func_fwhm_ev=1.0,
        resolution=0.5,
        scattering_angle=angle,
        min_wave_number=k_min,
        max_wave_number=k_max,
    )
    k, WR, f1, f2, q1, q2, S11, S12, S22 = xrts_code_single_ar(params=params)
    return k, WR, f1, f2, q1, q2, S11, S12, S22


def test():
    run_be_sr_mode(T=155.5, rho=30.0, Z=3.5, angle=90, plot=True)


if __name__ == "__main__":

    THIS_DIR = os.path.dirname(__file__)

    mcss_dir = "~/code/mcss/mcss_ndtt/pro/mcss"
    mcss_executable = "mcss_60"  # "mcss_ndtt"  'mcss_51'
    test()
    # compare_mcss_xdave_be()

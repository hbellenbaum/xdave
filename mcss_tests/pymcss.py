import matplotlib.pyplot as plt
import warnings
import numpy as np
import os


## Helper functions


def get_Z(Z):
    Z_min = np.floor(Z)
    Z_max = np.ceil(Z)
    return Z_min, Z_max


def get_frac(Z, Z_min, Z_max, xlim=0.0):
    frac_min = 0
    frac_max = 0
    tot = 1.0 - xlim

    if Z_min != Z_max:
        frac_max = tot * (Z - Z_min) / (Z_max - Z_min)
        frac_min = tot - frac_max
    else:
        frac_max = tot
        frac_min = 0.0
    return frac_min, frac_max


def get_ipd_from_status_file(status_file):
    ipd_message = "The IPD energy shift is:"
    # fr = open(status_file, 'r')
    ipd = None
    with open(status_file, "r") as fr:
        for line in fr.readlines():
            if ipd_message in line:
                ipd = line.split(": ")[1]
                ipd = ipd.split(" eV\n")[0]
    return float(ipd)


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


## MCSS Runs


class MCSSParameters:
    def __init__(
        self,
        mcss_executable="mcss",
        mcss_dir=None,
        deck_file=None,
        output_file="mcss_output",
        working_dir=os.path.dirname(__file__),
        # ionisation=1.0,
        temperature=10.0,
        mass_density=1.0,
        ANs=np.array([1, 1]),
        Zfs=np.array([0.5]),
        fracs=np.array([1.0]),
        # AN1=1,
        # AN2=1,
        resolution=0.1,
        ipd=None,
        input_lfc=0.0,
        probe_energy=None,
        deltaE=500,
        scattering_angle=17.0,
        min_wave_number=None,
        max_wave_number=None,
        mode_of_operation="XRTS_SPECTRAL",
        ee_pol_func_model="NUMERICAL_RPA",  # NUMERICAL_RPA, DANDREA_RPA_FIT, LINDHARD_RPA, TSYTOVICH_RPA, BORN_MERMIN, LANGDON_MATTE, FOURKAL_MATTE, GEN_LORENTZIAN
        ee_lfc_model="STATIC_UTSUMI_ICHIMARU",  # NONE, STATIC_GELDART_VOSKO, STATIC_UTSUMI_ICHIMARU, STATIC_FARID_ET_AL, STATIC_PADE_INTERP, INST_IWAMOTO_ICHIMARU, DYNAMIC_HONG_LEE  #TODO(Hannah): if MCSS does not recognise the model option, it will replace it with the default
        ii_potential_model="DEBYE_HUCKEL",  # EFFECTIVE_COULOMB, DEBYE_HUCKEL, FINITE_WAVELENGTH, NONLINEAR_HULTHEN, CHARGE_SWITCHING_DEBYE, SHORT_RANGE_REPULSION
        screen_cloud_model="FINITE_WAVELENGTH",  # FINITE_WAVELENGTH, DEBYE_HUCKEL
        ei_potential_model="EFFECTIVE_COULOMB",  # EFFECTIVE_COULOMB, HARD_EMPTY_CORE, SOFT_EMPTY_CORE
        bf_dsf_model="IMPULSE_SCHUMACHER",  # IMPULSE_SCHUMACHER, IMPULSE_HOLM_RIBBERFORS, PLANE_WAVE_FORM_FACTOR
        sec_core_power=6,
        ipd_model="NONE",  # NONE, DEBYE_HUCKEL, ION_SPHERE, STEWART_PYATT, "SAHA"
        source_func_type="ANALYTIC",
        source_func_shape=None,
        source_func_fwhm_ev=None,
        source_func_file_name=None,
    ) -> None:

        self.mode_of_operation = mode_of_operation
        self.mcss_executable = mcss_executable
        self.mcss_dir = mcss_dir
        self.deck_file = deck_file
        self.output_file = output_file
        self.working_dir = working_dir
        # self.Zf = ionisation
        self.Te = temperature
        self.rho = mass_density
        self.probe_energy = probe_energy
        self.deltaE = deltaE
        self.resolution = resolution
        self.ANs = ANs
        self.Zfs = Zfs
        self.fracs = fracs
        # self.AN1 = AN1
        # self.AN2 = AN2
        self.angle = scattering_angle

        self.ee_pol_func_model = ee_pol_func_model
        self.ee_lfc_model = ee_lfc_model
        self.ii_potential_model = ii_potential_model
        self.ii_potential_model = ii_potential_model
        self.screen_cloud_model = screen_cloud_model
        self.ei_potential_model = ei_potential_model
        self.bf_dsf_model = bf_dsf_model
        self.ipd_model = ipd_model
        self.sec_core_power = sec_core_power
        self.output_file_name = output_file + ".csv"
        self.status_file_name = output_file + "_status.txt"

        if ipd_model == "USER_DEFINED":
            assert ipd is not None
            self.ipd = ipd
        elif ipd_model == "NONE":
            self.ipd = 0.0

        if self.mode_of_operation == "XRTS_ANGULAR":
            assert min_wave_number and max_wave_number is not None
        self.min_wave_number = min_wave_number
        self.max_wave_number = max_wave_number

        if self.ee_lfc_model == "USER_DEFINED":
            self.user_defined_Gee = input_lfc
        elif self.ee_lfc_model == "NONE":
            # self.ee_lfc_model == ee_lfc_model
            self.user_defined_Gee = 0.0
        else:
            self.user_defined_Gee = -1.0

        # source_func_type = ("ANALYTIC",)
        if source_func_type == "ANALYTIC":
            self.use_source_func_data = "FALSE"
            assert source_func_shape is not None
            assert source_func_fwhm_ev is not None
            self.source_func_shape = source_func_shape
            self.source_func_fwhm_ev = source_func_fwhm_ev
            self.source_func_file_name = "default"
        elif source_func_type == "FROM_FILE":
            self.use_source_func_data = ".TRUE."
            assert source_func_file_name is not None
            self.source_func_file_name = source_func_file_name
            assert os.path.exists(self.source_func_file_name)
        else:
            raise NotImplementedError(f"Option {source_func_type} not recognized.")


def load_mcss_result(file):
    En, Es, _, wff, wbf, ff, bf, el, _ = np.genfromtxt(file, skip_header=1, delimiter=",", unpack=True)
    return En[::-1], wff[::-1], wbf[::-1], ff, bf, el


def load_mcss_result_wr(file):
    _, k, _, WR, f1, f2, q1, q2, S11, S12, S22 = np.genfromtxt(file, skip_header=1, delimiter=",", unpack=True)
    return k, WR


def xrts_code_single(params: MCSSParameters):

    res = params.resolution  # 0.1 # resulution in eV
    energy_max = params.probe_energy + params.deltaE  # + 500  # / 2
    energy_min = params.probe_energy - params.deltaE  # - 500  # /

    Z_min, Z_max = get_Z(params.Zfs[0])
    frac_min, frac_max = get_frac(params.Zfs[0], Z_min, Z_max)
    ipd_model = params.ipd_model

    if params.ipd_model == "USER_DEFINED":
        ipd_calc = params.ipd
        ipd_model = "NONE"
    elif params.ipd_model == "NONE":
        ipd_calc = 0.0
    else:
        ipd_calc = 0.0  # params.ipd

    # Read format file and create new deck for given parameters
    with open(params.working_dir + "/decks/my_mcss.deck", "rt") as fr:
        data = fr.read()
        data = data.format(
            mode_of_operation="XRTS_SPECTRAL",
            AN1=params.ANs[0],
            AN2=params.ANs[1],
            Z_min=Z_min,
            Z_max=Z_max,
            frac_min=frac_min,
            frac_max=frac_max,
            rho=params.rho,
            T=params.Te,
            probe_energy=params.probe_energy,
            angle=params.angle,
            n_points=int((energy_max - energy_min) / res),
            E_min=energy_min,
            E_max=energy_max,
            ee_pol_func_model=params.ee_pol_func_model,  # NUMERICAL_RPA, DANDREA_RPA_FIT, LINDHARD_RPA, TSYTOVICH_RPA, BORN_MERMIN, LANGDON_MATTE, FOURKAL_MATTE, GEN_LORENTZIAN
            ee_lfc_model=params.ee_lfc_model,  # NONE, STATIC_GELDART_VOSKO, STATIC_UTSUMI_ICHIMARU, STATIC_FARID_ET_AL, STATIC_INTERP, INST_IWAMOTO_ICHIMARU, DYNAMIC_HONG_LEE  #TODO(Hannah): if MCSS does not recognise the model option, it will replace it with the default
            ii_potential_model=params.ii_potential_model,  # EFFECTIVE_COULOMB, DEBYE_HUCKEL, FINITE_WAVELENGTH, NONLINEAR_HULTHEN, CHARGE_SWITCHING_DEBYE, SHORT_RANGE_REPULSION
            screen_cloud_model=params.screen_cloud_model,  # FINITE_WAVELENGTH, DEBYE_HUCKEL
            ei_potential_model=params.ei_potential_model,  # EFFECTIVE_COULOMB, HARD_EMPTY_CORE, SOFT_EMPTY_CORE
            bf_dsf_model=params.bf_dsf_model,  # IMPULSE_SCHUMACHER, IMPULSE_HOLM_RIBBERFORS, PLANE_WAVE_FORM_FACTOR
            sec_core_power=6,
            ipd_model=ipd_model,  # NONE, DEBYE_HUCKEL, ION_SPHERE, STEWART_PYATT,
            user_defined_ipd=ipd_calc,
            user_defined_Gee=params.user_defined_Gee,  # only the STATIC component!!!!!!
            output_file_name=params.output_file_name,
            status_file_name=params.status_file_name,
            use_source_func_data=params.use_source_func_data,
            source_func_shape=params.source_func_shape,
            source_func_fwhm_ev=params.source_func_fwhm_ev,
            source_func_file_name=params.source_func_file_name,
        )

    return run_mcss(data, params)


def xrts_code_single3(params: MCSSParameters):

    res = params.resolution  # 0.1 # resulution in eV
    # energy_max = params.probe_energy + 0.15 * params.probe_energy  # / 2
    # energy_min = params.probe_energy - 0.15 * params.probe_energy  # / 2

    energy_max = params.probe_energy + params.deltaE  # + 500  # / 2
    energy_min = params.probe_energy - params.deltaE  # - 500  # /

    Z1 = params.Zfs[0]
    Z2 = params.Zfs[1]
    x1 = params.fracs[0]
    Z_min, Z_max = get_Z(Z2)
    x2, x3 = get_frac(Z2, Z_min, Z_max, xlim=x1)
    ipd_model = params.ipd_model

    if params.ipd_model == "USER_DEFINED":
        ipd_calc = params.ipd
        ipd_model = "NONE"
    elif params.ipd_model == "NONE":
        ipd_calc = 0.0
    else:
        ipd_calc = 0.0  # params.ipd

    # Read format file and create new deck for given parameters
    with open(params.working_dir + "/decks/my_mcss3.deck", "rt") as fr:
        data = fr.read()
        data = data.format(
            mode_of_operation="XRTS_SPECTRAL",
            AN1=params.ANs[0],
            AN2=params.ANs[1],
            AN3=params.ANs[2],
            Z1=Z1,
            Z2=Z_min,
            Z3=Z_max,
            x1=x1,
            x2=x2,
            x3=x3,
            rho=params.rho,
            T=params.Te,
            probe_energy=params.probe_energy,
            angle=params.angle,
            n_points=int((energy_max - energy_min) / res),
            E_min=energy_min,
            E_max=energy_max,
            ee_pol_func_model=params.ee_pol_func_model,  # NUMERICAL_RPA, DANDREA_RPA_FIT, LINDHARD_RPA, TSYTOVICH_RPA, BORN_MERMIN, LANGDON_MATTE, FOURKAL_MATTE, GEN_LORENTZIAN
            ee_lfc_model=params.ee_lfc_model,  # NONE, STATIC_GELDART_VOSKO, STATIC_UTSUMI_ICHIMARU, STATIC_FARID_ET_AL, STATIC_INTERP, INST_IWAMOTO_ICHIMARU, DYNAMIC_HONG_LEE  #TODO(Hannah): if MCSS does not recognise the model option, it will replace it with the default
            ii_potential_model=params.ii_potential_model,  # EFFECTIVE_COULOMB, DEBYE_HUCKEL, FINITE_WAVELENGTH, NONLINEAR_HULTHEN, CHARGE_SWITCHING_DEBYE, SHORT_RANGE_REPULSION
            screen_cloud_model=params.screen_cloud_model,  # FINITE_WAVELENGTH, DEBYE_HUCKEL
            ei_potential_model=params.ei_potential_model,  # EFFECTIVE_COULOMB, HARD_EMPTY_CORE, SOFT_EMPTY_CORE
            bf_dsf_model=params.bf_dsf_model,  # IMPULSE_SCHUMACHER, IMPULSE_HOLM_RIBBERFORS, PLANE_WAVE_FORM_FACTOR
            sec_core_power=6,
            ipd_model=ipd_model,  # NONE, DEBYE_HUCKEL, ION_SPHERE, STEWART_PYATT,
            user_defined_ipd=ipd_calc,
            user_defined_Gee=params.user_defined_Gee,  # only the STATIC component!!!!!!
            output_file_name=params.output_file_name,
            status_file_name=params.status_file_name,
            use_source_func_data=params.use_source_func_data,
            source_func_shape=params.source_func_shape,
            source_func_fwhm_ev=params.source_func_fwhm_ev,
            source_func_file_name=params.source_func_file_name,
        )

    return run_mcss(data, params)


def xrts_code_single5(params: MCSSParameters):

    res = params.resolution  # 0.1 # resulution in eV
    # energy_max = params.probe_energy + 0.15 * params.probe_energy  # / 2
    # energy_min = params.probe_energy - 0.15 * params.probe_energy  # / 2
    energy_max = params.probe_energy + params.deltaE  # + 500  # / 2
    energy_min = params.probe_energy - params.deltaE  # - 500  # /

    # ZC1, ZC2 = get_Z(Z=params.Zf)
    # x2, x3 = get_frac(Z=params.Zf, Z_min=ZC1, Z_max=ZC2, x1=xH + xO)
    # ZO1, ZO2 = get_Z(params.Zf2)
    # x4, x5 = get_frac(Z=params.Zf2, Z_max=ZO1, Z_min=ZO2, x1=xH + xC)

    x1 = params.fracs[0]
    xs2 = params.fracs[1]
    xs3 = params.fracs[2]

    assert x1 + xs2 + xs3 == 1

    Z1 = params.Zfs[0]
    Zf2 = params.Zfs[1]
    Zf3 = params.Zfs[2]

    Z2_min, Z2_max = get_Z(Zf2)
    x2, x3 = get_frac(Zf2, Z2_min, Z2_max, xlim=x1 + xs3)

    Z3_min, Z3_max = get_Z(Zf3)
    x4, x5 = get_frac(Zf3, Z3_min, Z3_max, xlim=x1 + xs2)

    assert x1 + x2 + x3 + x4 + x5 == 1
    assert np.all((x1, x2, x3, x4, x5)) >= 0

    if params.ipd_model == "USER_DEFINED":
        ipd_calc = params.ipd
        ipd_model = "NONE"
    elif params.ipd_model == "NONE":
        ipd_calc = 0.0
        ipd_model = "NONE"
    else:
        ipd_calc = 0.0  # params.ipd
        ipd_model = "NONE"

    # Read format file and create new deck for given parameters
    with open(params.working_dir + "/decks/my_mcss5.deck", "rt") as fr:
        data = fr.read()
        data = data.format(
            mode_of_operation="XRTS_SPECTRAL",
            AN1=params.ANs[0],
            AN2=params.ANs[1],
            AN3=params.ANs[2],
            AN4=params.ANs[3],
            AN5=params.ANs[4],
            # AN6=params.ANs[3],
            # AN7=params.ANs[4],
            Z1=Z1,
            Z2=Z2_min,
            Z3=Z2_max,
            Z4=Z3_min,
            Z5=Z3_max,
            # Z6=Z4_min,
            # Z7=Z4_max,
            x1=x1,
            x2=x2,
            x3=x3,
            x4=x4,
            x5=x5,
            rho=params.rho,
            T=params.Te,
            probe_energy=params.probe_energy,
            angle=params.angle,
            n_points=int((energy_max - energy_min) / res),
            E_min=energy_min,
            E_max=energy_max,
            ee_pol_func_model=params.ee_pol_func_model,  # NUMERICAL_RPA, DANDREA_RPA_FIT, LINDHARD_RPA, TSYTOVICH_RPA, BORN_MERMIN, LANGDON_MATTE, FOURKAL_MATTE, GEN_LORENTZIAN
            ee_lfc_model=params.ee_lfc_model,  # NONE, STATIC_GELDART_VOSKO, STATIC_UTSUMI_ICHIMARU, STATIC_FARID_ET_AL, STATIC_INTERP, INST_IWAMOTO_ICHIMARU, DYNAMIC_HONG_LEE  #TODO(Hannah): if MCSS does not recognise the model option, it will replace it with the default
            ii_potential_model=params.ii_potential_model,  # EFFECTIVE_COULOMB, DEBYE_HUCKEL, FINITE_WAVELENGTH, NONLINEAR_HULTHEN, CHARGE_SWITCHING_DEBYE, SHORT_RANGE_REPULSION
            screen_cloud_model=params.screen_cloud_model,  # FINITE_WAVELENGTH, DEBYE_HUCKEL
            ei_potential_model=params.ei_potential_model,  # EFFECTIVE_COULOMB, HARD_EMPTY_CORE, SOFT_EMPTY_CORE
            bf_dsf_model=params.bf_dsf_model,  # IMPULSE_SCHUMACHER, IMPULSE_HOLM_RIBBERFORS, PLANE_WAVE_FORM_FACTOR
            sec_core_power=6,
            ipd_model=ipd_model,  # NONE, DEBYE_HUCKEL, ION_SPHERE, STEWART_PYATT,
            user_defined_ipd=ipd_calc,
            user_defined_Gee=params.user_defined_Gee,  # only the STATIC component!!!!!!
            output_file_name=params.output_file_name,
            status_file_name=params.status_file_name,
            # use_source_func_data=".FALSE.",
            # source_func_shape="GAUSSIAN",
            # source_func_fwhm_ev=1,
            # source_func_file_name="default",
            use_source_func_data=params.use_source_func_data,
            source_func_shape=params.source_func_shape,
            source_func_fwhm_ev=params.source_func_fwhm_ev,
            source_func_file_name=params.source_func_file_name,
        )

    return run_mcss(data, params)


def xrts_code_single5(params: MCSSParameters):

    res = params.resolution  # 0.1 # resulution in eV
    # energy_max = params.probe_energy + 0.15 * params.probe_energy  # / 2
    # energy_min = params.probe_energy - 0.15 * params.probe_energy  # / 2
    energy_max = params.probe_energy + params.deltaE  # + 500  # / 2
    energy_min = params.probe_energy - params.deltaE  # - 500  # /

    # ZC1, ZC2 = get_Z(Z=params.Zf)
    # x2, x3 = get_frac(Z=params.Zf, Z_min=ZC1, Z_max=ZC2, x1=xH + xO)
    # ZO1, ZO2 = get_Z(params.Zf2)
    # x4, x5 = get_frac(Z=params.Zf2, Z_max=ZO1, Z_min=ZO2, x1=xH + xC)

    x1 = params.fracs[0]
    xs2 = params.fracs[1]
    xs3 = params.fracs[2]

    assert x1 + xs2 + xs3 == 1

    Z1 = params.Zfs[0]
    Zf2 = params.Zfs[1]
    Zf3 = params.Zfs[2]

    Z2_min, Z2_max = get_Z(Zf2)
    x2, x3 = get_frac(Zf2, Z2_min, Z2_max, xlim=x1 + xs3)

    Z3_min, Z3_max = get_Z(Zf3)
    x4, x5 = get_frac(Zf3, Z3_min, Z3_max, xlim=x1 + xs2)

    assert x1 + x2 + x3 + x4 + x5 == 1
    assert np.all((x1, x2, x3, x4, x5)) >= 0

    if params.ipd_model == "USER_DEFINED":
        ipd_calc = params.ipd
        ipd_model = "NONE"
    elif params.ipd_model == "NONE":
        ipd_calc = 0.0
        ipd_model = "NONE"
    else:
        ipd_calc = 0.0  # params.ipd
        ipd_model = "NONE"

    # Read format file and create new deck for given parameters
    with open(params.working_dir + "/decks/my_mcss5.deck", "rt") as fr:
        data = fr.read()
        data = data.format(
            mode_of_operation="XRTS_SPECTRAL",
            AN1=params.ANs[0],
            AN2=params.ANs[1],
            AN3=params.ANs[2],
            AN4=params.ANs[3],
            AN5=params.ANs[4],
            Z1=Z1,
            Z2=Z2_min,
            Z3=Z2_max,
            Z4=Z3_min,
            Z5=Z3_max,
            x1=x1,
            x2=x2,
            x3=x3,
            x4=x4,
            x5=x5,
            rho=params.rho,
            T=params.Te,
            probe_energy=params.probe_energy,
            angle=params.angle,
            n_points=int((energy_max - energy_min) / res),
            E_min=energy_min,
            E_max=energy_max,
            ee_pol_func_model=params.ee_pol_func_model,  # NUMERICAL_RPA, DANDREA_RPA_FIT, LINDHARD_RPA, TSYTOVICH_RPA, BORN_MERMIN, LANGDON_MATTE, FOURKAL_MATTE, GEN_LORENTZIAN
            ee_lfc_model=params.ee_lfc_model,  # NONE, STATIC_GELDART_VOSKO, STATIC_UTSUMI_ICHIMARU, STATIC_FARID_ET_AL, STATIC_INTERP, INST_IWAMOTO_ICHIMARU, DYNAMIC_HONG_LEE  #TODO(Hannah): if MCSS does not recognise the model option, it will replace it with the default
            ii_potential_model=params.ii_potential_model,  # EFFECTIVE_COULOMB, DEBYE_HUCKEL, FINITE_WAVELENGTH, NONLINEAR_HULTHEN, CHARGE_SWITCHING_DEBYE, SHORT_RANGE_REPULSION
            screen_cloud_model=params.screen_cloud_model,  # FINITE_WAVELENGTH, DEBYE_HUCKEL
            ei_potential_model=params.ei_potential_model,  # EFFECTIVE_COULOMB, HARD_EMPTY_CORE, SOFT_EMPTY_CORE
            bf_dsf_model=params.bf_dsf_model,  # IMPULSE_SCHUMACHER, IMPULSE_HOLM_RIBBERFORS, PLANE_WAVE_FORM_FACTOR
            sec_core_power=6,
            ipd_model=ipd_model,  # NONE, DEBYE_HUCKEL, ION_SPHERE, STEWART_PYATT,
            user_defined_ipd=ipd_calc,
            user_defined_Gee=params.user_defined_Gee,  # only the STATIC component!!!!!!
            output_file_name=params.output_file_name,
            status_file_name=params.status_file_name,
            use_source_func_data=params.use_source_func_data,
            source_func_shape=params.source_func_shape,
            source_func_fwhm_ev=params.source_func_fwhm_ev,
            source_func_file_name=params.source_func_file_name,
        )

    return run_mcss(data, params)


def xrts_code_single7(params: MCSSParameters):

    res = params.resolution  # 0.1 # resulution in eV
    # energy_max = params.probe_energy + 0.15 * params.probe_energy  # / 2
    # energy_min = params.probe_energy - 0.15 * params.probe_energy  # / 2
    energy_max = params.probe_energy + params.deltaE  # + 500  # / 2
    energy_min = params.probe_energy - params.deltaE  # - 500  # /

    x1 = params.fracs[0]
    xs2 = params.fracs[1]
    xs3 = params.fracs[2]
    xs4 = params.fracs[3]

    assert x1 + xs2 + xs3 + xs4 == 1

    Z1 = params.Zfs[0]
    Zf2 = params.Zfs[1]
    Zf3 = params.Zfs[2]
    Zf4 = params.Zfs[3]

    Z2_min, Z2_max = get_Z(Zf2)
    x2, x3 = get_frac(Zf2, Z2_min, Z2_max, xlim=x1 + xs3 + xs4)

    Z3_min, Z3_max = get_Z(Zf3)
    x4, x5 = get_frac(Zf3, Z3_min, Z3_max, xlim=x1 + xs2 + xs4)

    Z4_min, Z4_max = get_Z(Zf4)
    x6, x7 = get_frac(Zf4, Z4_min, Z4_max, xlim=x1 + xs2 + xs3)

    assert x1 + x2 + x3 + x4 + x5 + x6 + x7 == 1
    assert np.all((x1, x2, x3, x4, x5, x6, x7)) >= 0

    if params.ipd_model == "USER_DEFINED":
        ipd_calc = params.ipd
        ipd_model = "NONE"
    elif params.ipd_model == "NONE":
        ipd_calc = 0.0
        ipd_model = "NONE"
    else:
        ipd_calc = 0.0  # params.ipd
        ipd_model = "NONE"

    # Read format file and create new deck for given parameters
    with open(params.working_dir + "/decks/my_mcss7.deck", "rt") as fr:
        data = fr.read()
        data = data.format(
            mode_of_operation="XRTS_SPECTRAL",
            AN1=params.ANs[0],
            AN2=params.ANs[1],
            AN3=params.ANs[2],
            AN4=params.ANs[3],
            AN5=params.ANs[4],
            AN6=params.ANs[5],
            AN7=params.ANs[6],
            Z1=Z1,
            Z2=Z2_min,
            Z3=Z2_max,
            Z4=Z3_min,
            Z5=Z3_max,
            Z6=Z4_min,
            Z7=Z4_max,
            x1=x1,
            x2=x2,
            x3=x3,
            x4=x4,
            x5=x5,
            x6=x6,
            x7=x7,
            rho=params.rho,
            T=params.Te,
            probe_energy=params.probe_energy,
            angle=params.angle,
            n_points=int((energy_max - energy_min) / res),
            E_min=energy_min,
            E_max=energy_max,
            ee_pol_func_model=params.ee_pol_func_model,  # NUMERICAL_RPA, DANDREA_RPA_FIT, LINDHARD_RPA, TSYTOVICH_RPA, BORN_MERMIN, LANGDON_MATTE, FOURKAL_MATTE, GEN_LORENTZIAN
            ee_lfc_model=params.ee_lfc_model,  # NONE, STATIC_GELDART_VOSKO, STATIC_UTSUMI_ICHIMARU, STATIC_FARID_ET_AL, STATIC_INTERP, INST_IWAMOTO_ICHIMARU, DYNAMIC_HONG_LEE  #TODO(Hannah): if MCSS does not recognise the model option, it will replace it with the default
            ii_potential_model=params.ii_potential_model,  # EFFECTIVE_COULOMB, DEBYE_HUCKEL, FINITE_WAVELENGTH, NONLINEAR_HULTHEN, CHARGE_SWITCHING_DEBYE, SHORT_RANGE_REPULSION
            screen_cloud_model=params.screen_cloud_model,  # FINITE_WAVELENGTH, DEBYE_HUCKEL
            ei_potential_model=params.ei_potential_model,  # EFFECTIVE_COULOMB, HARD_EMPTY_CORE, SOFT_EMPTY_CORE
            bf_dsf_model=params.bf_dsf_model,  # IMPULSE_SCHUMACHER, IMPULSE_HOLM_RIBBERFORS, PLANE_WAVE_FORM_FACTOR
            sec_core_power=6,
            ipd_model=ipd_model,  # NONE, DEBYE_HUCKEL, ION_SPHERE, STEWART_PYATT,
            user_defined_ipd=ipd_calc,
            user_defined_Gee=params.user_defined_Gee,  # only the STATIC component!!!!!!
            output_file_name=params.output_file_name,
            status_file_name=params.status_file_name,
            use_source_func_data=params.use_source_func_data,
            source_func_shape=params.source_func_shape,
            source_func_fwhm_ev=params.source_func_fwhm_ev,
            source_func_file_name=params.source_func_file_name,
        )

    return run_mcss(data, params)


def xrts_code_single_ar(params: MCSSParameters):

    res = params.resolution  # 0.1 # resulution in eV

    # Z_min, Z_max = get_Z(params.Zfs[0])
    # frac_min, frac_max = get_frac(params.Zfs[0], Z_min, Z_max)
    # ipd_model = params.ipd_model

    if params.ipd_model == "USER_DEFINED":
        ipd_calc = params.ipd
        ipd_model = "NONE"
    elif params.ipd_model == "NONE":
        ipd_model = "NONE"
        ipd_calc = 0.0
    else:
        ipd_calc = 0.0  # params.ipd
        warnings.warn(
            f"Overwriting ipd model {ipd_model} with ipd = {ipd_calc}. If the model is not set to SAHA, USER_DEFINED or NONE, this is a mistake!"
        )

    # Read format file and create new deck for given parameters
    with open(params.working_dir + "/decks/mcss_ar.deck", "rt") as fr:
        data = fr.read()
        data = data.format(
            AN1=params.ANs[0],
            AN2=params.ANs[1],
            Z_min=params.Zfs[0],
            Z_max=params.Zfs[1],
            frac_min=params.fracs[0],
            frac_max=params.fracs[1],
            rho=params.rho,
            T=params.Te,
            probe_energy=params.probe_energy,
            angle=params.angle,
            n_points=10000,  # hard-coded for now
            k_min=params.min_wave_number,
            k_max=params.max_wave_number,
            ee_pol_func_model=params.ee_pol_func_model,  # NUMERICAL_RPA, DANDREA_RPA_FIT, LINDHARD_RPA, TSYTOVICH_RPA, BORN_MERMIN, LANGDON_MATTE, FOURKAL_MATTE, GEN_LORENTZIAN
            ee_lfc_model=params.ee_lfc_model,  # NONE, STATIC_GELDART_VOSKO, STATIC_UTSUMI_ICHIMARU, STATIC_FARID_ET_AL, STATIC_PADE_INTERP, INST_IWAMOTO_ICHIMARU, DYNAMIC_HONG_LEE  #TODO(Hannah): if MCSS does not recognise the model option, it will replace it with the default
            ii_potential_model=params.ii_potential_model,  # EFFECTIVE_COULOMB, DEBYE_HUCKEL, FINITE_WAVELENGTH, NONLINEAR_HULTHEN, CHARGE_SWITCHING_DEBYE, SHORT_RANGE_REPULSION
            screen_cloud_model=params.screen_cloud_model,  # FINITE_WAVELENGTH, DEBYE_HUCKEL
            ei_potential_model=params.ei_potential_model,  # EFFECTIVE_COULOMB, HARD_EMPTY_CORE, SOFT_EMPTY_CORE
            sec_core_power=6,
            bf_dsf_model=params.bf_dsf_model,
            ipd_model=ipd_model,  # NONE, DEBYE_HUCKEL, ION_SPHERE, STEWART_PYATT,
            user_defined_ipd=ipd_calc,
            output_file_name=params.output_file_name,
            status_file_name=params.status_file_name,
            # use_source_func_data=".FALSE.",
            # source_func_shape="GAUSSIAN",
            # source_func_fwhm_ev=1,
            # source_func_file_name="default",
            use_source_func_data=params.use_source_func_data,
            source_func_shape=params.source_func_shape,
            source_func_fwhm_ev=params.source_func_fwhm_ev,
            source_func_file_name=params.source_func_file_name,
        )

    return run_mcss(data, params)


def xrts_code_single_ar3(params: MCSSParameters):

    res = params.resolution  # 0.1 # resulution in eV

    # Z_min, Z_max = get_Z(params.Zfs[0])
    # frac_min, frac_max = get_frac(params.Zfs[0], Z_min, Z_max)
    # ipd_model = params.ipd_model

    if params.ipd_model == "USER_DEFINED":
        ipd_calc = params.ipd
        ipd_model = "NONE"
    elif params.ipd_model == "NONE":
        ipd_model = "NONE"
        ipd_calc = 0.0
    else:
        ipd_calc = 0.0  # params.ipd
        warnings.warn(
            f"Overwriting ipd model {ipd_model} with ipd = {ipd_calc}. If the model is not set to SAHA, USER_DEFINED or NONE, this is a mistake!"
        )

    # Read format file and create new deck for given parameters
    with open(params.working_dir + "/decks/mcss_ar3.deck", "rt") as fr:
        data = fr.read()
        data = data.format(
            AN1=params.ANs[0],
            AN2=params.ANs[1],
            AN3=params.ANs[2],
            Z1=params.Zfs[0],
            Z2=params.Zfs[1],
            Z3=params.Zfs[2],
            x1=params.fracs[0],
            x2=params.fracs[1],
            x3=params.fracs[2],
            rho=params.rho,
            T=params.Te,
            probe_energy=params.probe_energy,
            angle=params.angle,
            n_points=10000,  # hard-coded for now
            k_min=params.min_wave_number,
            k_max=params.max_wave_number,
            ee_pol_func_model=params.ee_pol_func_model,  # NUMERICAL_RPA, DANDREA_RPA_FIT, LINDHARD_RPA, TSYTOVICH_RPA, BORN_MERMIN, LANGDON_MATTE, FOURKAL_MATTE, GEN_LORENTZIAN
            ee_lfc_model=params.ee_lfc_model,  # NONE, STATIC_GELDART_VOSKO, STATIC_UTSUMI_ICHIMARU, STATIC_FARID_ET_AL, STATIC_PADE_INTERP, INST_IWAMOTO_ICHIMARU, DYNAMIC_HONG_LEE  #TODO(Hannah): if MCSS does not recognise the model option, it will replace it with the default
            ii_potential_model=params.ii_potential_model,  # EFFECTIVE_COULOMB, DEBYE_HUCKEL, FINITE_WAVELENGTH, NONLINEAR_HULTHEN, CHARGE_SWITCHING_DEBYE, SHORT_RANGE_REPULSION
            screen_cloud_model=params.screen_cloud_model,  # FINITE_WAVELENGTH, DEBYE_HUCKEL
            ei_potential_model=params.ei_potential_model,  # EFFECTIVE_COULOMB, HARD_EMPTY_CORE, SOFT_EMPTY_CORE
            sec_core_power=6,
            bf_dsf_model=params.bf_dsf_model,
            ipd_model=ipd_model,  # NONE, DEBYE_HUCKEL, ION_SPHERE, STEWART_PYATT,
            user_defined_ipd=ipd_calc,
            output_file_name=params.output_file_name,
            status_file_name=params.status_file_name,
            # use_source_func_data=".FALSE.",
            # source_func_shape="GAUSSIAN",
            # source_func_fwhm_ev=1,
            # source_func_file_name="default",
            use_source_func_data=params.use_source_func_data,
            source_func_shape=params.source_func_shape,
            source_func_fwhm_ev=params.source_func_fwhm_ev,
            source_func_file_name=params.source_func_file_name,
        )

    return run_mcss(data, params)


def run_mcss(deck_data, params: MCSSParameters):
    with open(params.working_dir + f"/decks/{params.deck_file}.deck", "w") as fw:
        fw.write(deck_data)
        fw.write("\n")  # because fortran...

    os.chdir(params.working_dir)
    os.system(
        f"{params.mcss_dir}/{params.mcss_executable} {params.deck_file} > {params.working_dir}/mcss_{params.deck_file}.log"
    )
    if params.mode_of_operation == "XRTS_SPECTRAL":
        try:
            # ! E [eV]  E_{s} [eV]   \lambda_{s} [nm] W_{ff} [1/eV]   W_{bf} [1/eV]   P_{s}^{ff} [Arb.]   P_{s}^{bf} [Arb.]   P_{s}^{el} [Arb.]   P_{s} [Arb.]
            En, Es, _, wff, wbf, ff, bf, el, _ = np.genfromtxt(
                params.output_file_name, skip_header=1, delimiter=",", unpack=True
            )
        except:
            warnings.warn("MCSS has not run succesfully or the file location has not been specified correctly.")
            exit()
        return En[::-1], wff[::-1], wbf[::-1], ff, bf, el

    elif params.mode_of_operation == "XRTS_ANGULAR":
        if len(params.ANs) == 3:
            # ! k [1/\AA]           k [1/a_{B}]         	heta [\deg]        W_{R}(k)            f_{         1}(k)   f_{         2}(k)   f_{         3}(k)   q_{         1}(k)   q_{         2}(k)   q_{         3}(k)   S_{         1       S_{         1       S_{         1       S_{         2       S_{         2       S_{         3
            # S_{1  S_{1  S_{1   S_{2  S_{2  S_{3
            try:
                _, k, _, WR, f1, f2, f3, q1, q2, q3, S11, S12, S13, S22, S23, S33 = np.genfromtxt(
                    params.output_file_name, skip_header=1, delimiter=",", unpack=True
                )
                return k, WR, f1, f2, f3, q1, q2, q3, S11, S13, S12, S22, S23, S33
            except FileNotFoundError as e:
                warnings.warn(print(f"Cannot load MCSS output file: {e}"))
                exit()

        try:
            if params.ee_lfc_model == "NONE":
                # ! k [1/\AA]  k [1/a_{B}] 	heta [\deg]  W_{R}(k) f_{ 1}(k) f_{ 2}(k)  q_{ 1}(k)  q_{ 2}(k)  S_{ 1 1}(k)  S_{ 1 2}(k)  S_{ 2 2}(k)
                _, k, _, WR, f1, f2, q1, q2, S11, S12, S22 = np.genfromtxt(
                    params.output_file_name, skip_header=1, delimiter=",", unpack=True
                )
                lfc = np.zeros_like(k)
            else:
                _, k, _, WR, f1, f2, q1, q2, S11, S12, S22, lfc = np.genfromtxt(
                    params.output_file_name, skip_header=1, delimiter=",", unpack=True
                )

        except FileNotFoundError as e:
            warnings.warn(print(f"Cannot load MCSS output file: {e}"))
            exit()

        return k, WR, f1, f2, q1, q2, S11, S12, S22, lfc


# if __name__ == "__main__":
#     # test_wr()
#     pass

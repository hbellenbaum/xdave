# Ionic form factors
from models import ModelOptions
from plasma_state import PlasmaState

from bridge_functions import *
from utils import (
    forward_transform_fft,
    inverse_transform_fft,
    forward_transform_fftn,
    inverse_transform_fftn,
)
from constants import BOHR_RADIUS, ELEMENTARY_CHARGE, PI, BOLTZMANN_CONSTANT, VACUUM_PERMITTIVITY
from potentials import *
from unit_conversions import *

from scipy.interpolate import interp1d
from scipy.optimize import minimize
import numpy as np


def h0(eta, chi):
    return (1 + 2 * eta) / (1 - eta) * (1 - (1 + 2 * (1 - eta) ** 3 * chi / (1 + 2 * eta) ** 2) ** (1 / 2))


def h1(eta, chi):
    """
    Enforce h1=0 to get a value for eta
    """
    return h0(eta, chi) ** 2 / (24 * eta) - (1 + eta / 2) / (1 - eta) ** 2


def h2(eta, chi):
    return -(1 + eta - eta**2 / 5) / (12 * eta) - (1 - eta) * h0(eta, chi) / (12 * eta * chi)


def get_sigmac(ni, Zi, Ti):

    def h1_long(sigma_c):
        """
        Enforce h1=0 to get a value for eta
        """
        eta = PI / 6 * ni * sigma_c**3  # [ ]
        gamma = (
            Zi**2 * ELEMENTARY_CHARGE**2 / (VACUUM_PERMITTIVITY * sigma_c * BOLTZMANN_CONSTANT * Ti)
        )  # 1 / (kg^-1 m^-3 s^2  m  J ) = kg m^2 s^{-2} / J = J / J = [ ]
        chi = np.sqrt(24 * eta * gamma)  # [ ]
        h0_val = (1 + 2 * eta) / (1 - eta) * (1 - (1 + 2 * (1 - eta) ** 3 * chi / (1 + 2 * eta) ** 2) ** (1 / 2))
        return h0_val**2 / (24 * eta) - (1 + eta / 2) / (1 - eta) ** 2

    sol = minimize(h1_long, x0=1.0e-12, method="nelder-mead", bounds=((1.0e-14, 5.0e-11),))
    # print(sol)
    return sol.x


## ---------------------------- ##
## One-component implementation ##
## ---------------------------- ##


class OCPStaticStructureFactor:
    """
    Class for the one-component static structure factor.

    Attributes:
        state (PlasmaState): container holding all plasma parameters
        max_iterations (int): maximum number of iterations in the HNC solver
        mix_fraction (float): number between 0 and 1, values closer to 1 mean less of the new iterated solution in the HNC solver is included
        delta (float): tolerance to check convergence of the HNC result
        n (int): number of points in the HNC grid
    """

    def __init__(self, state: PlasmaState, max_iterations=5000, mix_fraction=0.8, delta=1.0e-8, n=8192, verbose=False):
        self.state = state
        self.beta = 1 / (BOLTZMANN_CONSTANT * state.ion_temperature)  # [1/J]
        self.n = n  # per themis [#]

        self.Rii = self.state.mean_sphere_radius(number_density=self.state.ion_number_density)  # [m]
        self.alpha = 2 / self.Rii  # [1/m]
        self.max_iterations = max_iterations

        assert 0 < mix_fraction < 1, f"Mix fraction has to be larger than 0 and smaller than 1."
        self.mix_fraction = mix_fraction
        self.delta = delta

        self.success = False
        self.verbose = verbose

        if self.state.ion_core_radius is None:
            self.ion_core_radius = get_sigmac(
                ni=state.ion_number_density, Zi=state.ion_charge, Ti=state.ion_temperature
            )
            self.state.ion_core_radius = self.ion_core_radius
        else:
            self.ion_core_radius = self.state.ion_core_radius

    def get_ii_static_structure_factor(
        self, k, sf_model="HNC", pseudo_potential="YUKAWA", bridge_function="IYETOMI", return_full=False
    ):
        """
        Main run function to obtain the ion-ion static structure factor depending on the model inputs.

        Parameters:
            k (float/array): wave number in units of 1/m
            sf_model (str): model for the static structure factor
            pseudo_potential (str): ion-ion potential for the hnc solver
            brige_function (str): model used for the bridge function, only applies to the xHNC model
            return_full (bool): if true, will return all outputs including pair correlation function, does not work for MSA

        Returns:
            float/ array: will return full array of static structure factors or single value depending on the k-input type
        """
        if sf_model == "MSA":
            warnings.warn(f"Use {sf_model} your own risk. It is dogshit.")
            if return_full:
                print("Cannot return full outputs for the MSA. Only returning static structure factor.")
            return self.mean_spherical_approximation_ocp_ii(k)
        elif sf_model == "HNC":
            ks, rs, giir, hiir, Siik = self.hnc_ocp_ii(k, pseudo_potential)
        elif sf_model == "EXTENDED_HNC":
            ks, rs, giir, hiir, Siik = self.xhnc_ocp_ii(k, pseudo_potential, bridge_function)
        elif sf_model == "MODIFIED_HNC":
            warnings.warn(
                f"Model {sf_model} for the ion-ion static structure factor currently under construction. Please come back later."
            )
        else:
            raise NotImplementedError(
                f"Model {sf_model} for the ion-ion static structure factor is not known. Try HNC."
            )

        # interpolate static structure factor to output k grid
        interp_sf = interp1d(ks, Siik, fill_value="extrapolate")
        Siik_new = interp_sf(k)
        if return_full:
            r = np.linspace(rs[0], rs[-1], len(k))
            interp_giir = interp1d(rs, giir)
            giir_new = interp_giir(r)
            interp_hiir = interp1d(rs, hiir)
            hiir_new = interp_hiir(r)
            return k, r, giir_new, hiir_new, Siik_new
        else:
            return Siik_new

    def get_screened_ii_static_structure_factor(
        self,
        k,
        lfc=0.0,
        sf_model="HNC",
        pseudo_potential="YUKAWA",
        bridge_function="IYETOMI",
        screening_model="HARD_CORE",
        return_full=False,
    ):
        """
        Following Gregori et al., HEDP 3 (2007): Eq. (32 - 34), this returns a screened OCP static structure factor.

        Parameters:
            k (float/array): wave number in units of 1/m
            sf_model (str): model for the static structure factor
            pseudo_potential (str): ion-ion potential for the hnc solver
            brige_function (str): model used for the bridge function, only applies to the xHNC model
            screening_model (str): left-over option, not applied here and should be ignored
            return_full (bool): if true, will return all outputs including pair correlation function, does not work here

        Returns:
            float/ array: will return full array of static structure factors or single value depending on the k-input type

        """
        Sii = self.get_ii_static_structure_factor(
            k=k,
            sf_model=sf_model,
            pseudo_potential=pseudo_potential,
            bridge_function=bridge_function,
            return_full=False,
        )
        ion_core_radius = self.ion_core_radius
        kappa_i = np.sqrt(
            self.state.charge_state
            * self.state.free_electron_number_density
            * ELEMENTARY_CHARGE**2
            / (VACUUM_PERMITTIVITY * BOLTZMANN_CONSTANT * self.state.ion_temperature)
        )
        kappa_e = np.sqrt(
            self.state.free_electron_number_density
            * ELEMENTARY_CHARGE**2
            / (VACUUM_PERMITTIVITY * BOLTZMANN_CONSTANT * self.state.electron_temperature)
        )
        S_ee0 = k**2 / (k**2 - kappa_e**2 * (1 - lfc))
        screening_correction = (
            -(kappa_i**2) / k**2 * (np.cos(k * ion_core_radius / 2)) ** 2 * kappa_e**2 / k**2 * S_ee0
        )
        Sii_screened = Sii / (1 + screening_correction * Sii)
        return Sii_screened

    def _hnc_ii_pseudopotential(
        self,
        k,
        r,
        Q,
        alpha,
        ion_core_radius=None,
        kappa_e=None,
        csd_core_charges=None,
        csd_parameters=None,
        srr_core_power=None,
        srr_sigma=None,
        model="YUKAWA",
    ):
        """
        Function to return the ion-ion pseudo-potential used in the HNC solver in both r and k space.

        Parameters:
            k (array): k grid in units of 1/m
            r (array): r grid in units of m
            Q (float): ion charge state
            alpha (float): screening constant used in the Yukawa and Debye-Huckel models
            ion_core_radius (float): effective ion radius used in SRR model
            kappa_e (float): inverse screening length
            csd_core_charges (array): charge states used in the CSD model
            csd_parameters (array): scale length used to control switch in CSD model
            srr_core_power (float): exponent in the SRR model
            srr_sigma (float): amptitude scaling factor in the SRR model
            model (str): option to control the potential applied

        Returns:
            array: pseudo-potential in r-space
            array: pseudo-potential in k-space
        """
        if model == "YUKAWA":
            return yukawa_r(Q, Q, r, alpha), yukawa_k(Q, Q, k, alpha)
        elif model == "DEBYE_HUCKEL":
            return debye_huckel_r(Q, Q, r, alpha, kappa_e), debye_huckel_k(Q, Q, k, alpha, kappa_e)
        elif model == "DEUTSCH":
            return deutsch_r(Q, Q, r, alpha), deutsch_k(Q, Q, k, alpha)
        elif model == "KELBG":
            return kelbg_r(Q, Q, r, alpha), kelbg_k(Q, Q, k, alpha)
        elif model == "SRR":
            warnings.warn(f"Model {model} should not be used.")
            return short_range_screening_r(
                self.state.ion_temperature, srr_core_power, ion_core_radius, srr_sigma, kappa_e
            ), np.zeros_like(k)
        elif model == "CSD":
            return charge_switching_debye_r(
                Q, Q, r, csd_parameters[0], csd_parameters[1], csd_core_charges[0], csd_core_charges[1], kappa_e
            ), charge_switching_debye_k(
                Q, Q, k, csd_parameters[0], csd_parameters[1], csd_core_charges[0], csd_core_charges[1], kappa_e
            )
        else:
            raise NotImplementedError(f"Pseudo-potential model {model} in the HNC solver not recognized.")

    def _hnc_bridge_function(self, rs, Rii, Gamma, model="IYETOMI"):
        """
        Returns the bridge function used in the xHNC solver.

        Parameters:
            rs (array): r-grid in units of m
            Rii (float): ion core radius in units of m
            Gamma (float): coupling strength, non-dimensional
            model (str): option to choose the bridge function model, currently the only option is IYETOMI

        Returns:
            array: values for the bridge function in r-space
        """
        if Gamma < 5:
            print(
                f"Don`t be a dumb-dumb. You cannot apply bridge functions for weakly coupled plasmas. Reverting back to B = 0."
            )
            return np.zeros_like(rs), np.zeros_like(rs)
        if model == "IYETOMI":
            return iyetomi_bridge_function(rs=rs, Rii=Rii, Gamma=Gamma)
        else:
            raise NotImplementedError(f"Bridge function {model} not recognized.")

    def mean_spherical_approximation_ocp_ii(self, k):
        """
        Analytic model for the static structure factor as described in Gregori et al., HEDP 3 (2007).

        Parameters:
            k (float/array): wave number in units of 1/m

        Returns:
            array/float: ion-ion static structure factors depending on the k-input
        """
        ni = self.state.ion_number_density
        Zi = self.state.ion_charge
        Ti = self.state.ion_temperature
        sigma_c = self.state.ion_core_radius

        eta = PI / 6 * ni * sigma_c**3
        gamma = Zi**2 * ELEMENTARY_CHARGE**2 / (4 * PI * VACUUM_PERMITTIVITY * sigma_c * BOLTZMANN_CONSTANT * Ti)
        chi = np.sqrt(24 * eta * gamma)

        q = sigma_c * k
        sinq = np.sin(q)
        cosq = np.cos(q)
        qcosq = q * cosq
        qsinq = q * sinq

        y0 = (
            -(((1 + 2 * eta) ** 2) / (1 - eta) ** 4)
            + h0(eta, chi) ** 2 / (4 * (1 - eta) ** 2)
            - (1 + eta) * h0(eta, chi) * chi / (12 * eta)
            - (5 + eta**2) * chi**2 / (60 * eta)
        )
        y1 = 6 * eta * h1(eta, chi) ** 2
        y2 = chi**2 / 6
        y3 = eta / 2 * (y0 + chi**2 * h2(eta, chi))
        y4 = eta * chi**2 / 60
        Cii = (
            24
            * eta
            / q**6
            * (
                y0 * q**3 * (sinq - qcosq)
                + y1 * q**2 * (2 * qsinq - (q**2 - 2) * cosq - 2)
                + y2 * q * ((3 * q**2 - 6) * sinq - (q**2 - 6) * qcosq)
                + y3 * ((4 * q**2 - 24) * qsinq - (q**4 - 12 * q**2 + 24) * cosq + 24)
                + y4
                * (6 * (q**4 - 20 * q**2 + 120) * qsinq - (q**6 - 30 * q**4 + 360 * q**2 - 720) * cosq - 720)
                / q**2
                - gamma * q**4 * cosq
            )
        )
        S_ii_ocp = 1 / (1 - Cii)
        self.success = True
        return S_ii_ocp

    def hnc_ocp_ii(self, k, pseudo_potential):
        """
        HNC solver to obtain ion-ion static structure factors from the HNC closure to the Orstein-Zernike equation.

        Parameters:
            k (float/array): wave number in units of 1/m, not actually used here as the grid is determined internally
            pseudo_potential (str): option to choose the ion-ion potential

        Returns:
            array: k-grid used in the solver in units of 1/m
            array: r-grid used in the solver in units of m
            array: pair correlation function, non-dimensional
            array: total correlation function, non-dimensional
            array: ion-ion static structure factors, non-dimensional
        """
        Ti = self.state.ion_temperature
        Zi = self.state.ion_charge
        ni = self.state.ion_number_density
        Rii = self.Rii
        beta = 1 / (BOLTZMANN_CONSTANT * Ti)  # [1/J]
        n = self.n  # per themis [#]

        alpha = self.alpha  # 2 / Rii  # [1/m]

        r0 = 1.0e-2 * Rii  # [m]
        rf = 1.0e2 * Rii  # [m]
        dr = (rf - r0) / n
        k0 = 1.0e-4 / BOHR_RADIUS  # * aB
        dk = np.pi / (n * dr)  # [1/m] as it should be [1/m],
        kf = k0 + n * dk
        rs = np.linspace(r0, rf, n)  # [m]
        ks = np.linspace(k0, kf, n)  # [1/m]

        Q = Zi  # [C]
        kappa_e = self.state.screening_length(
            ELECTRON_MASS,
            1,
            self.state.electron_temperature,
            self.state.free_electron_number_density,
        ).real

        # use thermodynamically normalized potential
        Us = self._hnc_ii_pseudopotential(
            Q=Q,
            r=rs,
            alpha=alpha,
            k=ks,
            kappa_e=kappa_e,
            ion_core_radius=self.state.ion_core_radius,
            srr_core_power=self.state.sec_power,
            srr_sigma=self.state.srr_sigma,
            csd_core_charges=[self.state.csd_core_charge, self.state.csd_core_charge],
            csd_parameters=[self.state.csd_parameter, self.state.csd_parameter],
            model=pseudo_potential,
        )
        Us_rs = beta * Us[0]  # [ ]
        Ul_ks = beta * Us[1]  # [m^3] ???

        cs0_rs = -Us_rs  # [ ]
        Ns0_rs = np.zeros_like(cs0_rs)  # [ ]

        # Set up variables for HNC solver
        max_iterations = self.max_iterations
        mix_fraction = self.mix_fraction
        delta = self.delta

        converged = False
        i = 0

        i = 0
        while i < max_iterations:

            if i == 0:
                Ns_rs = Ns0_rs.copy()
                Ns_rs_prev = Ns0_rs.copy()
                g_rs_prev = np.exp(Ns_rs_prev - Us_rs)
                cs_rs = cs0_rs.copy()

            # total correlation function
            h_rs = g_rs_prev - 1

            # direct correlation function in k-space
            cs_ks = forward_transform_fft(yr=cs_rs, dr=dr, dk=dk, r=rs, k=ks)
            c_ks = cs_ks - Ul_ks
            # this will need to be replaced by a matrix inversion
            h_ks = c_ks / (1 - ni * c_ks)

            # indirect correlation function
            Ns_ks = h_ks - cs_ks
            Ns_rs_new = inverse_transform_fft(yk=Ns_ks, dr=dr, dk=dk, r=rs, k=ks)

            # mixing
            Ns_rs[:] = (1 - mix_fraction) * Ns_rs_new + mix_fraction * Ns_rs_prev

            # calculate new g(r)
            g_rs_new = np.exp(Ns_rs - Us_rs)

            if np.any(g_rs_new) == np.nan:
                print(f"Careful, we have naan in the HNC solver. Try increasing the mix fraction.")

            # check convergence
            converged = np.sum((g_rs_prev - g_rs_new) ** 2) < delta
            if converged:
                self.success = True
                if self.verbose:
                    print(f"HNC solver converged after {i} iterations.")
                break

            # save variables for next iteration
            Ns_rs_prev[:] = Ns_rs
            g_rs_prev[:] = g_rs_new
            cs_rs[:] = h_rs - Ns_rs

            i += 1
        else:
            print(
                f"Exited the HNC solver after {max_iterations} iterations without convergence. Try increasing the max iterations."
            )
            self.success = False

        giir = g_rs_new

        hiir = giir - 1.0
        Siik = 1 + ni * forward_transform_fft(yr=hiir, dr=dr, dk=dk, r=rs, k=ks)  # [#]

        # extrapolation in log10 space (for stability) -> inspired by THEMIS
        Siik[1 - 1] = 1.0 / 10 ** (
            np.log10(1.0 / Siik[4 - 1])
            + (np.log10(1.0 / Siik[6 - 1]) - np.log10(1.0 / Siik[4 - 1]))
            * (np.log10(ks[1 - 1]) - np.log10(ks[4 - 1]))
            / (np.log10(ks[6 - 1]) - np.log10(ks[4 - 1]))
        )
        Siik[2 - 1] = 1.0 / 10 ** (
            np.log10(1.0 / Siik[4 - 1])
            + (np.log10(1.0 / Siik[6 - 1]) - np.log10(1.0 / Siik[4 - 1]))
            * (np.log10(ks[2 - 1]) - np.log10(ks[4 - 1]))
            / (np.log10(ks[6 - 1]) - np.log10(ks[4 - 1]))
        )
        return ks, rs, giir, hiir, Siik

    def xhnc_ocp_ii(self, k, pseudo_potential, bridge_function):
        """
        Extended HNC solver to obtain ion-ion static structure factors from the HNC closure to the Orstein-Zernike equation
        including bridge functions.

        Parameters:
            k (float/array): wave number in units of 1/m, not actually used here as the grid is determined internally
            pseudo_potential (str): option to choose the ion-ion potential
            bridge_function (str): option to choose the bridge function

        Returns:
            array: k-grid used in the solver in units of 1/m
            array: r-grid used in the solver in units of m
            array: pair correlation function, non-dimensional
            array: total correlation function, non-dimensional
            array: ion-ion static structure factors, non-dimensional
        """
        Zi = self.state.ion_charge
        ni = self.state.ion_number_density
        Rii = self.Rii
        beta = self.beta  # 1 / (BOLTZMANN_CONSTANT * Ti)  # [1/J]
        n = self.n  # 8192  # per themis [#]

        # Rii = mean_sphere_radius(ni)  # [m]
        alpha = self.alpha  # 2 / Rii  # [1/m]

        r0 = 1.0e-2 * Rii  # [m]
        rf = 1.0e2 * Rii  # [m]
        dr = (rf - r0) / n
        k0 = 1.0e-4 / BOHR_RADIUS  # * aB
        dk = np.pi / (n * dr)  # [1/m] as it should be [1/m],
        kf = k0 + n * dk
        rs = np.linspace(r0, rf, n)  # [m]
        ks = np.linspace(k0, kf, n)  # [1/m]

        Q = Zi  # [C]

        kappa_e = self.state.screening_length(
            ELECTRON_MASS,
            1,
            self.state.electron_temperature,
            self.state.free_electron_number_density,
        ).real

        # # use thermodynamically normalized potential
        Us = self._hnc_ii_pseudopotential(
            Q=Q,
            r=rs,
            alpha=alpha,
            k=ks,
            kappa_e=kappa_e,
            ion_core_radius=self.state.ion_core_radius,
            srr_core_power=self.state.sec_power,
            srr_sigma=self.state.srr_sigma,
            csd_core_charges=[self.state.csd_core_charge, self.state.csd_core_charge],
            csd_parameters=[self.state.csd_parameter, self.state.csd_parameter],
            model=pseudo_potential,
        )
        Us_rs = beta * Us[0]  # [ ]
        Ul_ks = beta * Us[1]  # [m^3] ???

        cs0_rs = -Us_rs  # [ ]
        Ns0_rs = np.zeros_like(cs0_rs)  # should be [ ]

        # Set up variables for HNC solver
        max_iterations = self.max_iterations
        mix_fraction = self.mix_fraction
        delta = self.delta

        converged = False
        i = 0
        Gamma = self.state.coupling_parameter(Za=Zi, beta=beta, da=Rii)
        _, Biir = self._hnc_bridge_function(rs=rs, Rii=Rii, Gamma=Gamma, model=bridge_function)

        while i < max_iterations:

            if i == 0:
                Ns_rs = Ns0_rs.copy()
                Ns_rs_prev = Ns0_rs.copy()
                g_rs_prev = np.exp(Ns_rs_prev - Us_rs + Biir)
                cs_rs = cs0_rs.copy()

            # total correlation function
            h_rs = g_rs_prev - 1

            # direct correlation function in k-space
            cs_ks = forward_transform_fft(yr=cs_rs, dr=dr, dk=dk, r=rs, k=ks)
            c_ks = cs_ks - Ul_ks
            # this will need to be replaced by a matrix inversion
            h_ks = c_ks / (1 - ni * c_ks)

            # indirect correlation function
            Ns_ks = h_ks - cs_ks
            Ns_rs_new = inverse_transform_fft(yk=Ns_ks, dr=dr, dk=dk, r=rs, k=ks)

            # mixing
            Ns_rs[:] = (1 - mix_fraction) * Ns_rs_new + mix_fraction * Ns_rs_prev

            # calculate new g(r)
            g_rs_new = np.exp(Ns_rs - Us_rs + Biir)

            if np.any(g_rs_new) == np.nan:
                print(f"Careful, we have naan in the HNC solver. Try increasing the mix fraction.")

            # check convergence
            converged = np.sum((g_rs_prev - g_rs_new) ** 2) < delta
            if converged:
                self.success = True
                if self.verbose:
                    print(f"HNC solver converged after {i} iterations.")
                break

            # save variables for next iteration
            Ns_rs_prev[:] = Ns_rs
            g_rs_prev[:] = g_rs_new
            cs_rs[:] = h_rs - Ns_rs

            i += 1
        else:
            self.success = False
            print(
                f"HNC solver exited after {max_iterations} iterations without convergence. Increase the max iterations."
            )

        giir = g_rs_new

        hiir = giir - 1.0
        Siik = 1 + ni * forward_transform_fft(yr=hiir, dr=dr, dk=dk, r=rs, k=ks)  # [#]

        # extrapolation in log10 space (for stability) -> inspired by THEMIS
        Siik[1 - 1] = 1.0 / 10 ** (
            np.log10(1.0 / Siik[4 - 1])
            + (np.log10(1.0 / Siik[6 - 1]) - np.log10(1.0 / Siik[4 - 1]))
            * (np.log10(ks[1 - 1]) - np.log10(ks[4 - 1]))
            / (np.log10(ks[6 - 1]) - np.log10(ks[4 - 1]))
        )
        Siik[2 - 1] = 1.0 / 10 ** (
            np.log10(1.0 / Siik[4 - 1])
            + (np.log10(1.0 / Siik[6 - 1]) - np.log10(1.0 / Siik[4 - 1]))
            * (np.log10(ks[2 - 1]) - np.log10(ks[4 - 1]))
            / (np.log10(ks[6 - 1]) - np.log10(ks[4 - 1]))
        )
        return ks, rs, giir, hiir, Siik


## ------------------------------ ##
## Multi-component implementation ##
## ------------------------------ ##


class MCPStaticStructureFactor:
    """
    Class containing the multi-component static structure factor calculations.

    Attributes:
        overlord_state (PlasmaState): mean plasma state
        state (PlasmaState): container holding all plasma parameters
        max_iterations (int): maximum number of iterations in the HNC solver
        mix_fraction (float): number between 0 and 1, values closer to 1 mean less of the new iterated solution in the HNC solver is included
        delta (float): tolerance to check convergence of the HNC result
        n (int): number of points in the HNC grid
    """

    def __init__(
        self,
        overlord_state: PlasmaState,
        states: np.array,
        max_iterations=5000,
        mix_fraction=0.8,
        delta=1.0e-6,
        n=8192,
        verbose=False,
    ):

        self.overlord_state = overlord_state
        self.beta = 1 / (BOLTZMANN_CONSTANT * overlord_state.ion_temperature)  # [1/J]
        self.n = n  # per themis [#]

        self.Rii = self.overlord_state.mean_sphere_radius(number_density=self.overlord_state.ion_number_density)  # [m]
        self.alpha = 2 / self.Rii  # [1/m]
        self.nspecies = len(states)

        self.a = self.b = len(states)
        self.nis = []
        self.xs = []
        self.Qs = []
        for i in range(0, len(states)):
            self.nis.append(states[i].ion_number_density)
            self.Qs.append(states[i].ion_charge)
            # TODO(Hannah): this should be done in the setup, ideally... and not repeated here
            self.xs.append(states[i].mass_density / overlord_state.mass_density)

        self.max_iterations = max_iterations
        self.mix_fraction = mix_fraction
        self.delta = delta
        self.states = states
        self.success = False
        self.verbose = verbose

    def get_ab_static_structure_factor(self, k, sf_model="HNC", pseudo_potential="DEBYE_HUCKEL", return_full=False):
        """
         Main run function to obtain the multi-component ion-ion static structure factor depending on the model inputs.

        Parameters:
            k (float/array): wave number in units of 1/m
            sf_model (str): model for the static structure factor
            pseudo_potential (str): ion-ion potential for the hnc solver
            brige_function (str): model used for the bridge function, only applies to the xHNC model
            return_full (bool): if true, will return all outputs including pair correlation function, does not work for MSA

        Returns:
            float/ array: will return full array of static structure factors or single value depending on the k-input type
        """
        # I think the HNC mix fraction et al., variables should be inputs here
        if sf_model == "MSA":
            raise NotImplementedError("The MSA has not been implemented for multi-component systems")
        elif sf_model == "SVT":
            warnings.warn(f"Model {sf_model} currently under construction. Please come back later.")
        elif sf_model == "HNC":
            ks, rs, giir, hiir, Sabs = self.hnc_ab_ss(k, pseudo_potential)
        else:
            raise NotImplementedError(
                f"Model {sf_model} for the multi-species ion-ion static structure factor not recognized. Try HNC."
            )

        # consider switching out the interpolation with something like np.interpolate which is meant to be faster
        interp_sf = interp1d(ks, Sabs, axis=-1, kind="linear")
        Sabs_new = interp_sf(k)
        if return_full:
            r = np.linspace(rs[0], rs[-1], len(k))
            interp_giir = interp1d(rs, giir, axis=-1, kind="linear")
            giir_new = interp_giir(r)
            interp_hiir = interp1d(rs, hiir, axis=-1, kind="linear")
            hiir_new = interp_hiir(r)
            return k, r, giir_new, hiir_new, Sabs_new
        else:
            return Sabs_new

    def _hnc_pseudopotential(
        self,
        k,
        r,
        Qa,
        Qb,
        alpha,
        ion_core_radius=None,
        kappa_e=None,
        csd_core_charges=None,
        csd_parameters=None,
        srr_core_power=None,
        srr_sigma=None,
        model="YUKAWA",
    ):
        """
        Function to return the ion-ion pseudo-potential used in the HNC solver in both r and k space.

        Parameters:
            k (array): k grid in units of 1/m
            r (array): r grid in units of m
            Q (float): ion charge state
            alpha (float): screening constant used in the Yukawa and Debye-Huckel models
            ion_core_radius (float): effective ion radius used in SRR model
            kappa_e (float): inverse screening length
            csd_core_charges (array): charge states used in the CSD model
            csd_parameters (array): scale length used to control switch in CSD model
            srr_core_power (float): exponent in the SRR model
            srr_sigma (float): amptitude scaling factor in the SRR model
            model (str): option to control the potential applied

        Returns:
            array: pseudo-potential in r-space
            array: pseudo-potential in k-space
        """
        if model == "COULOMB":
            return coulomb_r(Qa, Qb, r), coulomb_k(Qa, Qb, k)
        elif model == "YUKAWA":
            return yukawa_r(Qa, Qb, r, alpha), yukawa_k(Qa, Qb, k, alpha)
        elif model == "DEBYE_HUCKEL":
            return debye_huckel_r(Qa, Qb, r, alpha, kappa_e), debye_huckel_k(Qa, Qb, k, alpha, kappa_e)
        elif model == "DEUTSCH":
            return deutsch_r(Qa, Qb, r, alpha), deutsch_k(Qa, Qb, k, alpha)
        elif model == "KELBG":
            return kelbg_r(Qa, Qb, r, alpha), kelbg_k(Qa, Qb, k, alpha)
        elif model == "SRR":
            warnings.warn(f"Model {model} should not be used.")
            return short_range_screening_r(
                self.overlord_state.ion_temperature, srr_core_power, ion_core_radius, srr_sigma, kappa_e
            ), np.zeros_like(k)
        elif model == "CSD":
            return charge_switching_debye_r(
                Qa, Qb, r, csd_parameters[0], csd_parameters[1], csd_core_charges[0], csd_core_charges[1], kappa_e
            ), charge_switching_debye_k(
                Qa, Qb, k, csd_parameters[0], csd_parameters[1], csd_core_charges[0], csd_core_charges[1], kappa_e
            )
        else:
            raise NotImplementedError(f"Pseudo-potential model {model} in the HNC solver not recognized.")

    def hnc_ab_ss(self, k, pseudo_potential):
        """
        HNC solver for a multi-component system to obtain ion-ion static structure factors from the HNC closure to the Orstein-Zernike equation.

        Parameters:
            k (float/array): wave number in units of 1/m, not actually used here as the grid is determined internally
            pseudo_potential (str): option to choose the ion-ion potential

        Returns:
            array: k-grid used in the solver in units of 1/m
            array: r-grid used in the solver in units of m
            array: pair correlation function, non-dimensional
            array: total correlation function, non-dimensional
            array: ion-ion static structure factors, non-dimensional
        """
        beta = self.beta  # [1/J]
        n = self.n  # 8192  # per themis [#]

        alpha = self.alpha  # 2 / Rii  # [1/m]

        a = b = self.nspecies
        Qs = np.asarray(self.Qs, dtype=float)  # [C]

        kappa_e = self.overlord_state.screening_length(
            ELECTRON_MASS,
            1,
            self.overlord_state.electron_temperature,
            self.overlord_state.free_electron_number_density,
        ).real
        # Set up grid
        r0 = 1.0e-3 * BOHR_RADIUS  # [m]
        rf = 1.0e2 * BOHR_RADIUS  # [m]
        dr = (rf - r0) / n
        dk = np.pi / (n * dr)  # [1/m] as it should be [1/m],
        kf = r0 + n * dk
        rs = np.linspace(r0, rf, n)  # [m]
        ks = np.linspace(r0, kf, n)  # [1/m]

        # Set up initial matrices
        D = np.zeros((a, b))  # density matrix
        nis = self.nis
        I = np.eye(a)

        Us_rs = np.zeros((b, a, n))
        Ul_ks = np.zeros((b, a, n))

        # populate potentials: dimensionless
        for n1 in range(b):
            for n2 in range(a):
                Uab = self._hnc_pseudopotential(
                    k=ks,
                    Qa=Qs[n2],
                    Qb=Qs[n1],
                    r=rs,
                    alpha=alpha,
                    kappa_e=kappa_e,
                    ion_core_radius=self.states[n1].ion_core_radius,
                    srr_core_power=self.overlord_state.sec_power,
                    srr_sigma=self.overlord_state.srr_sigma,
                    csd_core_charges=[self.states[n1].csd_core_charge, self.states[n2].csd_core_charge],
                    csd_parameters=[self.states[n1].csd_parameter, self.states[n2].csd_parameter],
                    model=pseudo_potential,
                )
                Us_rs[n1, n2, :] = beta * Uab[0]
                Ul_ks[n1, n2, :] = beta * Uab[1]
                if n1 == n2:
                    # Populate density matrix
                    D[n1, n2] = nis[n1]  # [m^{-3}]

        converged = False
        i = 0

        Ns0_rs = np.zeros_like(Us_rs)  # [ ]
        cs0_rs = Us_rs  # [ ]

        prefactor_forward = 2 * np.pi * dr / ks[1:]
        prefactor_inverse = (dk / (2 * np.pi) ** 2) / rs[1:]

        max_iterations = self.max_iterations
        mix_fraction = self.mix_fraction
        delta = self.delta

        # epsilon = 1.0e-14
        while i < max_iterations:

            if i == 0:
                Ns_rs = Ns0_rs.copy()  # [ ]
                Ns_rs_prev = Ns0_rs.copy()  # [ ]
                g_rs_prev = np.exp(Ns_rs_prev - Us_rs)  # [ ]
                cs_rs = cs0_rs.copy()  # [ ]

            # total correlation function
            h_rs = g_rs_prev - 1  # [ ]

            cs_ks = forward_transform_fftn(yr=cs_rs, r=rs, norm=prefactor_forward)  #  [ ]
            c_ks = cs_ks - Ul_ks  # [ ]

            # Matrix inversion -> note I'm skipping the first element
            M = I[..., None] - D @ c_ks[..., 1:]
            M = np.moveaxis(M, -1, 0)
            c_ks_temp = np.moveaxis(c_ks[..., 1:], -1, 0)
            h_ks = np.linalg.solve(M, c_ks_temp)
            h_ks = np.moveaxis(h_ks, 0, -1)
            # Setting the first element to -1.0, this is wrong, but works for now
            h_ks = np.insert(h_ks, 0, -1.0, axis=-1)

            # indirect correlation function
            Ns_ks = h_ks - cs_ks
            Ns_rs_new = inverse_transform_fftn(yk=Ns_ks, k=ks, norm=prefactor_inverse)

            # Update new indirect correlation function for next iteration using the mix fraction
            Ns_rs = (1 - mix_fraction) * Ns_rs_new + mix_fraction * Ns_rs_prev

            g_rs_new = np.exp(Ns_rs - Us_rs)

            if np.any(np.isnan(g_rs_new)) or np.any(np.isnan(c_ks)) or np.any(np.isnan(cs_rs)):
                self.success = False
                print(f"\nCareful, we have naan at i={i}! Exciting.\n")
                break

            converged = np.sum((g_rs_prev - g_rs_new) ** 2) < delta
            if converged:
                self.success = True
                if self.verbose:
                    print(f"HNC solver converged after {i} iterations.")
                break

            # save variables for next iteration
            Ns_rs_prev[:] = Ns_rs
            g_rs_prev[:] = g_rs_new
            cs_rs[:] = h_rs - Ns_rs

            i += 1
        else:
            self.success = False
            print(f"Exited after {max_iterations} iterations without convergence.")

        giir = g_rs_new

        hiir = giir - 1.0

        Sabs = np.zeros((a, b, n))
        # This is only done once, so it could probably be sped up by generalizing the Fourier Transform for matrices, but it's not a priority right now
        for n1 in range(a):
            for n2 in range(b):
                if n1 == n2:
                    Sabs[n1, n2, :] = 1 + np.sqrt(nis[n1] * nis[n2]) * forward_transform_fft(
                        yr=hiir[n1, n2, :], r=rs, k=ks, dk=dk, dr=dr
                    )
                else:
                    Sabs[n1, n2, :] = np.sqrt(nis[n1] * nis[n2]) * forward_transform_fft(
                        yr=hiir[n1, n2, :], r=rs, k=ks, dk=dk, dr=dr
                    )

                # extrapolate first few points in k-space -> inspired by THEMIS
                Sabs[n1, n2, 0] = Sabs[n1, n2, 2] + (Sabs[n1, n2, 3] - Sabs[n1, n2, 2]) * (ks[0] - ks[1]) / (
                    ks[2] - ks[1]
                )
        return ks, rs, giir, hiir, Sabs

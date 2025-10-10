# Ionic form factors
from models import ModelOptions
from plasma_state import PlasmaState
from scipy.interpolate import interp1d

from bridge_functions import *
from utils import forward_transform_fft, inverse_transform_fft, forward_transform_fftn, inverse_transform_fftn
from constants import BOHR_RADIUS, ELEMENTARY_CHARGE, PI, BOLTZMANN_CONSTANT, VACUUM_PERMITTIVITY
from potentials import *
from unit_conversions import *


from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import os


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

    def __init__(
        self,
        state: PlasmaState,
        ion_core_radius=None,
        max_iterations=5000,
        mix_fraction=0.8,
        delta=1.0e-8,
        n=8192,
    ):
        self.state = state
        self.beta = 1 / (BOLTZMANN_CONSTANT * state.ion_temperature)  # [1/J]
        self.n = n  # per themis [#]

        self.Rii = self.state.mean_sphere_radius(number_density=self.state.ion_number_density)  # [m]
        self.alpha = 2 / self.Rii  # [1/m]
        self.max_iterations = max_iterations
        self.mix_fraction = mix_fraction
        self.delta = delta

        if ion_core_radius is None:
            self.ion_core_radius = get_sigmac(
                ni=state.ion_number_density, Zi=state.ion_charge, Ti=state.ion_temperature
            )
        else:
            self.ion_core_radius = ion_core_radius

    def get_ii_static_structure_factor(
        self, k, sf_model="HNC", pseudo_potential="YUKAWA", bridge_function="IYETOMI", return_full=False
    ):
        if sf_model == "MSA":
            warnings.warn(f"Use {sf_model} your own risk. It is dogshit.")

            if return_full:
                print("Cannot return full outputs for the MSA. Only returning static structure factor.")
            return self.mean_spherical_approximation_ocp_ii(k)
        elif sf_model == "HNC":
            ks, rs, giir, hiir, Siik = self.hnc_ocp_ii(k, pseudo_potential)
        elif sf_model == "EXTENDED_HNC":
            ks, rs, giir, hiir, Siik = self.xhnc_ocp_ii(k, pseudo_potential, bridge_function)
        else:
            raise NotImplementedError(
                f"Model {sf_model} for the static structure factor not yet implemented. Try MSA :)"
            )

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
        Following Gregori et al., HEDP 3 (2007): Eq. (32 - 34)
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
            * self.state.electron_number_density
            * ELEMENTARY_CHARGE**2
            / (VACUUM_PERMITTIVITY * BOLTZMANN_CONSTANT * self.state.ion_temperature)
        )
        kappa_e = np.sqrt(
            self.state.electron_number_density
            * ELEMENTARY_CHARGE**2
            / (VACUUM_PERMITTIVITY * BOLTZMANN_CONSTANT * self.state.electron_temperature)
        )
        S_ee0 = k**2 / (k**2 - kappa_e**2 * (1 - lfc))
        screening_correction = (
            -(kappa_i**2) / k**2 * (np.cos(k * ion_core_radius / 2)) ** 2 * kappa_e**2 / k**2 * S_ee0
        )
        Sii_screened = Sii / (1 + screening_correction * Sii)
        return Sii_screened

    def _screening_correction(self, k, r):
        return

    def _hnc_ii_pseudopotential(self, k, r, Q, alpha, model="YUKAWA"):
        if model == "YUKAWA":
            return yukawa_r(Q, Q, r, alpha), yukawa_k(Q, Q, k, alpha)
        else:
            raise NotImplementedError(f"Pseudo-potential model {model} not recognized.")

    def _hnc_bridge_function(self, rs, Rii, Gamma, model="IYETOMI"):
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
        ni = self.state.ion_number_density
        Zi = self.state.ion_charge
        Ti = self.state.ion_temperature
        # TODO(Hannah): move the ion diameter to the plasma state
        sigma_c = self.ion_core_radius

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
        return S_ii_ocp

    def hnc_ocp_ii(self, k, pseudo_potential):
        Ti = self.state.ion_temperature
        Zi = self.state.ion_charge
        ni = self.state.ion_number_density
        Rii = self.Rii  # self.state.mean_sphere_radius(number_density=ni)
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

        # use thermodynamically normalized potential
        Us_rs = beta * self._hnc_ii_pseudopotential(Q=Q, r=rs, alpha=alpha, k=ks, model=pseudo_potential)[0]  # [ ]
        Ul_ks = (
            beta * self._hnc_ii_pseudopotential(Q=Q, r=rs, alpha=alpha, k=ks, model=pseudo_potential)[1]
        )  # [m^3] ???

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
                print(f"Careful, we have naan")

            # check convergence
            converged = np.sum((g_rs_prev - g_rs_new) ** 2) < delta
            if converged:
                print(f"Converged after {i} iterations.")
                break

            # save variables for next iteration
            Ns_rs_prev[:] = Ns_rs
            g_rs_prev[:] = g_rs_new
            cs_rs[:] = h_rs - Ns_rs

            i += 1
        else:
            print(f"Exited after {max_iterations} iterations without convergence.")

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
        # Ti = self.overlord_state.ion_temperature
        Zi = self.state.ion_charge
        ni = self.state.ion_number_density
        Rii = self.Rii  # self.overlord_state.mean_sphere_radius(number_density=ni)
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

        # # use thermodynamically normalized potential
        Us_rs = beta * self._hnc_ii_pseudopotential(Q=Q, r=rs, alpha=alpha, k=ks, model=pseudo_potential)[0]  # [ ]
        Ul_ks = (
            beta * self._hnc_ii_pseudopotential(Q=Q, r=rs, alpha=alpha, k=ks, model=pseudo_potential)[1]
        )  # [m^3] ???

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
                print(f"Careful, we have naan.")

            # check convergence
            converged = np.sum((g_rs_prev - g_rs_new) ** 2) < delta
            if converged:
                print(f"Converged after {i} iterations.")
                break

            # save variables for next iteration
            Ns_rs_prev[:] = Ns_rs
            g_rs_prev[:] = g_rs_new
            cs_rs[:] = h_rs - Ns_rs

            i += 1
        else:
            print(f"Exited after {max_iterations} iterations without convergence.")

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

    def __init__(
        self,
        overlord_state: PlasmaState,
        states: np.array,
        ion_core_radius=None,
        max_iterations=5000,
        mix_fraction=0.8,
        delta=1.0e-8,
        n=8192,
    ):
        self.overlord_state = overlord_state
        self.ion_core_radius = ion_core_radius  # 1 * BOHR_RADIUS  ## this needs to be moved to the plasma state
        self.beta = 1 / (BOLTZMANN_CONSTANT * overlord_state.ion_temperature)  # [1/J]
        self.n = n  # per themis [#]

        self.Rii = self.overlord_state.mean_sphere_radius(number_density=self.overlord_state.ion_number_density)  # [m]
        self.alpha = 2 / self.Rii  # [1/m]
        self.nspecies = len(states)

        self.a = self.b = len(states)
        self.nis = []
        self.xs = []
        self.Qs = []
        for i in (0, len(states) - 1):
            self.nis.append(states[i].ion_number_density)
            self.Qs.append(states[i].ion_charge)
            self.xs.append(states[i].mass_density / overlord_state.mass_density)

        self.max_iterations = max_iterations
        self.mix_fraction = mix_fraction
        self.delta = delta

    def get_ab_static_structure_factor(self, k, sf_model="HNC", pseudo_potential="YUKAWA", return_full=False):
        if sf_model == "MSA":
            raise NotImplementedError("The MSA has not been implemented for multi-component systems")
        elif sf_model == "HNC":
            ks, rs, giir, hiir, Sabs = self.hnc_ab_ss(k, pseudo_potential)
        else:
            raise NotImplementedError(f"Model {sf_model} not recognized. Try HNC.")

        # consider switching out the interpolation with something like np.interpolate which is meant to be faster
        interp_sf = interp1d(ks, Sabs)
        Sabs_new = interp_sf(k)
        if return_full:
            r = np.linspace(rs[0], rs[-1], len(k))
            interp_giir = interp1d(rs, giir)
            giir_new = interp_giir(r)
            interp_hiir = interp1d(rs, hiir)
            hiir_new = interp_hiir(r)
            return k, r, giir_new, hiir_new, Sabs_new
        else:
            return Sabs_new

    def _hnc_pseudopotential(self, k, r, Qa, Qb, alpha, model="YUKAWA"):
        if model == "YUKAWA":
            return yukawa_r(Qa, Qb, r, alpha), yukawa_k(Qa, Qb, k, alpha)
        else:
            raise NotImplementedError(f"Pseudo-potential model {model} not recognized.")

    def hnc_ab_ss(self, k, pseudo_potential):
        beta = self.beta  # [1/J]
        n = self.n  # 8192  # per themis [#]

        alpha = self.alpha  # 2 / Rii  # [1/m]

        a = b = self.nspecies
        Qs = np.asarray(self.Qs, dtype=float)  # [C]

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
        for n1 in (0, b - 1):
            for n2 in (0, a - 1):
                Uab = self._hnc_pseudopotential(k=ks, Qa=Qs[n2], Qb=Qs[n1], r=rs, alpha=alpha, model=pseudo_potential)
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

            # Matrix inversion
            Dc = D @ c_ks
            M = I[..., None] - Dc
            M = np.moveaxis(M, -1, 0)

            # try to filter out indices where the matrix is singular
            conds = np.linalg.cond(M)

            good_idx = np.where(conds < 1.0e12)[0]
            bad_idx = np.where(conds >= 1.0e12)[0]
            h_ks = np.empty_like(c_ks)
            if len(good_idx) > 0:
                M_good = M[good_idx]
                c_good = c_ks[..., good_idx].transpose(2, 0, 1)
                h_good = np.linalg.solve(M_good.transpose(0, 2, 1), c_good.transpose(0, 2, 1)).transpose(0, 2, 1)
                h_ks[..., good_idx] = h_good.transpose(1, 2, 0)
            for ik in bad_idx:
                h_ks[..., ik] = c_ks[..., ik] @ np.linalg.pinv(M[ik])

            # indirect correlation function
            Ns_ks = h_ks - cs_ks
            Ns_rs_new = inverse_transform_fftn(yk=Ns_ks, k=ks, norm=prefactor_inverse)

            # Update new indirect correlation function for next iteration using the mix fraction
            Ns_rs = (1 - mix_fraction) * Ns_rs_new + mix_fraction * Ns_rs_prev

            g_rs_new = np.exp(Ns_rs - Us_rs)

            if np.any(np.isnan(g_rs_new)) or np.any(np.isnan(c_ks)) or np.any(np.isnan(cs_rs)):
                print(f"\nCareful, we have naan at i={i}! Exciting.\n")
                break

            converged = np.sum((g_rs_prev - g_rs_new) ** 2) < delta
            if converged:
                print(f"Converged after {i} iterations.")
                break

            # save variables for next iteration
            Ns_rs_prev[:] = Ns_rs
            g_rs_prev[:] = g_rs_new
            cs_rs[:] = h_rs - Ns_rs

            i += 1
        else:
            print(f"Exited after {max_iterations} iterations without convergence.")

        giir = g_rs_new

        hiir = giir - 1.0

        Sabs = np.zeros((a, b, n))
        # This is only done once, so it could probably be sped up by generalizing the Fourier Transform for matrices, but it's not a priority right now
        for n1 in (0, a - 1):
            for n2 in (0, b - 1):
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


def test_ocp():
    r"""
    Comparison against K W\"unsch PhD Thesis (2011), Fig. 4.5
    """
    plt.style.use("~/Desktop/resources/plotting/poster.mplstyle")

    # Case 1: Gamma_ii = 12.3, Ti = 4 eV
    T = 4 * eV_TO_K
    Zi = 2
    rho = 498.16  # kg/m^3
    state = PlasmaState(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_state=Zi,
        atomic_mass=2,
        atomic_number=2,
        binding_energies=None,
    )
    ni = rho / (2 * amu_TO_kg)
    Rii = (3 / (4 * np.pi * ni)) ** (1 / 3)
    Rii1 = Rii
    beta = 1 / (BOLTZMANN_CONSTANT * T)
    Gamma = Zi**2 * ELEMENTARY_CHARGE**2 * beta / (4 * np.pi * VACUUM_PERMITTIVITY * Rii)
    print(f"Gamma1 = {Gamma}")

    sigma_c = 2.15 * BOHR_RADIUS

    k = np.linspace(1.0e-1 / BOHR_RADIUS, 10 / BOHR_RADIUS, 200)
    kernel = OCPStaticStructureFactor(state=state, ion_core_radius=sigma_c, max_iterations=1000)
    Sii_HNC = kernel.get_ii_static_structure_factor(k=k, sf_model="HNC")
    Sii_xHNC = kernel.get_ii_static_structure_factor(k=k, sf_model="EXTENDED_HNC")
    Sii_MSA = kernel.get_ii_static_structure_factor(k=k, sf_model="MSA")

    fn = "validation/static_sf/verification/Wuensch_Thesis_Fig4-5/T_4eV_Gamma_12.3_HNC-OCP.csv"  # os.path.join(DATA_DIR, f"T_{T*K_TO_eV:.0f}eV_Gamma_{Gamma:.1f}_HNC-OCP.csv")
    dat1 = np.genfromtxt(fn, delimiter=",")
    fn = "validation/static_sf/verification/Wuensch_Thesis_Fig4-5/T_4eV_Gamma_12.3_MSA-OCP.csv"  # os.path.join(DATA_DIR, f"T_{T*K_TO_eV:.0f}eV_Gamma_{Gamma:.1f}_HNC-OCP.csv")
    dat11 = np.genfromtxt(fn, delimiter=",")

    T = 20 * eV_TO_K
    Zi = 2
    rho = 498.16  # kg/m^3
    state = PlasmaState(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_state=Zi,
        atomic_mass=2,
        atomic_number=2,
        binding_energies=None,
    )
    ni = rho / (2 * amu_TO_kg)
    Rii = (3 / (4 * np.pi * ni)) ** (1 / 3)
    Rii2 = Rii
    beta = 1 / (BOLTZMANN_CONSTANT * T)
    Gamma2 = Zi**2 * ELEMENTARY_CHARGE**2 * beta / (4 * np.pi * VACUUM_PERMITTIVITY * Rii)
    print(f"Gamma2 = {Gamma2}")
    fn = "validation/static_sf/verification/Wuensch_Thesis_Fig4-5/T_20eV_Gamma_2.7_HNC-OCP.csv"
    dat2 = np.genfromtxt(fn, delimiter=",")
    fn = "validation/static_sf/verification/Wuensch_Thesis_Fig4-5/T_20eV_Gamma_2.7_MSA-OCP.csv"
    dat22 = np.genfromtxt(fn, delimiter=",")

    sigma_c = 1.5 * BOHR_RADIUS

    kernel = OCPStaticStructureFactor(state=state, ion_core_radius=sigma_c, max_iterations=5000, mix_fraction=0.9)
    Sii_HNC2 = kernel.get_ii_static_structure_factor(k=k, sf_model="HNC")
    # Sii_xHNC2 = kernel.get_ii_static_structure_factor(k=k, sf_model="EXTENDED_HNC")
    Sii_MSA2 = kernel.get_ii_static_structure_factor(k=k, sf_model="MSA")

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(k * BOHR_RADIUS, Sii_MSA, label=f"MSA, Gamma={Gamma:.1f}", c="orange", ls="--")
    ax.plot(k * BOHR_RADIUS, Sii_HNC, label=f"HNC, Gamma={Gamma:.1f}", c="crimson", ls="-.")
    ax.plot(k * BOHR_RADIUS, Sii_xHNC, label=f"xHNC, Gamma={Gamma:.1f}", c="crimson", ls=":")
    ax.scatter(
        dat1[:, 0] * BOHR_RADIUS / Rii1, dat1[:, 1], label=f"HNC - Wuensch, Gamma={Gamma:.1f}", c="brown", marker="x"
    )
    ax.scatter(
        dat11[:, 0] * BOHR_RADIUS / Rii1,
        dat11[:, 1],
        label=f"MSA - Wuensch, Gamma={Gamma:.1f}",
        c="orange",
        marker="o",
    )
    ax.plot(k * BOHR_RADIUS, Sii_HNC2, label=f"HNC, Gamma={Gamma2:.1f}", c="navy", ls="-.")
    # plt.plot(k * BOHR_RADIUS, Sii_xHNC2, label="xHNC", c="navy", ls=":")
    ax.plot(k * BOHR_RADIUS, Sii_MSA2, label=f"MSA, Gamma={Gamma2:.1f}", c="dodgerblue", ls="--")
    ax.scatter(
        dat2[:, 0] * BOHR_RADIUS / Rii2, dat2[:, 1], label=f"HNC - Wuensch, Gamma={Gamma2:.1f}", c="black", marker="x"
    )
    ax.scatter(
        dat22[:, 0] * BOHR_RADIUS / Rii2,
        dat22[:, 1],
        label=f"MSA - Wuensch, Gamma={Gamma2:.1f}",
        c="dodgerblue",
        marker="o",
    )
    ax.legend()
    ax.axhline(1.0, lw=1, ls=":", c="gray")
    ax.set_xlim(-0.1, 6.0)
    ax.set_xlabel(r"$k$ [$a_B^{-1}$]")
    ax.set_ylabel(r"$S_{ii}(k)$")
    plt.show()
    fig.savefig(f"static_ii_sf_comparison.pdf", dpi=200)


def test_screened_ocp():
    r"""
    Comparison against K W\"unsch PhD Thesis (2011), Fig. 4.5
    """
    # plt.style.use("~/Desktop/resources/plotting/poster.mplstyle")

    # Case 1: Gamma_ii = 12.3, Ti = 4 eV
    T = 4 * eV_TO_K
    Zi = 2
    rho = 498.16  # kg/m^3
    state = PlasmaState(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_state=Zi,
        atomic_mass=2,
        atomic_number=2,
        binding_energies=None,
    )
    ni = rho / (2 * amu_TO_kg)
    Rii = (3 / (4 * np.pi * ni)) ** (1 / 3)
    Rii1 = Rii
    beta = 1 / (BOLTZMANN_CONSTANT * T)
    Gamma = Zi**2 * ELEMENTARY_CHARGE**2 * beta / (4 * np.pi * VACUUM_PERMITTIVITY * Rii)
    print(f"Gamma1 = {Gamma}")

    sigma_c = 2.15 * BOHR_RADIUS

    k = np.linspace(1.0e-1 / BOHR_RADIUS, 10 / BOHR_RADIUS, 200)
    kernel = OCPStaticStructureFactor(state=state, ion_core_radius=sigma_c, max_iterations=1000)
    Sii_HNC = kernel.get_screened_ii_static_structure_factor(k=k, sf_model="HNC")
    Sii_xHNC = kernel.get_screened_ii_static_structure_factor(k=k, sf_model="EXTENDED_HNC")
    Sii_MSA = kernel.get_screened_ii_static_structure_factor(k=k, sf_model="MSA")

    fn = "validation/static_sf/verification/Wuensch_Thesis_Fig4-5/T_4eV_Gamma_12.3_HNC-OCP.csv"  # os.path.join(DATA_DIR, f"T_{T*K_TO_eV:.0f}eV_Gamma_{Gamma:.1f}_HNC-OCP.csv")
    dat1 = np.genfromtxt(fn, delimiter=",")
    fn = "validation/static_sf/verification/Wuensch_Thesis_Fig4-5/T_4eV_Gamma_12.3_MSA-OCP.csv"  # os.path.join(DATA_DIR, f"T_{T*K_TO_eV:.0f}eV_Gamma_{Gamma:.1f}_HNC-OCP.csv")
    dat11 = np.genfromtxt(fn, delimiter=",")

    T = 20 * eV_TO_K
    Zi = 2
    rho = 498.16  # kg/m^3
    state = PlasmaState(
        electron_temperature=T,
        ion_temperature=T,
        mass_density=rho,
        charge_state=Zi,
        atomic_mass=2,
        atomic_number=2,
        binding_energies=None,
    )
    ni = rho / (2 * amu_TO_kg)
    Rii = (3 / (4 * np.pi * ni)) ** (1 / 3)
    Rii2 = Rii
    beta = 1 / (BOLTZMANN_CONSTANT * T)
    Gamma2 = Zi**2 * ELEMENTARY_CHARGE**2 * beta / (4 * np.pi * VACUUM_PERMITTIVITY * Rii)
    print(f"Gamma2 = {Gamma2}")
    fn = "validation/static_sf/verification/Wuensch_Thesis_Fig4-5/T_20eV_Gamma_2.7_HNC-OCP.csv"
    dat2 = np.genfromtxt(fn, delimiter=",")
    fn = "validation/static_sf/verification/Wuensch_Thesis_Fig4-5/T_20eV_Gamma_2.7_MSA-OCP.csv"
    dat22 = np.genfromtxt(fn, delimiter=",")

    sigma_c = 1.5 * BOHR_RADIUS

    kernel = OCPStaticStructureFactor(state=state, ion_core_radius=sigma_c, max_iterations=5000, mix_fraction=0.9)
    Sii_HNC2 = kernel.get_screened_ii_static_structure_factor(k=k, sf_model="HNC")
    # Sii_xHNC2 = kernel.get_ii_static_structure_factor(k=k, sf_model="EXTENDED_HNC")
    Sii_MSA2 = kernel.get_screened_ii_static_structure_factor(k=k, sf_model="MSA")

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(k * BOHR_RADIUS, Sii_MSA, label=f"MSA, Gamma={Gamma:.1f}", c="orange", ls="--")
    ax.plot(k * BOHR_RADIUS, Sii_HNC, label=f"HNC, Gamma={Gamma:.1f}", c="crimson", ls="-.")
    ax.plot(k * BOHR_RADIUS, Sii_xHNC, label=f"xHNC, Gamma={Gamma:.1f}", c="crimson", ls=":")
    # ax.scatter(
    #     dat1[:, 0] * BOHR_RADIUS / Rii1, dat1[:, 1], label=f"HNC - Wuensch, Gamma={Gamma:.1f}", c="brown", marker="x"
    # )
    # ax.scatter(
    #     dat11[:, 0] * BOHR_RADIUS / Rii1,
    #     dat11[:, 1],
    #     label=f"MSA - Wuensch, Gamma={Gamma:.1f}",
    #     c="orange",
    #     marker="o",
    # )
    ax.plot(k * BOHR_RADIUS, Sii_HNC2, label=f"HNC, Gamma={Gamma2:.1f}", c="navy", ls="-.")
    # plt.plot(k * BOHR_RADIUS, Sii_xHNC2, label="xHNC", c="navy", ls=":")
    ax.plot(k * BOHR_RADIUS, Sii_MSA2, label=f"MSA, Gamma={Gamma2:.1f}", c="dodgerblue", ls="--")
    # ax.scatter(
    #     dat2[:, 0] * BOHR_RADIUS / Rii2, dat2[:, 1], label=f"HNC - Wuensch, Gamma={Gamma2:.1f}", c="black", marker="x"
    # )
    # ax.scatter(
    #     dat22[:, 0] * BOHR_RADIUS / Rii2,
    #     dat22[:, 1],
    #     label=f"MSA - Wuensch, Gamma={Gamma2:.1f}",
    #     c="dodgerblue",
    #     marker="o",
    # )
    ax.legend()
    ax.axhline(1.0, lw=1, ls=":", c="gray")
    ax.set_xlim(-0.1, 6.0)
    # ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel(r"$k$ [$a_B^{-1}$]")
    ax.set_ylabel(r"$S_{ii}(k)$")
    plt.show()
    fig.savefig(f"static_ii_sf_comparison.pdf", dpi=200)


def test_mcp():

    Ti_eV = 10  # 8 * eV_TO_K  # eV
    Ti = Ti_eV * eV_TO_K
    T = Ti
    rho_cgs = 1.5
    rho = rho_cgs * 1000
    ZH = 1
    ZC = 4
    xH = 0.5
    xC = 0.5


if __name__ == "__main__":
    # test_ocp()
    test_screened_ocp()

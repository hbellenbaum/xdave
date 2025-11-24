# # This file is part of the Fermi-Dirac integral code fermidirac.
# # Copyright (C) 2025  Thomas Gawne

# # This program is free software: you can redistribute it and/or modify
# # it under the terms of the GNU General Public License as published by
# # the Free Software Foundation, either version 3 of the License, or
# # (at your option) any later version.

# # This program is distributed in the hope that it will be useful,
# # but WITHOUT ANY WARRANTY; without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# # GNU General Public License for more details.
# # You should have received a copy of the GNU General Public License
# # along with this program.  If not, see <https://www.gnu.org/licenses/>.


# # Implementation of Anita (1993) Padé approximant form of the Fermi-Dirac integrals and their inverses.
# # https://doi.org/10.1086/191748


## Modications: H. Bellenbaum
# All functions used here are copied over minor modification from https://gitlab.com/tgawne/fermidirac/-/tree/main
# Some function names were changed to avoid conflicts.
# fd_1h.py: R1_mk -> pos_R1_m1_7_k1_7
#           R2_mk -> pos_R2_m2_10_k2_11
# fd_3h.py: R1_mk -> R1_m1_6_k1_7
#           R2_mk -> R2_m2_9_k2_10
# fd_5h.py: R1_mk -> R1_mk_m1_6_k1_7
#           R2_mk -> R2_m2_10_k2_9
# fd_m1h.py: R1_mk -> R1_m1_7_k1_7
#            R2_mk -> R2_m2_11_k2_11

# The fd_m3h function was added by H. Bellenbaum, so was the general fdi

from .constants import SQRT_PI
from scipy.special import gamma
from numpy import ndarray, zeros
from numpy import exp as npexp
from numpy import sqrt as npsqrt
from math import exp as mathexp
from math import sqrt as mathsqrt


def gamma_function(z):
    """
    Calculates the Gamma function for a given z.
    These are currently only available for the corresponding fermi-dirac integrals (see below).



    """
    # There are some simplifications to avoid unnecessary calculations:
    # \Gamma(n) = (n-1)! for n=2,3,...
    if z == 0.5:
        return SQRT_PI
    elif z == 1.5:
        return 0.5 * SQRT_PI
    elif z == 2.5:
        return 0.75 * SQRT_PI
    elif z == 3.5:
        # this is, miraculously, accurate to the 16th floating point, so I'm sticking with it
        return 1.875 * SQRT_PI
    elif z == -0.5:
        return -2 * SQRT_PI
    else:
        return gamma(z)


def fdi(eta: int | float | ndarray, j: int | float, normalize: bool = True) -> int | float | ndarray:
    """
    Calculates the Fermi-Dirac integral for a given j.

    Parameters:
    -----------
        eta: int | float | ndarray
            Value(s) at which to calculate the Fermi-Dirac integral.

    Returns:
    --------
        fd: type of f
            Fermi-Dirac integral.
    """

    if j == -1.5:
        if normalize:
            return fd_m3h(eta=eta) / gamma_function(j + 1)
        return fd_m3h(eta=eta)
    elif j == -0.5:
        if normalize:
            return fd_m1h(eta=eta) / gamma_function(j + 1)
        return fd_m1h(eta=eta)
    elif j == 0.5:
        if normalize:
            return fd_1h(eta=eta) / gamma_function(j + 1)
        return fd_1h(eta=eta)
    elif j == 1.5:
        if normalize:
            return fd_3h(eta=eta) / gamma_function(j + 1)
        return fd_3h(eta=eta)
    elif j == 2.5:
        if normalize:
            return fd_5h(eta=eta) / gamma_function(j + 1)
        return fd_5h(eta=eta)


# --------------------------------#
# -- Fermi-Dirac Integral j=1/2 --#
# --------------------------------#


def pos_R1_m1_7_k1_7(x: int | float | ndarray) -> int | float | ndarray:
    """
    Calculates the R^1_m1k1 term in Anita (1993) [https://doi.org/10.1086/191748].
    m1 = 7, k1 = 7.

    Parameters:
    -----------
        x: int | float | ndarray
            Input of the function.

    Returns:
    --------
        R1_mk: type of x

    """
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    x6 = x5 * x
    x7 = x6 * x

    mterm = (
        5.75834152995465e6
        + 1.30964880355883e7 * x
        + 1.07608632249013e7 * x2
        + 3.93536421893014e6 * x3
        + 6.42493233715640e5 * x4
        + 4.16031909245777e4 * x5
        + 7.77238678539648e2 * x6
        + 1.00000000000000e0 * x7
    )

    kterm = (
        6.49759261942269e6
        + 1.70750501625775e7 * x
        + 1.69288134856160e7 * x2
        + 7.95192647756086e6 * x3
        + 1.83167424554505e6 * x4
        + 1.95155948326832e5 * x5
        + 8.17922106644547e3 * x6
        + 9.02129136642157e1 * x7
    )

    return mterm / kterm


def pos_R2_m2_10_k2_11(x: int | float | ndarray) -> int | float | ndarray:
    """
    Calculates the R^2_m2k2 term in Anita (1993) [https://doi.org/10.1086/191748].
    m2 = 10, k2 = 11

    Parameters:
    -----------
        x: int | float | ndarray
            Input of the function.

    Returns:
    --------
        R2_mk: type of x
            Output of the function.
    """
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    x6 = x5 * x
    x7 = x6 * x
    x8 = x7 * x
    x9 = x8 * x
    x10 = x9 * x
    x11 = x10 * x

    mterm = (
        4.85378381173415e-14
        + 1.64429113030738e-11 * x
        + 3.76794942277806e-09 * x2
        + 4.69233883900644e-07 * x3
        + 3.40679845803144e-05 * x4
        + 1.32212995937796e-03 * x5
        + 2.60768398973913e-02 * x6
        + 2.48653216266227e-01 * x7
        + 1.08037861921488e00 * x8
        + 1.91247528779676e00 * x9
        + 1.00000000000000e00 * x10
    )

    kterm = (
        7.28067571760518e-14
        + 2.45745452167585e-11 * x
        + 5.62152894375277e-09 * x2
        + 6.96888634549649e-07 * x3
        + 5.02360015186394e-05 * x4
        + 1.92040136756592e-03 * x5
        + 3.66887808002874e-02 * x6
        + 3.24095226486468e-01 * x7
        + 1.16434871200131e00 * x8
        + 1.34981244060549e00 * x9
        + 2.01311836975930e-01 * x10
        - 2.14562434782759e-02 * x11
    )

    return mterm / kterm


def fd_1h(eta: int | float | ndarray) -> int | float | ndarray:
    """
    Calculates the Fermi-Dirac integral for k = 1/2.

    Parameters:
    -----------
        eta: int | float | ndarray
            Value(s) at which to calculate the Fermi-Dirac integral.

    Returns:
    --------
        fd: type of f
            Fermi-Dirac integral for k = 1/2.
    """

    n = 1.5  # (1 + k)

    if isinstance(eta, ndarray):
        fd = zeros(eta.shape)
        eta_cond = eta < 2

        expdata = npexp(eta[eta_cond])
        fd[eta_cond] = expdata * pos_R1_m1_7_k1_7(expdata)

        eta2 = eta[~eta_cond]
        fd[~eta_cond] = eta2**n * pos_R2_m2_10_k2_11(1 / eta2**2)

        return fd

    # Otherwise just usual if-else
    if eta < 2:
        expeta = mathexp(eta)
        return expeta * pos_R1_m1_7_k1_7(expeta)

    return eta**n * pos_R2_m2_10_k2_11(1 / eta**2)


# --------------------------------#
# -- Fermi-Dirac Integral j=3/2 --#
# --------------------------------#


def R1_m1_6_k1_7(x: int | float | ndarray) -> int | float | ndarray:
    """
    Calculates the R^1_m1k1 term in Anita (1993) [https://doi.org/10.1086/191748].
    m1 = 6, k1 = 7.

    Parameters:
    -----------
        x: int | float | ndarray
            Input of the function.

    Returns:
    --------
        R1_mk: type of x

    """
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    x6 = x5 * x
    x7 = x6 * x

    mterm = (
        4.32326386604283e4
        + 8.55472308218786e4 * x
        + 5.95275291210962e4 * x2
        + 1.77294861572005e4 * x3
        + 2.21876607796460e3 * x4
        + 9.90562948053193e1 * x5
        + 1.00000000000000e0 * x6
    )

    kterm = (
        3.25218725353467e4
        + 7.01022511904373e4 * x
        + 5.50859144223638e4 * x2
        + 1.95942074576400e4 * x3
        + 3.20803912586318e3 * x4
        + 2.20853967067789e2 * x5
        + 5.05580641737527e0 * x6
        + 1.99507945223266e-2 * x7
    )

    return mterm / kterm


def R2_m2_9_k2_10(x: int | float | ndarray) -> int | float | ndarray:
    """
    Calculates the R^2_m2k2 term in Anita (1993) [https://doi.org/10.1086/191748].
    m2 = 9, k2 = 10

    Parameters:
    -----------
        x: int | float | ndarray
            Input of the function.

    Returns:
    --------
        R2_mk: type of x
            Output of the function.
    """
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    x6 = x5 * x
    x7 = x6 * x
    x8 = x7 * x
    x9 = x8 * x
    x10 = x9 * x

    mterm = (
        2.80452693148553e-13
        + 8.60096863656367e-11 * x
        + 1.62974620742993e-08 * x2
        + 1.63598843752050e-06 * x3
        + 9.12915407846722e-05 * x4
        + 2.62988766922117e-03 * x5
        + 3.85682997219346e-02 * x6
        + 2.78383256609605e-01 * x7
        + 9.02250179334496e-01 * x8
        + 1.00000000000000e00 * x9
    )

    kterm = (
        7.01131732871184e-13
        + 2.10699282897576e-10 * x
        + 3.94452010378723e-08 * x2
        + 3.84703231868724e-06 * x3
        + 2.04569943213216e-04 * x4
        + 5.31999109566385e-03 * x5
        + 6.39899717779153e-02 * x6
        + 3.14236143831882e-01 * x7
        + 4.70252591891375e-01 * x8
        - 2.15540156936373e-02 * x9
        + 2.34829436438087e-03 * x10
    )

    return mterm / kterm


def fd_3h(eta: int | float | ndarray) -> int | float | ndarray:
    """
    Calculates the Fermi-Dirac integral for k = 3/2.

    Parameters:
    -----------
        eta: int | float | ndarray
            Value(s) at which to calculate the Fermi-Dirac integral.

    Returns:
    --------
        fd: type of f
            Fermi-Dirac integral for k = 3/2.
    """

    n = 2.5  # (1 + k)

    if isinstance(eta, ndarray):
        fd = zeros(eta.shape)
        eta_cond = eta < 2

        expdata = npexp(eta[eta_cond])
        fd[eta_cond] = expdata * R1_m1_6_k1_7(expdata)

        eta2 = eta[~eta_cond]
        fd[~eta_cond] = eta2**n * R2_m2_9_k2_10(1 / eta2**2)

        return fd

    # Otherwise just usual if-else
    if eta < 2:
        expeta = exp(eta)
        return expeta * R1_m1_6_k1_7(expeta)

    return eta**n * R2_m2_9_k2_10(1 / eta**2)


# --------------------------------#
# -- Fermi-Dirac Integral j=5/2 --#
# --------------------------------#


def R1_mk_m1_6_k1_7(x: int | float | ndarray) -> int | float | ndarray:
    """
    Calculates the R^1_m1k1 term in Anita (1993) [https://doi.org/10.1086/191748].
    m1 = 6, k1 = 7.

    Parameters:
    -----------
        x: int | float | ndarray
            Input of the function.

    Returns:
    --------
        R1_mk: type of x

    """
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    x6 = x5 * x
    x7 = x6 * x

    mterm = (
        6.61606300631656e4
        + 1.20132462801652e5 * x
        + 7.67255995316812e4 * x2
        + 2.10427138842443e4 * x3
        + 2.44325236813275e3 * x4
        + 1.02589947781696e2 * x5
        + 1.00000000000000e0 * x6
    )

    kterm = (
        1.99078071053871e4
        + 3.79076097261066e4 * x
        + 2.60117136841197e4 * x2
        + 7.97584657659364e3 * x3
        + 1.10886130159658e3 * x4
        + 6.35483623268093e1 * x5
        + 1.16951072617142e0 * x6
        + 3.31482978240026e-3 * x7
    )

    return mterm / kterm


def R2_m2_10_k2_9(x: int | float | ndarray) -> int | float | ndarray:
    """
    Calculates the R^2_m2k2 term in Anita (1993) [https://doi.org/10.1086/191748].
    m2 = 10, k2 = 9

    Parameters:
    -----------
        x: int | float | ndarray
            Input of the function.

    Returns:
    --------
        R2_mk: type of x
            Output of the function.
    """
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    x6 = x5 * x
    x7 = x6 * x
    x8 = x7 * x
    x9 = x8 * x
    x10 = x9 * x

    mterm = (
        8.42667076131315e-12
        + 2.31618876821567e-09 * x
        + 3.54323824923987e-07 * x2
        + 2.77981736000034e-05 * x3
        + 1.14008027400645e-03 * x4
        + 2.32779790773633e-02 * x5
        + 2.39564845938301e-01 * x6
        + 1.24415366126179e00 * x7
        + 3.18831203950106e00 * x8
        + 3.42040216997894e00 * x9
        + 1.00000000000000e00 * x10
    )

    kterm = (
        2.94933476646033e-11
        + 7.68215783076936e-09 * x
        + 1.12919616415947e-06 * x2
        + 8.09451165406274e-05 * x3
        + 2.81111224925648e-03 * x4
        + 3.99937801931919e-02 * x5
        + 2.27132567866839e-01 * x6
        + 5.31886045222680e-01 * x7
        + 3.70866321410385e-01 * x8
        + 2.27326643192516e-02 * x9
    )

    return mterm / kterm


def fd_5h(eta: int | float | ndarray) -> int | float | ndarray:
    """
    Calculates the Fermi-Dirac integral for k = 5/2.

    Parameters:
    -----------
        eta: int | float | ndarray
            Value(s) at which to calculate the Fermi-Dirac integral.

    Returns:
    --------
        fd: type of f
            Fermi-Dirac integral for k = 5/2.
    """

    n = 3.5  # (1 + k)

    if isinstance(eta, ndarray):
        fd = zeros(eta.shape)
        eta_cond = eta < 2

        expdata = npexp(eta[eta_cond])
        fd[eta_cond] = expdata * R1_mk_m1_6_k1_7(expdata)

        eta2 = eta[~eta_cond]
        fd[~eta_cond] = eta2**n * R2_m2_10_k2_9(1 / eta2**2)

        return fd

    # Otherwise just usual if-else
    if eta < 2:
        expeta = exp(eta)
        return expeta * R1_mk_m1_6_k1_7(expeta)

    return eta**n * R2_m2_10_k2_9(1 / eta**2)


# ---------------------------------#
# -- Fermi-Dirac Integral j=-1/2 --#
# ---------------------------------#


def R1_m1_7_k1_7(x: int | float | ndarray) -> int | float | ndarray:
    """
    Calculates the R^1_m1k1 term in Anita (1993) [https://doi.org/10.1086/191748].
    m1 = 7, k1 = 7.

    Parameters:
    -----------
        x: int | float | ndarray
            Input of the function.

    Returns:
    --------
        R1_mk: type of x

    """
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    x6 = x5 * x
    x7 = x6 * x

    mterm = (
        1.71446374704454e7
        + 3.88148302324068e7 * x
        + 3.16743385304962e7 * x2
        + 1.14587609192151e7 * x3
        + 1.83696370756153e6 * x4
        + 1.14980998186874e5 * x5
        + 1.98276889924768e3 * x6
        + 1.00000000000000e0 * x7
    )

    kterm = (
        9.67282587452899e6
        + 2.87386436731785e7 * x
        + 3.26070130734158e7 * x2
        + 1.77657027846367e7 * x3
        + 4.81648022267831e6 * x4
        + 6.13709569333207e5 * x5
        + 3.13595854332114e4 * x6
        + 4.35061725080755e2 * x7
    )

    return mterm / kterm


def R2_m2_11_k2_11(x: int | float | ndarray) -> int | float | ndarray:
    """
    Calculates the R^2_m2k2 term in Anita (1993) [https://doi.org/10.1086/191748].
    m2 = 11, k2 = 11

    Parameters:
    -----------
        x: int | float | ndarray
            Input of the function.

    Returns:
    --------
        R2_mk: type of x
            Output of the function.
    """
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    x6 = x5 * x
    x7 = x6 * x
    x8 = x7 * x
    x9 = x8 * x
    x10 = x9 * x
    x11 = x10 * x

    mterm = (
        -4.46620341924942e-15
        - 1.58654991146236e-12 * x
        - 4.44467627042232e-10 * x2
        - 6.84738791621745e-08 * x3
        - 6.64932238528105e-06 * x4
        - 3.69976170193942e-04 * x5
        - 1.12295393687006e-02 * x6
        - 1.60926102124442e-01 * x7
        - 8.52408612877447e-01 * x8
        - 7.45519953763928e-01 * x9
        + 2.98435207466372e00 * x10
        + 1.00000000000000e00 * x11
    )

    kterm = (
        -2.23310170962369e-15
        - 7.94193282071464e-13 * x
        - 2.22564376956228e-10 * x2
        - 3.43299431079845e-08 * x3
        - 3.33919612678907e-06 * x4
        - 1.86432212187088e-04 * x5
        - 5.69764436880529e-03 * x6
        - 8.34904593067194e-02 * x7
        - 4.78770844009440e-01 * x8
        - 4.99759250374148e-01 * x9
        + 1.86795964993052e00 * x10
        + 4.16485970495288e-01 * x11
    )

    return mterm / kterm


def fd_m1h(eta: int | float | ndarray) -> int | float | ndarray:
    """
    Calculates the Fermi-Dirac integral for k = -1/2.

    Parameters:
    -----------
        eta: int | float | ndarray
            Value(s) at which to calculate the Fermi-Dirac integral.

    Returns:
    --------
        fd: type of f
            Fermi-Dirac integral for k = -1/2.
    """

    n = 0.5  # (1 + k)

    if isinstance(eta, ndarray):
        fd = zeros(eta.shape)
        eta_cond = eta < 2

        expdata = npexp(eta[eta_cond])
        fd[eta_cond] = expdata * R1_m1_7_k1_7(expdata)

        eta2 = eta[~eta_cond]
        fd[~eta_cond] = eta2**n * R2_m2_11_k2_11(1 / eta2**2)

        return fd

    # Otherwise just usual if-else
    if eta < 2:
        expeta = exp(eta)
        return expeta * R1_m1_7_k1_7(expeta)

    return eta**n * R2_m2_11_k2_11(1 / eta**2)


# ---------------------------------#
# -- Fermi-Dirac Integral j=-3/2 --#
# ---------------------------------#


def exp(x):
    if isinstance(x, ndarray):
        return npexp(x)

    return mathexp(x)


def sqrt(x):
    if isinstance(x, ndarray):
        return npsqrt(x)

    return mathsqrt(x)


def fd_m3h(eta: int | float | ndarray) -> int | float | ndarray:
    if eta < -2:
        return fdm3h_lt_m2(eta)
    if eta < 0:
        return fdm3h_m2_to_0(eta)
    if eta < 2:
        return fdm3h_0_to_2(eta)
    if eta < 5:
        return fdm3h_2_to_5(eta)
    if eta < 10:
        return fdm3h_5_to_10(eta)
    if eta < 20:
        return fdm3h_10_to_20(eta)
    if eta < 40:
        return fdm3h_20_to_40(eta)
    return fdm3h_gt_40(eta)


def fdm3h_lt_m2(phi):
    exp_phi = exp(phi)
    t = exp_phi * 7.38905609893065023
    return exp_phi * (
        -3.54490770181103205
        + exp_phi
        * (
            82737.595643818605
            + t
            * (
                18481.5553495836940
                + t * (1272.73919064487495 + t * (26.3420403338352574 - t * 0.00110648970639283347))
            )
        )
        / (16503.7625405383183 + t * (6422.0552658801394 + t * (890.85389683932154 + t * (51.251447078851450 + t))))
    )


def fdm3h_m2_to_0(phi):
    s = -0.5 * phi
    t = 1 - s
    return -(
        946.638483706348559
        + t
        * (
            76.3328330396778450
            + t
            * (
                62.7809183134124193
                + t
                * (
                    83.8442376534073219
                    + t
                    * (
                        23.2285755924515097
                        + t
                        * (
                            3.21516808559640925
                            + t * (1.58754232369392539 + t * (0.687397326417193593 + t * 0.111510355441975495))
                        )
                    )
                )
            )
        )
    ) / (
        889.4123665319664
        + s
        * (
            126.7054690302768
            + s
            * (
                881.4713137175090
                + s
                * (
                    108.2557767973694
                    + s
                    * (
                        289.38131234794585
                        + s * (27.75902071820822 + s * (34.252606975067480 + s * (1.9592981990370705 + s)))
                    )
                )
            )
        )
    )


def fdm3h_0_to_2(phi):
    t = 0.5 * phi
    return -(
        754.61690882095729
        + t
        * (
            565.56180911009650
            + t
            * (
                494.901267018948095
                + t
                * (
                    267.922900418996927
                    + t
                    * (
                        110.418683240337860
                        + t
                        * (
                            39.4050164908951420
                            + t * (10.8654460206463482 + t * (2.11194887477009033 + t * 0.246843599687496060))
                        )
                    )
                )
            )
        )
    ) / (
        560.03894899770103
        + t
        * (
            70.007586553114572
            + t
            * (
                582.42052644718871
                + t
                * (
                    56.181678606544951
                    + t
                    * (
                        205.248662395572799
                        + t * (12.5169006932790528 + t * (27.2916998671096202 + t * (0.53299717876883183 + t)))
                    )
                )
            )
        )
    )


def fdm3h_2_to_5(phi):
    t = 0.3333333333333333333 * (phi - 2)
    return -(
        526.022770226139287
        + t
        * (
            631.116211478274904
            + t
            * (
                516.367876532501329
                + t
                * (
                    267.894697896892166
                    + t
                    * (
                        91.3331816844847913
                        + t
                        * (
                            17.5723541971644845
                            + t * (1.46434478819185576 + t * (1.29615441010250662 + t * 0.223495452221465265))
                        )
                    )
                )
            )
        )
    ) / (
        354.867400305615304
        + t
        * (
            560.931137013002977
            + t
            * (
                666.070260050472570
                + t
                * (
                    363.745894096653220
                    + t
                    * (
                        172.272943258816724
                        + t * (23.7751062504377332 + t * (12.5916012142616255 + t * (-0.888604976123420661 + t)))
                    )
                )
            )
        )
    )


def fdm3h_5_to_10(phi):
    t = 0.2 * phi - 1
    return -(
        18.0110784494455205
        + t
        * (
            36.1225408181257913
            + t
            * (
                38.4464752521373310
                + t
                * (
                    24.1477896166966673
                    + t
                    * (
                        9.27772356782901602
                        + t * (2.49074754470533706 + t * (0.163824586249464178 - t * 0.00329391807590771789))
                    )
                )
            )
        )
    ) / (
        18.8976860386360201
        + t
        * (
            49.3696375710309920
            + t
            * (
                60.9273314194720251
                + t * (43.6334649971575003 + t * (20.6568810936423065 + t * (6.11094689399482273 + t)))
            )
        )
    )


def fdm3h_10_to_20(phi):
    t = 0.1 * phi - 1
    return -(
        4.10698092142661427
        + t
        * (
            17.1412152818912658
            + t
            * (
                32.6347877674122945
                + t
                * (
                    36.6653101837618939
                    + t
                    * (
                        25.9424894559624544
                        + t
                        * (
                            11.2179995003884922
                            + t * (2.30099511642112478 + t * (0.0928307248942099967 - t * 0.00146397877054988411))
                        )
                    )
                )
            )
        )
    ) / (
        6.40341731836622598
        + t
        * (
            30.1333068545276116
            + t
            * (
                64.0494725642004179
                + t
                * (
                    80.5635003792282196
                    + t * (64.9297873014508805 + t * (33.3013900893183129 + t * (9.61549304470339929 + t)))
                )
            )
        )
    )


def fdm3h_20_to_40(phi):
    t = 0.05 * phi - 1
    return -(
        95.2141371910496454
        + t
        * (
            420.050572604265456
            + t
            * (
                797.778374374075796
                + t
                * (
                    750.378359146985564
                    + t
                    * (
                        324.818150247463736
                        + t
                        * (
                            50.3115388695905757
                            + t * (0.372431961605507103 + t * (-0.103162211894757911 + t * 0.00191752611445211151))
                        )
                    )
                )
            )
        )
    ) / (
        212.232981736099697
        + t
        * (
            1043.79079070035083
            + t
            * (
                2224.50099218470684
                + t
                * (
                    2464.84669868672670
                    + t * (1392.55318009810070 + t * (346.597189642259199 + t * (22.7314613168652593 - t)))
                )
            )
        )
    )


def fdm3h_gt_40(phi):
    factor = -2  # = 1/(k+1)
    w = 1 / (phi * phi)
    s = 1 - 1600 * w
    return (
        factor
        / sqrt(phi)
        * (
            1
            + w
            * (12264.3569103180524 + s * (3204.34872454052352 + s * (140.119604748253961 + s * 0.523918919699235590)))
            / (9877.87829948067200 + s * (2644.71979353906092 + s * (128.863768007644572 + s)))
        )
    )

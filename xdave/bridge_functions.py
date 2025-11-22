import numpy as np
import warnings


def iyetomi_bridge_function(rs, Rii, Gamma):
    """
    Bridge function for strongly coupled one-component systems based on Iyetomi et al., Phys. Rev. A 46 (1992).

    Parameters:
        rs (array): r-grid to calculate bridge function over in units of m
        Rii (float): effective ion core radius in units of m
        Gamma (float): coupling strength, non-dimensional

    Returns:
        array: non-dimensional grid over which bridge function is calculated
        array: bridge function in r-space
    """
    ## Iyetomi, Phys. Rev. A 46 (1992) bridge function
    if Gamma < 5:
        warnings.warn(
            f"Iyetomi bridge function is not valid for weakly coupled plasmas. In this regime, the extended HNC solver should not be applied."
        )
    xs = rs / Rii
    lnGamma = np.log(Gamma)
    b0 = 0.258 - 0.0612 * lnGamma + 0.0123 * lnGamma**2 - 1 / Gamma
    b1 = 0.0269 + 0.0318 * lnGamma + 0.00814 * lnGamma**2
    c1 = 0.489 - 0.280 * lnGamma + 0.0294 * lnGamma**2
    c2 = -0.412 + 0.219 * lnGamma - 0.0251 * lnGamma**2
    c3 = 0.0988 - 0.0534 * lnGamma + 0.00682 * lnGamma**2
    Bii_rs = Gamma * (-b0 + c1 * xs**4 + c2 * xs**6 + c3 * xs**8) * np.exp(-b1 / b0 * xs**2)
    return xs, Bii_rs

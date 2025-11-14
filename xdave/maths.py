from numpy import exp, log1p

# NOTE(HB): These are not used here.


def is_iterable(test_variable):
    r"""
    As noted on: https://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable
    there is no one single way of testing whether an object is iterable that is 'best'. The general advice seems to be
    to perform a try-except on the iter() method.
    """
    try:
        iter(test_variable)
        return True
    except TypeError:
        return False


def log1pexp(x):
    r"""
    Returns the function log1pexp(x) = \ln(1 + exp(x)) via the expressions recommended by Eq. (10) by Martin M\"achler.

    Parameters
    ----------
    x: float scalar/iterable
        The argument of the function.

    References
    ----------
    M. M\"achler, "Accurately Computing log(1-exp(-|a|)) Assessed by the Rmpfr package",
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    x_0 = -37.0
    x_1 = 18.0
    x_2 = 33.3
    if is_iterable(x):
        f = x
        mask_1 = x <= x_0
        mask_2 = (x_0 < x) & (x <= x_1)
        mask_3 = (x_1 < x) & (x <= x_2)
        mask_4 = x > x_2
        f[mask_1] = exp(x[mask_1])
        f[mask_2] = log1p(exp(x[mask_2]))
        f[mask_3] = x[mask_3] + exp(-x[mask_3])
        f[mask_4] = x[mask_4]
        return f
    else:
        if x <= x_0:
            return exp(x)
        elif x_0 < x <= x_1:
            return log1p(exp(x))
        elif x_1 < x <= x_2:
            return x + exp(-x)
        else:
            return x

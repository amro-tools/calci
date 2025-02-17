import numpy as np

def finite_difference(func, x: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """Compute the derivative of `func` wrt to x.

    Args:
        func: a function that takes an input array with shape `m` and returns a result, which is an array of shape `n` or a scalar
        x (np.ndarray): an array of shape `m` at which to evaluate the derivative
        epsilon (float, optional): Finite difference step length. Defaults to 1e-7.

    Returns:
        derivative: The derivative at `x`, computed with central finite differences. An array of shape (*m, *n).
    """
    # We use the `atleast_1d` function to treat the case where func returns a single scalar
    # on equal footing with cases where it returns an array
    res_shape = np.atleast_1d(func(x)).shape

    deriv_shape = x.shape + res_shape
    derivatives = np.zeros(deriv_shape)

    # Use a numpy nditer to iterate over the input array
    # More info here: https://numpy.org/doc/stable/reference/arrays.nditer.html#arrays-nditer
    with np.nditer(x, flags=["multi_index"], op_flags=["readwrite"]) as it:
        for x_ in it:
            # save the original number
            original = float(x_[...])

            # plus epsilon
            x_[...] += epsilon
            ep = np.atleast_1d(func(x))

            # minus epsilon
            x_[...] -= 2.0 * epsilon
            em = np.atleast_1d(func(x))

            # restore to original
            x_[...] = original

            # record the derivative
            derivatives[it.multi_index] = (ep - em) / (2.0 * epsilon)

    # Before we return, we remove superfluous dimensions
    return np.squeeze(derivatives)
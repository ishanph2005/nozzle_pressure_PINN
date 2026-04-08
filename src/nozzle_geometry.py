"""
Nozzle Geometry

Canonical converging-diverging nozzle area profile A(x) derived from a
6th-degree polynomial fit to the 8 experimental port measurements.

    x:    0.125  0.25  0.375  0.5   0.625  0.75  0.875  1.0
    A(x): 1.44   1.00  1.13   1.28  1.42   1.59  1.77   1.94

Throat is at x ≈ 0.25 where A = 1.0 (minimum area).
"""

import numpy as np

# Polynomial coefficients for A(x), degree 6 down to degree 0.
# Fit exactly through the 8 measurement stations.
_A_COEFFS = [
    92.84266667, -358.4, 555.52,
    -440.41212121, 187.68278788, -39.46954545, 4.17625
]

# Derivative coefficients dA/dx (degree 5 down to degree 0)
_DA_COEFFS = [
    557.056, -1792.0, 2222.08,
    -1321.23636364, 375.36557576, -39.46954545
]

# Key geometric parameters
X_THROAT = 0.25
A_THROAT = 1.0
X_DOMAIN = (0.125, 1.0)


def nozzle_area_np(x):
    """Compute nozzle cross-sectional area A(x) using NumPy. Horner's method."""
    c = _A_COEFFS
    return c[6] + x * (c[5] + x * (c[4] + x * (c[3] + x * (c[2] + x * (c[1] + x * c[0])))))


def nozzle_area_deriv_np(x):
    """Compute dA/dx using NumPy. Horner's method."""
    c = _DA_COEFFS
    return c[5] + x * (c[4] + x * (c[3] + x * (c[2] + x * (c[1] + x * c[0]))))


def nozzle_area_torch(x):
    """Compute nozzle cross-sectional area A(x) using PyTorch tensors."""
    c = _A_COEFFS
    return c[6] + x * (c[5] + x * (c[4] + x * (c[3] + x * (c[2] + x * (c[1] + x * c[0])))))

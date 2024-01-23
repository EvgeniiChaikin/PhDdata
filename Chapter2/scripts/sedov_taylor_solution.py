"""
Sedov-Taylor adiabatic solution to a point-like explosion
"""


import numpy as np
import matplotlib.pylab as plt
from scipy.special import gamma as Gamma

# Constants
constants = {
    "NEWTON_GRAVITY_CGS": 6.67408e-8,
    "SOLAR_MASS_IN_CGS": 1.98848e33,
    "PARSEC_IN_CGS": 3.08567758e18,
    "PROTON_MASS_IN_CGS": 1.672621898e-24,
    "BOLTZMANN_IN_CGS": 1.38064852e-16,
    "YEAR_IN_CGS": 3.15569252e7,
    "H_mass_fraction": 0.73738788833,
}


def calc_a(g, nu=3):
    """
    exponents of the polynomials of the sedov solution
    g - the polytropic gamma
    nu - the dimension
    """
    a = [0] * 8

    a[0] = 2.0 / (nu + 2)
    a[2] = (1 - g) / (2 * (g - 1) + nu)
    a[3] = nu / (2 * (g - 1) + nu)
    a[5] = 2 / (g - 2)
    a[6] = g / (2 * (g - 1) + nu)

    a[1] = (((nu + 2) * g) / (2.0 + nu * (g - 1.0))) * (
        (2.0 * nu * (2.0 - g)) / (g * (nu + 2.0) ** 2) - a[2]
    )
    a[4] = a[1] * (nu + 2) / (2 - g)
    a[7] = (2 + nu * (g - 1)) * a[1] / (nu * (2 - g))
    return a


def calc_beta(v, g, nu=3):
    """
    beta values for the sedov solution (coefficients of the polynomials of the similarity variables)
    v - the similarity variable
    g - the polytropic gamma
    nu- the dimension
    """

    beta = (
        (nu + 2)
        * (g + 1)
        * np.array(
            (
                0.25,
                (g / (g - 1)) * 0.5,
                -(2 + nu * (g - 1))
                / 2.0
                / ((nu + 2) * (g + 1) - 2 * (2 + nu * (g - 1))),
                -0.5 / (g - 1),
            ),
            dtype=np.float64,
        )
    )

    beta = np.outer(beta, v)

    beta += (g + 1) * np.array(
        (
            0.0,
            -1.0 / (g - 1),
            (nu + 2) / ((nu + 2) * (g + 1) - 2.0 * (2 + nu * (g - 1))),
            1.0 / (g - 1),
        ),
        dtype=np.float64,
    ).reshape((4, 1))

    return beta


def sedov(t, E0, rho0, g, n=1000, nu=3):
    """
    solve the sedov problem
    t - the time
    E0 - the initial energy
    rho0 - the initial density
    n - number of points (10000)
    nu - the dimension
    g - the polytropic gas gamma
    """
    # the similarity variable
    v_min = 2.0 / ((nu + 2) * g)
    v_max = 4.0 / ((nu + 2) * (g + 1))

    v = v_min + np.arange(n) * (v_max - v_min) / (n - 1.0)

    a = calc_a(g, nu)
    beta = calc_beta(v, g=g, nu=nu)
    lbeta = np.log(beta)

    r = np.exp(-a[0] * lbeta[0] - a[2] * lbeta[1] - a[1] * lbeta[2])
    rho = ((g + 1.0) / (g - 1.0)) * np.exp(
        a[3] * lbeta[1] + a[5] * lbeta[3] + a[4] * lbeta[2]
    )
    p = np.exp(
        nu * a[0] * lbeta[0] + (a[5] + 1) * lbeta[3] + (a[4] - 2 * a[1]) * lbeta[2]
    )
    u = beta[0] * r * 4.0 / ((g + 1) * (nu + 2))
    p *= 8.0 / ((g + 1) * (nu + 2) * (nu + 2))

    # we have to take extra care at v=v_min, since this can be a special point.
    # It is not a singularity, however, the gradients of our variables (wrt v) are.
    # r -> 0, u -> 0, rho -> 0, p-> constant

    u[0] = 0.0
    rho[0] = 0.0
    r[0] = 0.0
    p[0] = p[1]

    # volume of an n-sphere
    vol = (np.pi ** (nu / 2.0) / Gamma(nu / 2.0 + 1)) * np.power(r, nu)

    # note we choose to evaluate the integral in this way because the
    # volumes of the first few elements (i.e near v=vmin) are shrinking
    # very slowly, so we dramatically improve the error convergence by
    # finding the volumes exactly. This is most important for the
    # pressure integral, as this is on the order of the volume.

    # (dimensionless) energy of the model solution
    de = rho * u * u * 0.5 + p / (g - 1)
    # integrate (trapezium rule)
    q = np.inner(de[1:] + de[:-1], np.diff(vol)) * 0.5

    # the factor to convert to this particular problem
    fac = (q * (t**nu) * rho0 / E0) ** (-1.0 / (nu + 2))

    # shock speed
    shock_speed = fac * (2.0 / (nu + 2))
    rho_s = ((g + 1) / (g - 1)) * rho0
    r_s = shock_speed * t * (nu + 2) / 2.0
    p_s = (2.0 * rho0 * shock_speed * shock_speed) / (g + 1)
    u_s = (2.0 * shock_speed) / (g + 1)

    r *= fac * t
    u *= fac
    p *= fac * fac * rho0
    rho *= rho0

    return r, p, rho, u, r_s, p_s, rho_s, u_s, shock_speed


if __name__ == "__main__":
    time_kyr = 1e3  # in kyr
    n_H = 0.1  # Gas density in H/cc
    n_radial_bins = 300  # Number of radial bins
    gamma = 5.0 / 3.0  # Adiabatic index
    dimensions = 3  # Number of dimensions
    X = 0.7373879  # Hydrogen mass fractions
    E_SN = 1e51  # erg

    # r_s radii
    # P_s Pressures
    # rho_s Densities
    # v_s Radial velocities
    # r_shock shock radius form the explosion

    r_s, P_s, rho_s, v_s, r_shock, _, _, _, _ = sedov(
        time_kyr * constants["YEAR_IN_CGS"] * 1e3,
        E_SN,
        n_H / X * constants["PROTON_MASS_IN_CGS"],
        gamma,
        n_radial_bins,
        dimensions,
    )

    # Example of using output arrays

    fig, ax = plt.subplots(2, 1, figsize=(6, 10), sharex=True)
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["axes.linewidth"] = 2

    LABEL_SIZE = 25

    ax[1].set_xlabel("Distance from SN [pc]", fontsize=LABEL_SIZE)
    ax[0].set_ylabel("$n_{\\rm H} \\, \\rm [cm^{-3}]$", fontsize=LABEL_SIZE)
    ax[1].set_ylabel("$v \\, \\rm [km s^{-1}]$", fontsize=LABEL_SIZE)
    ax[1].set_ylabel("log $n_{\\rm ^{60}Fe} \\rm \\, [cm^{-3}]$", fontsize=LABEL_SIZE)

    ax[0].plot(
        r_s / constants["PARSEC_IN_CGS"],
        rho_s * (X / constants["PROTON_MASS_IN_CGS"]),
        lw=2,
        color="deepskyblue",
    )

    ax[1].plot(r_s / constants["PARSEC_IN_CGS"], v_s / 1e5, lw=2, color="deepskyblue")

    ax[0].text(
        0.96,
        0.96,
        "$t = {:.1f}$ Myr".format(time_kyr / 1e3),
        ha="right",
        va="top",
        transform=ax[0].transAxes,
        fontsize=LABEL_SIZE,
    )

    plt.savefig("sedov_taylor.pdf", bbox_inches="tight", pad_inches=0.1)

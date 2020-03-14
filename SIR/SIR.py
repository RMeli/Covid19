"""
Susceptible-Infected-Recovered (SIR) Model
without demographics (births/deaths) with mortality

References:

* Modeling Infectious Diseases in Humans and Animals,
  M. J. Keeling and P. Rohani,
  Princeton University Press,
  2008
"""

import numpy as np

from scipy import integrate

from matplotlib import pyplot as plt

# Total UK population (2019 estimate)
N = 67545757


def SIR(t, y, R0, gamma, m):
    """
    SIR ODEs

    Parameters
    ----------
    t: double
        Current time
    y: np.ndarray
        Current state
    R0: double
        Basic reproductive ratio
    m: double
        Mortality rate

    Notes
    -----
    Deaths are included in dR/dt (recovered rate) instead of dI/dt (infected rate);
    this is to simulate a delayed death after infection (incubation period). 
    For this reason the infected rate is not influenced by the death rate.
    """

    assert 0 <= m <= 1

    # Changes in S, I and R do not depend on R
    S, I = y[0], y[1]

    # Gamma and beta are related by R0
    beta = gamma * R0

    dSdt = -beta * I * S
    dIdt = beta * I * S - gamma * I
    dRdt = gamma * I - m * gamma * I
    dDdt = m * gamma * I  # Deaths

    return np.array([dSdt, dIdt, dRdt, dDdt])


def solve_SIR(I0, tint, R0, gamma, m):
    """
    Solve system of SIR ODEs

    Parameters
    ----------
    I0: int
        Initial number of infected people
    tint: Tuple[double]
        Time interval
    R0: double
        Basic reproductive ratio
    m: double
        Mortality rate
    """

    # Initial susceptible and infected people as population fraction
    S0 = (N - I0) / N
    I0 = I0 / N

    y0 = np.array([S0, I0, 0, 0])

    sir = integrate.solve_ivp(
        lambda t, y: SIR(t, y, R0, gamma, m if m is not None else 0.0),
        tint,
        y0,
        rtol=1e-12,
        atol=1e-12,
    )

    # Check conservation of people
    assert np.allclose(sir.y.sum(axis=0), S0 + I0)

    return sir


def plot_sir(I0, tint, R0, gamma, m=None, tag=None):
    """
    Plot SIR model

    Parameters
    ----------
    I0: int
        Initial number of infected people
    tint: Tuple[double]
        Time interval
    R0: double
        Basic reproductive ratio
    m: double (optional)
        Mortality rate
    tag: str
        Plot tag
    """

    sir = solve_SIR(I0, tint, R0, gamma, m)

    fig = plt.figure()

    plt.plot(sir.t, sir.y[0], label="Susceptible")
    plt.plot(sir.t, sir.y[1], label="Infected")
    plt.plot(sir.t, sir.y[2], label="Recovered")

    if m is not None:
        plt.plot(sir.t, sir.y[3], label="Dead")
    plt.legend()

    plt.title(f"SIR Model ($R_0$  = {R0:.2f} {f', m = {m}' if m is not None else ''})")
    plt.xlabel("Time")
    plt.ylabel("Fraction of Population")

    if tag is None:
        tag = "simple" if m is None else "mortality"
    plt.savefig(f"plots/SIR_{tag}.pdf")
    plt.close(fig)


def plot_infected(I0, tint, R0, gamma, m=None):
    """
    Plot fraction of infected people (SIR model) for different values of R0

    Parameters
    ----------
    I0: int
        Initial number of infected people
    tint: Tuple[double]
        Time interval
    R0: List[double]
        Basic reproductive ratio
    m: double
        Mortality rate
    """

    fig = plt.figure()

    for r0 in R0:
        sir = solve_SIR(I0, tint, r0, gamma, m)

        plt.plot(sir.t, sir.y[1], label=f"$R_0$  = {r0:.2f}")

    plt.legend()
    plt.title(f"SIR Model (Infected)")
    plt.xlabel("Time")
    plt.ylabel("Fraction of Infected Population")

    tag = "simple" if m is None else "mortality"
    plt.savefig(f"plots/SIR_{tag}_R.pdf")
    plt.close(fig)


# Number of infected people in the UK (13/03/2020)
i0 = 797
tint = (0, 40)

# No mortality
plot_sir(I0=i0, tint=tint, R0=2, gamma=1)

# 2.5% mortality
plot_sir(I0=i0, tint=tint, R0=2, gamma=1, m=0.025)

# No mortality
plot_infected(I0=i0, tint=tint, R0=[1.5, 2.0, 2.5, 3.0, 3.5], gamma=1)

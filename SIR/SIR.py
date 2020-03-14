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


def SIR_mortality(t, y, R0, m, beta=1):
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
    gamma = beta / R0

    dSdt = -beta * I * S
    dIdt = beta * I * S - gamma * I
    dRdt = gamma * I - m * gamma * I
    dDdt = m * gamma * I  # Deaths

    return np.array([dSdt, dIdt, dRdt, dDdt])


def solve_SIR(I0, tf, R0, m):
    """
    Solve system of SIR ODEs

    Parameters
    ----------
    I0: int
        Initial number of infected people
    tf: float
        Final integration time
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
        lambda t, y: SIR_mortality(t, y, R0, m if m is not None else 0.0), (0, tf), y0, rtol=1e-12, atol=1e-12
    )

    # Check conservation of people
    assert np.allclose(sir.y.sum(axis=0), S0 + I0)

    return sir


def plot_sir(I0, tf, R0, m=None):
    """
    Plot SIR model

    Parameters
    ----------
    I0: int
        Initial number of infected people
    tf: float
        Final integration time
    R0: double
        Basic reproductive ratio
    m: double (optional)
        Mortality rate
    """

    sir = solve_SIR(I0, tf, R0, m)

    plt.figure()

    plt.plot(sir.t, sir.y[0], label="Susceptible")
    plt.plot(sir.t, sir.y[1], label="Infected")
    plt.plot(sir.t, sir.y[2], label="Recovered")

    if m is not None:
        plt.plot(sir.t, sir.y[3], label="Dead")
    plt.legend()

    plt.title(f"SIR Model ($R_0$  = {R0:.2f} {f', m = {m}' if m is not None else ''})")
    plt.xlabel("Time")
    plt.ylabel("Fraction of Population")

    tag = "simple" if m is None else "mortality"
    plt.savefig(f"plots/SIR_{tag}.pdf")


def plot_infected(I0, tf, R0, m=None):
    """
    Plot fraction of infected people (SIR model) for different values of R0

    Parameters
    ----------
    I0: int
        Initial number of infected people
    tf: float
        Final integration time
    R0: List[double]
        Basic reproductive ratio
    m: double
        Mortality rate
    """

    plt.figure()

    for r0 in R0:
        sir = solve_SIR(I0, tf, r0, m)

        plt.plot(sir.t, sir.y[1], label=f"$R_0$  = {r0:.2f}")

    plt.legend()
    plt.title(f"SIR Model (Infected)")
    plt.xlabel("Time")
    plt.ylabel("Fraction of Infected Population")

    tag = "simple" if m is None else "mortality"
    plt.savefig(f"plots/SIR_{tag}_R.pdf")


# Number of infected people in the UK (13/03/2020)
i0 = 797

# No mortality
plot_sir(I0=i0, tf=60, R0=2)

# 2.5% mortality
plot_sir(I0=i0, tf=60, R0=2, m=0.025)

# No mortality
plot_infected(I0=i0, tf=60, R0=[1.5, 2.0, 2.5, 3.0, 3.5])

# 2.5% mortality
plot_infected(I0=i0, tf=60, R0=[1.5, 2.0, 2.5, 3.0, 3.5], m=0.025)

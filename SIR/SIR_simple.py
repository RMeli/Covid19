import numpy as np

from scipy import integrate

from matplotlib import pyplot as plt
import seaborn as sns

N = 68e6


def f(t, y, R0, beta=1):
    S, I, R = y[0], y[1], y[2]

    gamma = beta / R0

    dSdt = -beta / N * I * S
    dIdt = beta / N * I * S - gamma * I
    dRdt = gamma * I

    return np.array([dSdt, dIdt, dRdt])


def solve_SIR(S0, I0, tf, R0):

    y0 = np.array([S0, I0, tf])

    sir = integrate.solve_ivp(lambda t, y: f(t, y, R0), (0, 60), y0, rtol=1e-12, atol=1e-12)

    return sir


def plot_sir(S0, I0, tf, R0):

    sir = solve_SIR(S0, I0, tf, R0)

    # Check conservation of people
    assert np.allclose(sir.y.sum(axis=0), S0 + I0)

    plt.figure()
    plt.plot(sir.t, sir.y[0], label="Susceptible")
    plt.plot(sir.t, sir.y[1], label="Infected")
    plt.plot(sir.t, sir.y[2], label="Recovered")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Number of People")
    plt.savefig("plots/SIR_simple.pdf")


def plot_infected(S0, I0, tf, R0):

    plt.figure()

    for r0 in R0:
        sir = solve_SIR(S0, I0, tf, r0)

        plt.plot(sir.t, sir.y[1], label=f"$R_0$  = {r0:.2f}")

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Number of Infected People")
    plt.savefig("plots/SIR_simple_R.pdf")


plot_sir(S0=N, I0=700, tf=60, R0=2.5)

plot_infected(S0=N, I0=700, tf=60, R0=[1.5, 2.0, 2.5, 3.0, 3.5])